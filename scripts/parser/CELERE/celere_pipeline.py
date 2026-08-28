#!/usr/bin/env python3
"""
CELERE performance model and GVSOC pipeline app generator.

Reads the model's layer template (csv/xlsx/ods), costs it against a JSON hardware config, and
emits one multi-cluster app per (mode, pool) under
software/apps/multi_cluster/celere_<mode>_<pool>/main.c.

    python celere_pipeline.py CELERE_273PRBs_64ant.csv config.json
    python celere_pipeline.py CELERE_273PRBs_64ant.csv config.json --mode ctt --iterations 8
    python celere_pipeline.py CELERE_273PRBs_64ant.csv config.json --wait --plot

    ctt  compute-then-transfer: one L1 buffer set, compute and DMA serialized.  Stage s is
         skewed by s steps, so it works on item m at step m + s.
    cwt  compute-while-transfer: double-buffered L1 (2x X_BUF), compute and DMA overlapped.
         Stage s is skewed by 2s steps, which is what leaves a spare step for the transfer.

SCOPE.  THIS PARTITION IS TUNED FOR ctt ON TENSORPOOL, and that is the only configuration whose
widths, buffer bindings and LPDDR layout were actually fitted -- 0.159 ms of period against
Terapool's 0.787, and 2803 KiB of L1 against a 3948 KiB budget.  The other three apps are
generated and modelled but are a starting point, not a result: Terapool runs its GEMMs on the
PE path and would want different widths, and CWT DOES NOT FIT EITHER POOL -- doubling the whole
X_BUF puts it 943 KiB over on TensorPool and 1355 KiB over on Terapool, so those two apps will
not link until `tiles` on the last stage is raised, or the doubling is made partial the way
cevit_pipeline.py does it.

THE PARTITION IS GIVEN, NOT SEARCHED.  It lives in STAGES / RESIDENT / BIND below: 5 stages of
2 8 8 8 4 clusters on a 4-wide x 8-tall grid, 30 working plus 2 idle.  Row numbers are
1-indexed csv.reader RECORDS, not .csv line numbers -- description cells contain newlines.

    stage 1  x2  split i_3   rows 5, 7, 8      front end, south LPDDR edge
    stage 2  x8  split i_3   rows 11-31        one whole encoder pass
    stage 3  x8  split i_3   rows 11-31        the same pass, weight set 1
    stage 4  x8  split i_2   rows 34-43        FFN, R3 base and decode, south edge
    stage 5  x4  split i_3   rows 45-48        head, tiled x2, north edge

AN ENCODER PASS IS ONE STAGE.  Every axis it splits -- i_3 = 16 in the projection, i_1 = 16 in
the dual-global pass, i_1 = 176 in the local pass, i_1 = 2400 in the skip block -- is a
multiple of the 16 box columns, and with spatial_fft = False no attention pass ever mixes two
columns, so rows 11-31 run with ZERO cluster-to-cluster traffic.  Turning spatial_fft on or
adding an ('attn','spatial',...) schedule entry breaks that, and build_stages counts elements,
not dependencies, so it would not catch it -- re-read column E of rows 15-30 first.

THE TWO REPETITIONS ARE UNROLLED onto stages 2 and 3, not looped: a backward edge would make
the period the sum of both passes instead of the max over stages, and what it costs is one
weight set of L1 per encoder cluster.  The template's FLOPs column already carries the
xN_repetitions factor, so each placement is charged flops / n_rep and build_stages refuses a
table that does not place a row exactly n_rep times.

STAGE 1 DECLARES pad=2, leaving cids 2-3 idle.  cids are assigned sequentially and the emitted
LPDDR address rule derives a leg's node from `lane`, so a stage touching LPDDR must start on a
column boundary; packed from cid 0, stages 4 and 5 would start at 18 and 26 and the
_Static_asserts would refuse to compile.  The freed clusters cannot go anywhere else: every
stage splits an axis needing a divisor of 16.

    stage 1  cids  0-1   row 0      south edge, 2 nodes per lane
    (idle)   cids  2-3
    stage 2  cids  4-11  rows 1-2
    stage 3  cids 12-19  rows 3-4
    stage 4  cids 20-27  rows 5-6   south edge, 2 lanes per node
    stage 5  cids 28-31  row 7      north edge, 1 node per lane

OP LOWERING
    dense       ONE GEMM per row, never a batch loop: M = prod(i[:-1])/div, N = i[-1],
    matmul      P = o[-1], div = lanes x tiles.  Every leading axis collapses into the row
                count, so X and Y span the whole tensor; W is under-counted, read once rather
                than once per block.  RedMulE on TensorPool (M rounded up to
                POOL_REDMULE_TILES), matmul_4x2_parallel_f16vec on Terapool.
    layernorm   layernorm_parallel_2x4_f16vec, M = prod(i[:-1])/div, N = i[-1]
    softmax     softmax_parallel_2x4_f16vec, same shape rule
    gelu        gelu_f16, in place on Y
    add         axpy_f16vecp_local_unrolled4(0x3C00, X, Y, len)   (0x3C00 = fp16 1.0)
    rope, pool  mempool_wait() -- no such kernel exists in software/kernels
    layout      mempool_wait(), sized by the PE layout fit

Every stand-in keeps the rendezvous the kernel it replaces would have ended on, so the cores
cannot drift apart.

LAYOUT ROWS.  Of the template's twelve, four (DROPPED_ROWS) ride a DMA -- row 3 the inbound
LPDDR read, rows 9, 33 and 44 the hand-offs out of stages 1, 3 and 4, which is in each case the
boundary's reason for existing -- and the other eight stay on their stage as mempool_wait()
spins.  THIS IS A TIMING MODEL, NOT A LOWERING: no permutation kernel is emitted, so those rows
move no bytes, and PE_LAYOUT_FLOPS_PER_ELEM was fitted on patchify_f16, a strided gather, so it
over-counts a contiguous permutation.  Re-fit with --bench before leaning on those numbers.

BUFFERS.  Stages 1-4 do not run a chain -- rows read operands produced many rows earlier -- so
each pins NAMED REGIONS inside X_BUF that stay live for the whole stage.  RESIDENT says which
regions and how big, BIND which region each row reads and writes ('' = the scratch arena, read
at its base and written one padded input past it), SCRATCH_OVER which inbound tensors the arena
may sit on top of.  A row naming the SAME region twice is a view, not a copy.  _stage_layout is
the single authority on every offset inside X_BUF.  W_BUF gives every row its own region and is
sized to their SUM: a stage's weights are resident, re-read on every item.

HAND-OFFS.  Every boundary carries ONE tensor, because an encoder pass is self-contained and
the only thing that has to cross between two of them is Z.  Every cluster stores its share
box-outer, so a lane's box columns are one contiguous run.  Two legs also PERFORM a layout
row on the way -- the lattice scatter into stage 2 and the observation gather into stage 4 --
as a table of segments; the other two are plain copies.

LPDDR.  Five legs cross the chip boundary, four of them on the south edge: the model input in
and the WIN summaries out on stage 1, both back in as the R3 base on stage 4, and the channel
estimate out on the north edge on stage 5.  THE NODE, NOT THE LANE, IS THE UNIT -- a node's
contents do not depend on who reads them, which is the only reason the 2-lane writer and the
8-lane reader of the south edge can name the same bytes.  _leg_fanout derives whole nodes per
lane or lanes per node from the width, and a shared channel is charged at bandwidth/lanes.
DISTANCE IS NOT CHARGED: stage 4's lane k reaches sideways to node k/2 and stage 5 drains from
the far row, so re-measure against RTL before treating the hops as free.  hbm_chan_placement is
mirrored in HBM_CHAN_PLACEMENT because the generated header spells it as a Python list literal
no C can read; the two are kept in step by hand.

TILING.  Row 48 on stage 5 runs in two tiles: untiled it would hold 736,512 elements live.  A
tiled stage drains itself, so its transfers are serialised inside its own compute and land in
`latency` / `drain_ms` rather than the transfer column.  Only the last stage may be tiled, and
raising `tiles` is the one-line fix when the cwt build's doubled X_BUF does not fit L1.

CONFIG.  PE_peak / GEMM_peak in TFLOPS, PE_utilization deriving the PE path, TE_utilization
deriving GEMM_peak on a pool with a tensor engine, bandwidth_c2c_gbs and bandwidth_lpddr_gbs in
GB/s, lpddr_channels, frequency in Hz.  An off-chip leg is charged at
bandwidth_lpddr_gbs * lpddr_channels capped by bandwidth_c2c_gbs (see _lpddr_bw);
pipeline_clusters is only a cross-check against the stage table.
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Header fields located by label; everything else is positional (DEFAULT_COLS).  Both
# the "Data in" and the "Data_in" spelling of the traffic columns are accepted.
HEADER_ALIASES: Dict[str, List[str]] = {
    'n_rep':   ['N_repetitions', 'N_rep', 'Repetitions'],
    'kernel':  ['Kernel'],
    'flops':   ['FLOPs', 'Flops', 'FLOPS'],
    'weights': ['Weights'],
    'din':     ['Data in', 'Data_in'],
    'dout':    ['Data out', 'Data_out'],
}

# Block name and operation carry no header in the template.
DEFAULT_COLS = {'block': 0, 'op': 2}

# Optional tensor-shape columns; real-kernel mode needs them, --wait does not.
DIM_HEADERS = {
    'in_1': 'i_1', 'in_2': 'i_2', 'in_3': 'i_3', 'in_4': 'i_4',
    'out_1': 'o_1', 'out_2': 'o_2', 'out_3': 'o_3', 'out_4': 'o_4',
}

BYTES_PER_ELEM = 2                       # FP16

# TCDM / NoC line width, in elements.  A DMA source that is not 64 B aligned emits lines
# astride a destination page, each of which the FlooNoC interface splits into two network
# transactions: the leg then runs at exactly half rate.  X_BUF is 8 KiB aligned, so the
# OFFSET is the whole story.
DMA_LINE_ELEMS = 64 // BYTES_PER_ELEM

GEMM_KERNELS = {'dense', 'matmul', 'conv1d'}
LAYOUT_KERNELS = {'reshape', 'split', 'split&reshape', 'transpose'}

# Kernels with no baremetal implementation in software/kernels.  They are emitted as
# mempool_wait() spins sized to the modelled latency; the string is the reason printed
# into the generated source.
NO_KERNEL = {
    'rope': 'no rope kernel exists',
}

TENSOR_ENGINE_POOLS = {'TensorPool'}     # picks TE_utilization and the RedMulE emitter

# RedMulE tiles per cluster, matching nb_redmule_tiles in the arch config and the
# -DNUM_REDMULE_TILES the build passes.  redmule_synch_parallel splits a GEMM's rows
# across them with an integer divide, so every emitted M is rounded up to a multiple of
# it; see _gemm_rows.
# emitted M is rounded up to a multiple of it; see _gemm_rows.
POOL_REDMULE_TILES = {'TensorPool': 16}

# LPDDR nodes per edge -- hbm_chan_placement in the arch configs, [0,4,0,4] on both
# arch_tensorpool256.py and arch_terapool.py.  gen_flex_arch.py passes that through as a
# Python list literal, which no C can read, so this constant is the app's only view of the
# placement and the two are kept in step by hand.  WEST AND EAST HOLD NOTHING, so
# hbm_addr() -- west node 0 -- must never appear in a generated app.  The node SIZE is not
# here: the emitted app reads ARCH_HBM_NODE_ADDR_SPACE out of flex_cluster_arch.h.
HBM_CHAN_PLACEMENT = {'west': 0, 'north': 4, 'east': 0, 'south': 4}

# The edge a leg uses when its Tensor does not name one.  A Tensor may override it: stage
# 1's WIN spill goes SOUTH although it is an output, and stage 4 reads the R3 base off the
# SOUTH edge although the pipeline drains north.
LPDDR_IN_EDGE = 'south'
LPDDR_OUT_EDGE = 'north'

# A permutation kernel moves ONE ELEMENT PER CORE per this many cycles, counted off the
# assembly clang emits at -O3 for the mempool target.  This replaces a FLOPs-per-element
# fit that was quoted against a `patchify_f16` kernel which did not exist in
# software/kernels/baremetal; the kernels named here do.
#
#   permute   permute_f16p walks a level table with an asm inner run -- 8 p.lhu, 8 p.sh,
#             a decrement and a branch, 18 instructions per 8 elements.  Every CELERE row
#             lowered to it has a run of 48 or more, so the level counters and carries are
#             under a tenth of the work and do not appear here.
#
# IT IS A FLOOR.  It carries neither the load-use stall nor the bank conflicts a scattered
# access pattern provokes, both of which push the real figure up.  Re-fit from an RTL run
# with --bench; the model is optimistic until you do.
LAYOUT_CYCLES_PER_ELEM: Dict[str, float] = {
    'permute': 2.25,
}

# ---- Where each of the template's layout rows is actually performed ---------
#
# THREE OUTCOMES, NOT TWO, and the difference is the whole discipline here.  A layout row
# carries 0 FLOPs, so the template can say nothing about what it costs; what it costs
# depends entirely on which engine performs it, and one of the three engines is "none".
#
# FREE_ROWS -- a VIEW.  The operand is already in the order its consumer wants, so nothing
# runs and nothing moves.  This is not the same as being carried by a transfer: there is no
# transfer.  Charging one of these anything at all is OVER-counting, which is how they were
# costed before (a mempool_wait each).
#
#   row 13   the q|k|v split.  The template is explicit -- "Zero-copy slice; no kernel.
#            Three contiguous sub-ranges at stride o_4".  Row 15 reads it strided and
#            materialises what the GEMM needs; see below.
#   row 23   UF.  U leaves row 20 in window-major order, which puts each window's spans
#            beside its own summary, so a fold is a CONTIGUOUS [L_l, d_head] sub-block and
#            row 27 can contract against it where it lies.  Verified against
#            redmule_synch_parallel, which addresses X as a dense [M, N] and has no
#            leading-dimension argument -- that is the test for every claim of this kind
#            here, and it is what rows 15 and 22 FAIL.
#   row 41   the R3 base.  Its two source regions are PAT and WIN, which arrive as two
#            LPDDR legs, and _stage_layout lands inbound tensors back to back -- so the
#            concatenation is the layout, and there is nothing left to do.
FREE_ROWS = (13, 23, 41)

# DROPPED_ROWS -- carried by a MOVE that was happening anyway, so the row is the address
# pattern of a descriptor and costs no extra cycles.  The DMA is priced PER RUN, not per
# byte -- one burst per repetition, one TCDM line per cycle, a line being
# cores * banking * 4 bytes (8 KiB here) -- so this is only free when the runs are long:
#
#   row 9    stage 1 -> 2 entry.  Interleaves the window summaries among the spans while
#            assembling the lattice: row 7 embedded PAT and WIN CONCATENATED, row 9 wants
#            them window-major.  Four rectangles of whole token rows -- see
#            _handoff_segments -- and the spell row it also inserts has no source, so what
#            the leg carries is the 149 rows and the consumer's 150th is the pitch.
#   row 29   NOTHING MOVES.  The residual it loads IS the lattice the stage already holds:
#            stages 2 and 3 take Z as their inbound tensor and row 30 reads it where it
#            landed, so this is a second reader of a resident region, not a copy.  It is
#            here rather than in FREE_ROWS because the row names a load the model draws as
#            an edge; the app has nothing to issue for it either way.
#   row 33   stage 3 -> 4 entry.  Drops the spell row -- row 9 put it LAST in every box
#            column -- so the crop is the PITCH DIFFERENCE between the two ends, one
#            rectangle per column.  It is not a plain copy only because a lane holds more
#            than one column, and the gap sits inside its share.
DROPPED_ROWS = (9, 29, 33)

# LAYOUT_LOWERING -- a real PE kernel runs it, because no move is going that way or because
# a descriptor would be dearer than the cores:
#
#   row 3    RA unpack + patchify.  Composes three factorizations; six levels, run = 48.
#            On-chip by ruling: the inbound leg carries the RAW GRID and this patchifies it
#            after it lands, rather than assuming DRAM was written pre-grouped.
#   row 15   the head gather.  Row 13 leaves Q, K and V strided by 3*d_model inside QKV;
#            rows 17 and 20 are GEMMs and RedMulE takes a DENSE operand, so QG, KG and UG
#            have to exist.  Run = d_head = 128.
#   row 22   the local fold.  Re-folds Q and K FROM ROW 13, not from the rotated QG/KG --
#            the local pass takes pb_fine instead of rope, so it needs the unrotated pair.
#            Same strided source, same dense-operand requirement.  Run = 128.
#   row 28   the head merge, as TWO static folds (n_full of L_l, one of L_rem).  Run = 128.
#   row 40   the span-major restore.  In-stage on stage 4: its input is row 38's output,
#            which is computed there, so no entry transfer can carry it.  Run = 256.
#   row 44   the per-PRB re-pack.  This one sits at the 4 -> 5 boundary and STILL goes to
#            the cores, which is the clearest case of the per-run rule: its runs are 48
#            halfwords, so a descriptor would spend 19,008 line-cycles moving 1/170th of a
#            line each time, against about 2,000 cycles on the cores.
LAYOUT_LOWERING: Dict[int, str] = {3: 'permute', 15: 'permute', 22: 'permute',
                                   28: 'permute', 40: 'permute', 44: 'permute'}

# A layout row whose op names MORE THAN ONE TENSOR moves each of them: row 15 gathers Q, K
# and U, row 22 folds Q and K, and the template's out dims describe ONE of them.  The count
# multiplies both the calls _permute_levels returns and the output _row_shape charges, so
# neither the emitted code nor the cost model sees only the first.
PERMUTE_TENSORS: Dict[int, int] = {15: 3, 22: 2}


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------
#
# One entry per stage, in order.  `rows` is the stage's ordered work as template row
# numbers, `n_par` how many clusters run it in parallel, `tiles` how many passes one
# cluster makes over the trailing run of items starting at `tile_from`, `pad` how many cids
# it leaves IDLE after itself, and `split` the axis the lanes divide (documentation only).

# One whole encoder pass, in template order: the projection, the merged dual-global
# attention, the local spectral pass and the skip block.  Stages 2 and 3 each run all of it.
# Row 29 is NOT here: the residual load is a MOVE, and a strided one, so it rides a
# descriptor and overlaps row 30's GEMM exactly as the template says it should.  See
# DROPPED_ROWS.  Rows 13 and 23 stay in the list although they are FREE_ROWS -- they are
# real steps of the network and belong in the ledger; they simply cost nothing and emit
# nothing.
_ENCODER_PASS = [11, 12, 13, 15, 16, 17, 18, 19, 20,   # LN, W_qkv, split, gather, rope,
                 22, 23,                               #   scores, mask, softmax, A x V
                 24, 25, 26, 27, 28, 30, 31]           # local fold + pass, then W_o + skip

STAGES: List[dict] = [
    # The front end.  ROW 3 IS HERE, not on the inbound leg: the RA unpack composes three
    # factorizations and bottoms out at a 48-halfword run, so the leg carries the RAW GRID
    # and permute_f16p patchifies it after it lands.  Rows 5, 7 and 8 come to ~0.091 ms even at two lanes on TensorPool,
    # still under the ~0.135 ms period.  pad=2 leaves cids 2-3 idle so that stages 4 and 5
    # start on a column boundary; nothing else can take them, since every other stage
    # splits an axis needing a divisor of 16 or of 2384 = 149 x 16.
    dict(n_par=2, pad=2, tiles=1, tile_from=0, split='i_3 (16 -> 8 spatial patches)',
         rows=[3, 5, 7, 8]),
    # Encoder pass 0, whole, on 8 lanes -- every row is cluster-local over this lane's 2
    # box columns.  The lattice arrives already assembled: row 9 rides the inbound leg.
    dict(n_par=8, tiles=1, tile_from=0, split='i_3 (16 -> 2 spatial patches)',
         rows=list(_ENCODER_PASS)),
    # Encoder pass 1.  Identical rows, shapes and buffer layout; only the weight set
    # differs.  The two stages take their lattice from different places (row 9's output
    # into the first, row 31's into the second) but both read it out of the same slot.
    dict(n_par=8, tiles=1, tile_from=0, split='i_3 (16 -> 2 spatial patches)',
         rows=list(_ENCODER_PASS)),
    # The block FFN, the span-major restore, the R3 base and the decode.  Row 33's gather
    # (dropped) has taken the stream onto i_2 = 2384 observation carriers; 298 per lane is
    # 149 spans x 2 box columns, so the split stays box-aligned.  Rows 40-43 are here
    # rather than on the head because the R3 base is on LPDDR and this stage reads it back,
    # which saves carrying 915,456 elements across a boundary.  Eight lanes: at four this
    # stage is the bottleneck at 0.188 ms, at eight it is 0.095.
    dict(n_par=8, tiles=1, tile_from=0, split='i_2 (2384 -> 298 = 2 box columns)',
         rows=[34, 35, 36, 37, 38, 40, 41, 42, 43, 44]),
    # The head.  Four lanes, each draining to its own node on the 4-wide north edge.  Row
    # 44's per-PRB blocking is done on the PRODUCER side, at the tail of stage 4, so the
    # hand-off carries the already-packed tensor: its runs are 48 halfwords, and a
    # descriptor doing the re-pack would pay a TCDM line-cycle for each of 19,008 of them
    # against about 2,000 cycles on the cores.  Row 48 is TILED: at
    # four lanes it would otherwise hold 736,512 elements live.
    dict(n_par=4, tiles=2, tile_from=3, split='i_3 (16 -> 4 spatial patches)',
         rows=[45, 46, 47, 48]),
]


# ---------------------------------------------------------------------------
# Hand-offs
# ---------------------------------------------------------------------------
#
# One entry per tensor crossing a stage boundary, in the order the producer lays it out --
# and so the consumer.  `elems` is the WHOLE tensor across every lane of the producing
# stage, so a lane owns elems/lanes of it and a consumer needs elems/lanes_next.

@dataclass(frozen=True)
class Tensor:
    name: str
    elems: int
    # A LAYOUT ROW THIS LEG PERFORMS, by name in _handoff_segments.  '' is a plain copy,
    # which is every leg whose two ends agree on the order of the token rows inside a box
    # column.  Naming one replaces the leg's single descriptor with that row's segments.
    scatter: str = ''
    # LPDDR legs only.  '' takes the direction's default edge; naming one overrides it,
    # which is how stage 1's WIN spill and stage 4's R3 base read both go SOUTH.
    edge: str = ''


# stage index (0-based) -> the tensor(s) it hands to the next stage.  EVERY CLUSTER STORES
# ITS SHARE BOX-OUTER -- a lane's box columns are one contiguous run -- which is what lets a
# leg address one column at a time and lets a plain copy be a plain copy.
_LATTICE = 150 * 16 * 256         # 614,400 -- 150 lattice rows x 16 box columns x d_model

HANDOFF: Dict[int, List[Tensor]] = {
    # Row 9 (dropped) rides this leg: the scatter that assembles the [150, 16] lattice out
    # of the 149 embedded token rows.  2 lanes to 8, fan-out.  Declared LARGER than the
    # producer holds -- row 9 reads 610,304 elements and writes 614,400, the extra 4,096
    # being the learned spell row, one per box column, which no DMA can bring from stage 1
    # -- so that the rule "a stage DMAs the next stage's Data_in and no more" still holds.
    0: [Tensor('Z lattice', _LATTICE, scatter='row9_lattice')],
    # Row 31's Z, back on the lattice.  8 to 8, straight copy.  The ONLY thing the two
    # encoder passes share: pass 1 reads it at its row 11 and again at its row 29.
    1: [Tensor('Z lattice', _LATTICE)],
    # Row 33 (dropped) rides this leg: the gather that drops the spell row the FFN does
    # not see, leaving 2384 observation carriers.  8 to 8, and NOT a straight copy -- the
    # dropped row sits at the end of each of a lane's box columns, so the crop is the
    # pitch difference between the two ends.
    2: [Tensor('ZO observation carriers', 2384 * 256, scatter='row33_crop')],
    # Row 44 ran on the PRODUCER's cores, at the tail of stage 4 -- its runs are 48
    # halfwords, far too short for a descriptor -- so this leg carries the already-packed
    # tensor and is a plain copy.  8 to 4, fan-in.  What is declared is row 44's output,
    # which is exactly row 45's Data_in.
    3: [Tensor('per-PRB blocks', 1188 * 16 * 48)],
}

# The legs that cross the chip boundary.  THE NODE IS THE UNIT, not the lane: what a node
# holds does not depend on who reads it, which is the only reason the 2-lane writer and
# the 8-lane reader of the south edge can be talked about together.  So elems / edge_nodes
# is a node's contents, and _leg_fanout derives the rest from the stage's width:
#
#   n_par <= edge_nodes:  each lane takes edge_nodes/n_par whole nodes, one channel each
#   n_par >  edge_nodes:  n_par/edge_nodes lanes share a node, and that leg is charged at
#                         bandwidth/lanes (_lpddr_leg_ms)
#
#   south node j (j = 0..3), holding box columns 4j..4j+3:
#       0        model input  210,432 elements   (pre-split 4 ways off-chip)
#       210,432  WIN           18,432 elements   (stage 1 spills it back)
#                             228,864 elements   -- what row 41 concatenates
#
# The model input's per-node split is a premise about the OFF-CHIP LAYOUT, and it is what
# lets row 3 be dropped: whoever writes the grid to DRAM writes each south node the quarter
# of the patch lattice that consumes it, already transposed, patch-grouped and box-split.
_PAT = 'model input (patch-grouped, pre-split per node)'
_WIN = 'WIN window summaries (R3 base spill)'

LPDDR_IN: Dict[int, List[Tensor]] = {
    0: [Tensor(_PAT, 137 * 16 * 384)],
    # The two halves of row 41's R3 base, read back by the stage that consumes them, under
    # the SAME NAMES stage 1 used: _lpddr_layout keys node offsets by name, so this stage
    # reads exactly where the front end wrote and neither side states the layout twice.
    3: [Tensor(_PAT, 137 * 16 * 384),
        Tensor(_WIN, 12 * 16 * 384, edge='south')],
}
LPDDR_OUT: Dict[int, List[Tensor]] = {
    # Row 5's window summaries, spilled to the same south node this lane read its input
    # from and directly after it, which is what makes [input | WIN] exactly the R3 base.
    0: [Tensor(_WIN, 12 * 16 * 384, edge='south')],
    4: [Tensor('channel estimate', 1092 * 16 * 336)],
}


# ---------------------------------------------------------------------------
# The distributed Q/K/V/Z buffer
# ---------------------------------------------------------------------------
#
# Three tables saying where every operand of stages 1-4 lives.  _stage_layout turns them
# into offsets and is the only thing that computes one.

@dataclass(frozen=True)
class Region:
    """One pinned region of X_BUF, live for a whole stage.

    `elems` is the WHOLE-STAGE element count, like Tensor.elems -- _stage_layout divides it
    by the stage's lane count.  A region whose name matches an INBOUND tensor rides that
    tensor's landing slot rather than getting space of its own.
    """
    name: str
    elems: int


# stage index -> the regions it pins.  Stage 5 is a chain and has no entry: every row of it
# uses the scratch arena.
RESIDENT: Dict[int, List[Region]] = {
    # WIN lands directly after the inbound model input, whose per-node share is a multiple
    # of DMA_LINE_ELEMS, so the two sit flush.  Row 7's stem dense reads [model input | WIN]
    # as one 149-row stream, and the same two regions in the same order are what stage 1
    # spills to its south nodes and stage 4 reads back as BSE.  One layout, three users.
    0: [Region('WIN', 12 * 16 * 384)],
    # An encoder pass.  The inbound hand-off delivers Z and rows 11, 29 and 31 work in it;
    # row 12 writes the widened Q|K|V block that rows 13, 15, 16, 17, 22 and 24 all view
    # (QF/KF are a strided view of Q and K, not a copy); row 20 writes U, which row 23 views
    # as the local pass's V and row 30 reads the spell rows out of.  On stage 3 'Z lattice'
    # is also the inbound tensor's name, so it rides that landing slot.
    1: [Region('Z lattice', _LATTICE),
        Region('QKV', 150 * 16 * 768),
        Region('U', 16 * 2 * 150 * 128)],
    2: [Region('Z lattice', _LATTICE),
        Region('QKV', 150 * 16 * 768),
        Region('U', 16 * 2 * 150 * 128)],
    # Row 41 concatenates the two halves of the R3 base into BSE, which row 43 reads after
    # row 42's GEMM has overwritten the arena -- so it cannot live there.
    3: [Region('BSE R3 base', 149 * 16 * 384)],
}

# (stage index, template row) -> (region the row READS, region it WRITES).  '' on either
# side means the scratch arena: read at its base, write one padded input past it, which is
# the chain behaviour every unlisted row keeps.  A row naming the SAME region twice is a
# view, not a copy: the spin still runs, but the layout reserves no room for a second tensor.
BIND: Dict[Tuple[int, int], Tuple[str, str]] = {}


def _bind_encoder(stage: int) -> None:
    """One whole encoder pass, rows 11-31, on one stage."""
    BIND.update({
        (stage, 11): ('Z lattice', ''),      # layernorm reads the pinned lattice
        (stage, 12): ('', 'QKV'),            # W_qkv widens into the pinned Q|K|V block
        (stage, 13): ('QKV', 'QKV'),         # split: three views of one buffer
        (stage, 15): ('QKV', 'QKV'),         # gather into the head fold, in place
        (stage, 16): ('QKV', 'QKV'),         # rope rotates Q and K in place
        (stage, 17): ('QKV', ''),            # QK^T reads Q and K where they are
        (stage, 20): ('', 'U'),              # A x V -> U
        # The local pass reads the SAME Q, K and U through a different stride: row 22 is a
        # strided fold of Q and K and row 23 a strided view of U, so both name one region
        # twice -- the fold produces no tensor, it only re-addresses one.
        (stage, 22): ('QKV', 'QKV'),
        (stage, 23): ('U', 'U'),
        (stage, 24): ('QKV', ''),            # local scores read the folded Q and K
        (stage, 29): ('Z lattice', 'Z lattice'),  # residual load: a regroup, not a copy
        (stage, 31): ('', 'Z lattice'),           # Z += WO, written back in place
    })


_bind_encoder(1)
_bind_encoder(2)

BIND.update({
    # -- the front end.  The pool reads the inbound model input and writes the WIN region
    # pinned directly after it; row 7 then reads BOTH as one 149-row stream, which is why
    # it starts at the model input's base rather than the arena's.
    (0, 5): (_PAT, 'WIN'),
    (0, 7): (_PAT, ''),
    # -- stages 4 and 5 are chains, but their first row still has to name its inbound
    # tensor: the arena does not start at 0 on a stage with live inbound regions.
    (3, 34): ('ZO observation carriers', ''),
    (4, 45): ('per-PRB blocks', ''),
    # Row 41 reads the two inbound, adjacent halves of the R3 base as one 149-row stream
    # and writes the pinned region row 43 will read.
    (3, 41): (_PAT, 'BSE R3 base'),
    # Row 43 is the R3 residual: axpy adds that base into the decode output.
    (3, 43): ('BSE R3 base', ''),
})

# (stage index, outbound tensor name) -> (region it is read FROM, element offset of the
# tensor inside it), for tensors whose source is NOT simply the last item's output.  Every
# hand-off in this partition is what the stage's last row wrote, so only the WIN spill
# needs an entry.
OUTBOUND_AT: Dict[Tuple[int, str], Tuple[str, int]] = {
    # The WIN spill: row 5's output, still in its pinned region at the end of the stage
    # (the front end's last row is 7, which writes the arena).
    (0, 'WIN window summaries (R3 base spill)'): ('WIN', 0),
}

# stage index -> inbound tensors the scratch arena may be placed ON TOP OF, because they
# are dead before the first row that uses the arena runs.  The encoder stages have no
# entry: their one inbound tensor IS the pinned Z lattice, live from arrival to the
# hand-off out, and stage 4's model input and WIN are read seven rows in.
SCRATCH_OVER: Dict[int, List[str]] = {
    3: ['ZO observation carriers'],       # dead after row 34 normalizes it
    4: ['per-PRB blocks'],                # dead after row 45 reads it
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerRow:
    """One kernel row of the template."""
    excel_row: int
    block: str
    n_rep: int
    op: str
    kernel: str
    flops: float
    weights_bytes: float
    din_bytes: float
    dout_bytes: float
    is_gemm: bool
    in_dims: List[int] = field(default_factory=list)
    out_dims: List[int] = field(default_factory=list)


@dataclass
class Item:
    """One unit of work on a stage: one placement of one template row.

    `tiles` is the item's OWN pass count -- the stage's `tiles` for an item inside the
    tiled run, 1 for everything before it.  Every shape and footprint below divides by
    n_par * item.tiles, so a partly tiled stage costs exactly what it runs.
    """
    row: LayerRow
    label: str = ''
    excel_row: int = 0           # the row this item is costed against
    tiles: int = 1
    ms: float = 0.0              # latency of ONE cluster's ONE tile of this item


@dataclass
class Stage:
    idx: int
    n_par: int
    tiles: int
    tile_from: int               # index into items of the first tiled item
    split: str
    items: List[Item]
    pad: int = 0                 # cids left IDLE after this stage; see _stage_bases
    latency: float = 0.0         # compute [ms] of ONE cluster, ALL tiles
    transfer_ms: float = 0.0     # hand-off + LPDDR, per cluster; 0 on a tiled stage
    drain_ms: float = 0.0        # of `latency`, the part a tiled stage spends draining
    din_ms: float = 0.0          # inbound leg, already inside transfer_ms
    # Element offset of every named region inside this cluster's X_BUF, plus ARENA and
    # '@end'.  Filled by compute() from _stage_layout, and the authority every emitter
    # reads: see THE DISTRIBUTED Q/K/V/Z BUFFER.
    layout: Dict[str, int] = field(default_factory=dict)
    # Element offset of each item's operand-B / parameter region inside W_BUF, keyed by
    # position in `items`; key -1 holds one past the end.  See _stage_w_layout.
    w_layout: Dict[int, int] = field(default_factory=dict)
    x_elems: int = 0             # per-cluster buffer needs (static X/W/Y declaration)
    w_elems: int = 0
    y_elems: int = 0
    kernel_elems: int = 0        # per-cluster max over layers of (in + out) + weights
    din_bytes: float = 0.0       # per cluster
    dout_bytes: float = 0.0      # per cluster


@dataclass
class Results:
    flops_per_ms: Dict[str, Dict[str, float]]
    row_latencies: Dict[str, Dict[int, float]]
    total_latency: Dict[str, float]
    stages: Dict[str, List[Stage]]
    best_throughput: Dict[str, float]
    best_throughput_overlap: Dict[str, float]


# ---------------------------------------------------------------------------
# Cell value helpers
# ---------------------------------------------------------------------------

EXCEL_ERRORS = ('#NAME?', '#REF!', '#VALUE!', '#DIV/0!', '#N/A', '#NULL!', '#NUM!')


def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and x != x:      # NaN
        return True
    return isinstance(x, str) and not x.strip()


def _is_excel_error(x) -> bool:
    return isinstance(x, str) and x.strip().upper() in EXCEL_ERRORS


def _to_str(x) -> str:
    return '' if _is_blank(x) else str(x).strip()


def _to_float(x) -> float:
    """Numeric cell -> float.  CELERE writes thousands separators ("841,728")."""
    if _is_blank(x):
        return 0.0
    if _is_excel_error(x):
        raise ValueError(f"cell holds the spreadsheet error {x!r}")
    if isinstance(x, str):
        x = x.replace(',', '').replace('%', '').strip()
    return float(x)


def _to_int(x) -> int:
    return int(round(_to_float(x)))


def col_letter(idx: int) -> str:
    s = ''
    idx += 1
    while idx:
        idx, r = divmod(idx - 1, 26)
        s = chr(ord('A') + r) + s
    return s


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def read_grid(path: str) -> List[List]:
    """Read csv / xlsx / xlsm / ods into a row-major 2D list."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == '.csv':
        with p.open(newline='', encoding='utf-8-sig') as fh:
            return [list(r) for r in csv.reader(fh)]
    if ext in ('.xlsx', '.xlsm'):
        from openpyxl import load_workbook
        wb = load_workbook(str(p), data_only=True)
        return [list(r) for r in wb[wb.sheetnames[0]].iter_rows(values_only=True)]
    if ext == '.ods':
        import pandas as pd
        df = pd.read_excel(str(p), engine='odf', header=None)
        return df.values.tolist()
    raise ValueError(f'unsupported template format: {ext}')


def find_header_row(grid: List[List]) -> int:
    """First row labelling every required field."""
    for r, row in enumerate(grid):
        cells = [_to_str(c) for c in row]
        if all(any(a in cells for a in aliases) for aliases in HEADER_ALIASES.values()):
            return r
    raise ValueError(
        'no header row found: expected one row labelling '
        + ', '.join(a[0] for a in HEADER_ALIASES.values()))


def build_column_map(header: List) -> Dict[str, int]:
    cells = [_to_str(c) for c in header]
    cols = dict(DEFAULT_COLS)
    for field_name, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in cells:
                cols[field_name] = cells.index(a)
                break
        else:
            raise ValueError(
                f"header row has no column for {field_name} "
                f"(looked for {aliases}); row is {cells}")
    for key, label in DIM_HEADERS.items():          # optional
        if label in cells:
            cols[key] = cells.index(label)
    return cols


def find_data_end(grid: List[List], cols: Dict[str, int], first: int) -> int:
    """Exclusive end: the totals row that closes the layer table.

    Everything past it -- the parameter block and the prose -- has an empty kernel column
    and is dropped by parse_layers anyway; this only keeps check_for_errors off it.
    """
    for r in range(first, len(grid)):
        row = grid[r]
        op = _to_str(row[cols['op']]) if cols['op'] < len(row) else ''
        if op.upper().startswith('TOTAL'):
            return r
    return len(grid)


def check_for_errors(grid: List[List], cols: Dict[str, int],
                     first: int, last: int) -> None:
    bad = []
    for r in range(first, last):
        for name, c in cols.items():
            if c < len(grid[r]) and _is_excel_error(grid[r][c]):
                bad.append(f'{col_letter(c)}{r + 1} ({name}) = {grid[r][c]!r}')
    if bad:
        raise ValueError('template has unresolved cells:\n  ' + '\n  '.join(bad))


def parse_layers(grid: List[List], cols: Dict[str, int],
                 first: int, last: int) -> List[LayerRow]:
    layers: List[LayerRow] = []
    for r in range(first, last):
        row = grid[r]

        def cell(key: str):
            c = cols.get(key)
            return None if c is None or c >= len(row) else row[c]

        op = _to_str(cell('op'))
        kernel = _to_str(cell('kernel'))
        # A row with no kernel is not a layer.  The section headers and the totals block
        # have an operation and sometimes a FLOPs figure but no kernel, which is what
        # keeps them out of the model.
        if not op or not kernel or _is_blank(cell('flops')):
            continue

        def dims(prefix: str) -> List[int]:
            vals = []
            for k in range(1, 5):
                key = f'{prefix}_{k}'
                if key not in cols:
                    break
                c = cell(key)
                vals.append(None if _is_blank(c) else _to_int(c))
            while vals and vals[-1] is None:
                vals.pop()
            return [v for v in vals if v is not None]

        layers.append(LayerRow(
            excel_row=r + 1,
            block=_to_str(cell('block')),
            n_rep=max(1, _to_int(cell('n_rep'))) if not _is_blank(cell('n_rep')) else 1,
            op=op,
            kernel=kernel,
            flops=_to_float(cell('flops')),
            weights_bytes=_to_float(cell('weights')) * BYTES_PER_ELEM,
            din_bytes=_to_float(cell('din')) * BYTES_PER_ELEM,
            dout_bytes=_to_float(cell('dout')) * BYTES_PER_ELEM,
            is_gemm=kernel.lower() in GEMM_KERNELS,
            in_dims=dims('in'),
            out_dims=dims('out'),
        ))
    return layers


def load_template(path: str) -> List[LayerRow]:
    grid = read_grid(path)
    hdr = find_header_row(grid)
    cols = build_column_map(grid[hdr])
    first = hdr + 1
    last = find_data_end(grid, cols, first)
    check_for_errors(grid, cols, first, last)
    return parse_layers(grid, cols, first, last)


# ---------------------------------------------------------------------------
# Stage assembly
# ---------------------------------------------------------------------------

def _stage_bases(stages: List[Stage]) -> Tuple[List[int], int]:
    """Each stage's first cid, and one past the last cid the pipeline OCCUPIES.

    NOT a running sum of n_par: a stage may leave cids IDLE after it (Stage.pad) so that
    the next one starts on a column boundary.  The emitted tables are indexed by cid, so
    N_CLUSTERS has to be the OCCUPIED total and span the gaps.  THE ONLY PLACE cid BASES
    ARE COMPUTED.
    """
    bases, at = [], 0
    for st in stages:
        bases.append(at)
        at += st.n_par + st.pad
    return bases, at


def build_stages(layers: List[LayerRow]) -> List[Stage]:
    """Bind the STAGES table to the parsed rows, checking the two stay in step."""
    by_row = {r.excel_row: r for r in layers}
    stages: List[Stage] = []
    placed: Dict[int, int] = {}

    for i, spec in enumerate(STAGES):
        items: List[Item] = []
        for n, rn in enumerate(spec['rows']):
            if rn not in by_row:
                raise ValueError(
                    f"stage {i + 1} names template row {rn}, which the parser did not "
                    f"read as a layer (kernel column empty, or outside the data region)")
            row = by_row[rn]
            tiles = spec['tiles'] if n >= spec['tile_from'] else 1
            items.append(Item(row=row, label=row.op, excel_row=rn, tiles=tiles))
            placed[rn] = placed.get(rn, 0) + 1
        stages.append(Stage(idx=i, n_par=spec['n_par'], tiles=spec['tiles'],
                            tile_from=spec['tile_from'], split=spec['split'],
                            items=items, pad=spec.get('pad', 0)))

    # The placement invariant: the FLOPs column already carries the xN_repetitions factor
    # and _row_latency divides each placement by it, so the stage table costs a repeated
    # row correctly exactly when it places it n_rep times.  Catches both halves of the
    # likely mistake -- a row placed once (half its work vanishes) and a row placed on a
    # third stage (n_rep is stale).
    problems = []
    for r in layers:
        want = 0 if r.excel_row in DROPPED_ROWS else r.n_rep
        got = placed.get(r.excel_row, 0)
        if got != want:
            problems.append(
                f'  row {r.excel_row} ({r.kernel} {r.op[:40]!r}): placed {got} time(s), '
                f'N_repetitions = {r.n_rep}'
                + (' (and it is in DROPPED_ROWS)' if r.excel_row in DROPPED_ROWS else ''))
    if problems:
        raise ValueError(
            'the stage table does not place every template row N_repetitions times:\n'
            + '\n'.join(problems)
            + f'\nDROPPED_ROWS = {list(DROPPED_ROWS)} must be placed 0 times.')

    # Only the LAST stage may be tiled: _render_tile_drain knows only the last stage's
    # destination, and main() suppresses the hand-off for a tiled stage on the strength of
    # it being last.
    early = [st.idx + 1 for st in stages[:-1] if st.tiles > 1]
    if early:
        raise ValueError(
            f"stage(s) {early} are tiled but are not the last stage; the per-tile drain "
            f"only knows how to write the last stage's output")
    for st in stages:
        if st.tiles > 1 and st.tile_from >= len(st.items):
            raise ValueError(
                f'stage {st.idx + 1} sets tile_from={st.tile_from} but has only '
                f'{len(st.items)} items; nothing would be tiled')
        if st.tiles == 1 and st.tile_from:
            raise ValueError(
                f'stage {st.idx + 1} sets tile_from={st.tile_from} with tiles=1')
    # Every LPDDR leg has to divide cleanly into nodes -- a lane owning whole nodes, or
    # several lanes sharing one.  Nothing in the C can see HBM_CHAN_PLACEMENT, so the
    # ratio is checked here; _leg_fanout raises on anything that does not divide.
    bases, _total = _stage_bases(stages)
    for key in ('hin', 'hout'):
        for i in _TABLES[key]:
            for t in _tensors(key, i):
                edge = _leg_edge(t, key)
                _leg_fanout(stages[i].n_par, edge, f"stage {i + 1} tensor '{t.name}'")
                # The emitted address rule derives a leg's node from `lane`, which counts
                # from the stage's base cid, so it agrees with the cluster's column only
                # when that base IS a column boundary.  The C _Static_asserts it too, but
                # this says which stage and by how much.  The grid WIDTH is not visible
                # here, so the edge's node count stands in for it -- the arch configs put
                # one node per column on both edges the pipeline uses.
                nodes = HBM_CHAN_PLACEMENT[edge]
                if bases[i] % nodes:
                    raise ValueError(
                        f"stage {i + 1} touches LPDDR but starts at cid {bases[i]}, which "
                        f"is {bases[i] % nodes} past a column boundary ({nodes} nodes on "
                        f"the {edge} edge = one per column).  Give an earlier stage "
                        f"pad={nodes - bases[i] % nodes} to push this one onto a boundary")
    return stages


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _transfer_ms(nbytes: float, bw_gbs: Optional[float]) -> float:
    return 0.0 if not bw_gbs else nbytes / (bw_gbs * 1e6)


def _lpddr_bw(cfg: dict) -> Optional[float]:
    """Bandwidth an L1 <-> LPDDR leg is charged at [GB/s], or None if untimed.

    The DRAM side is worth the CUMULATIVE figure -- per-channel bandwidth times the
    channel count, because the transfer is striped across them -- but the bytes still
    cross the on-chip network to reach the memory-side node, so the leg is CAPPED BY
    bandwidth_c2c_gbs.  Past that point, adding channels buys nothing.
    """
    per_ch = cfg.get('bandwidth_lpddr_gbs')
    if not per_ch:
        return None
    bw = per_ch * cfg.get('lpddr_channels', 1)
    cap = cfg.get('bandwidth_c2c_gbs')
    return min(bw, cap) if cap else bw


def _lane_bytes(t: Tensor, lanes: int) -> float:
    """Bytes ONE lane of the consuming (scatter) / producing (gather) side moves."""
    return t.elems * BYTES_PER_ELEM / max(1, lanes)


def _lpddr_leg_ms(tensors: List[Tensor], key: str, st: 'Stage',
                  bw_gbs: Optional[float]) -> float:
    """Time one stage's LPDDR legs cost it [ms], per lane.

    DERATED FOR A SHARED CHANNEL: billing every lane its own bytes at the full bandwidth
    is right only while each lane owns a node.  Stage 4's 8 lanes sit two to a node on the
    4-node south edge, so its two inbound legs are charged double -- without this the
    model would report a bandwidth the memory system cannot deliver.
    """
    total = 0.0
    for t in tensors:
        _nodes, lanes = _leg_fanout(st.n_par, _leg_edge(t, key), t.name)
        total += _transfer_ms(_lane_bytes(t, st.n_par) * lanes, bw_gbs)
    return total


# The three tables, keyed the way _tensors() asks for them.
_TABLES: Dict[str, Dict[int, List[Tensor]]] = {
    'hand': HANDOFF, 'hin': LPDDR_IN, 'hout': LPDDR_OUT}


def _tensors(key: str, i: int) -> List[Tensor]:
    return _TABLES[key].get(i, [])


def _row_shape(row: LayerRow, n_par: int, tiles: int) -> dict:
    """Per-cluster, per-TILE kernel shape for one template row."""
    k = row.kernel.lower()
    ind, outd = row.in_dims, row.out_dims
    if not ind:
        raise ValueError(
            f"row {row.excel_row} ({row.kernel} '{row.op}') has no i_* dims; the "
            f"template needs the tensor-shape columns for real-kernel mode "
            f"(use --wait to fall back to timing-only mempool_wait).")

    div = max(1, n_par * tiles)

    if k in GEMM_KERNELS:
        # ONE GEMM per row, never a batch loop: every leading axis -- token axes on a
        # dense, the attention batch on a matmul -- collapses into the row count.  The MAC
        # count is identical and X and Y span the whole tensor, which is the traffic the
        # app exists to measure.  The one quantity this under-counts is W.
        return {'cls': 'gemm', 'M': max(1, _prod(ind[:-1]) // div),
                'N': ind[-1], 'P': outd[-1]}
    if k in ('softmax', 'layernorm'):
        # The _2x4_f16vec kernels walk two ROWS and four COLUMNS at a time through a
        # 32-bit `*(v2h *)&A[i * N + j]`, so N must be a multiple of 4: a multiple of 2 or
        # the odd rows land on a 2-byte boundary and every v2h load on them is misaligned,
        # and a multiple of 4 or the j loop runs past the row end.  Padding N pads the row
        # STRIDE; the trip count is ceil(N/4) either way, so no cycles move.
        return {'cls': k, 'M': _pad(max(1, _prod(ind[:-1]) // div), 2),
                'N': _pad(ind[-1], 4), 'N_true': ind[-1]}
    if k in ('gelu', 'relu'):
        # Each core takes 8 contiguous elements per step.
        return {'cls': k, 'size': _pad(max(1, _prod(ind) // div), 8)}
    if k == 'add':
        # axpy's UNROLLED4 body moves 8 elements per core per step.
        return {'cls': 'add', 'len': _pad(max(1, _prod(ind) // div), 8)}
    if k == 'pool':
        # A REAL REDUCTION, not a permutation: it contracts the window axis and leaves
        # every (i_3, i_4) element beside it alone, so pool_mean_f16p runs it as a vector
        # accumulate over contiguous rows.  `lane` is what stays independent.
        lane = max(1, _prod(outd[2:]) if len(outd) > 2 else 1)
        n_in = max(1, ind[1] if len(ind) > 1 else 1)
        n_out = max(1, outd[1] if len(outd) > 1 else 1)
        win = -(-n_in // n_out)          # ceil: the last window is short, the divisor is not
        return {'cls': 'pool', 'n_in': n_in, 'n_out': n_out, 'win': win,
                'lane': max(1, lane // div),
                'in': max(1, _prod(ind) // div), 'out': max(1, _prod(outd) // div)}
    if row.excel_row in FREE_ROWS:
        # A VIEW.  Nothing runs, nothing moves, nothing is claimed -- see FREE_ROWS for
        # why each of the three is one.  It stays an Item so the ledger still lists it.
        return {'cls': 'free', 'in': 0, 'out': 0,
                'why': 'view: the operand is already in the order its consumer wants'}
    if row.excel_row in LAYOUT_LOWERING:
        # `in` is the whole source tensor whatever the row moves out of it -- rows 15 and 22
        # read stripes of one QKV -- so only the OUTPUT carries the tensor count.
        return {'cls': LAYOUT_LOWERING[row.excel_row],
                'in': max(1, _prod(ind) // div),
                'out': max(1, PERMUTE_TENSORS.get(row.excel_row, 1)
                           * _prod(outd or ind) // div),
                'why': 'index permutation, on the cores'}
    if k in NO_KERNEL:
        # A wait, not a kernel -- but the tensors it stands for are live all the same.
        # rope rotates Q and K IN PLACE, so it names one buffer.
        inplace = k == 'rope'
        return {'cls': 'wait', 'in': max(1, _prod(ind) // div),
                'out': 0 if inplace else max(1, _prod(outd or ind) // div),
                'why': NO_KERNEL[k], 'layout': False}
    raise ValueError(f"row {row.excel_row} ('{row.op}'): no kernel is mapped to "
                     f"'{row.kernel}'.")


def _pad(n: int, mult: int) -> int:
    """Round n up to a multiple of `mult`.

    The baremetal kernels are vectorised and unrolled and none of them guards its tail: a
    dimension that is not a multiple of the kernel's step is read and written past its
    end, and for a 2-D kernel an ODD row length also puts every other row on a 2-byte
    boundary, where a 32-bit v2h access is misaligned.  Padding the dimension pads the
    buffer with it; the trip counts are unchanged, so no cycles move.
    """
    return ((n + mult - 1) // mult) * mult


def _gemm_rows(M: int, pool: str) -> int:
    """The M a GEMM is actually issued with on this pool.

    redmule_synch_parallel gives engine k rows [k*(M/tiles), ...) via TWO integer
    divisions, so an M that is not a multiple of the tile count both drops the remainder
    and misaligns every engine's slice.  The cost model keeps the TRUE M -- the padding is
    an artefact of the kernel's row split, not work the network does.
    """
    tiles = POOL_REDMULE_TILES.get(pool)          # PE-matmul pools split by core, not tile
    if not tiles:
        return M
    return ((M + tiles - 1) // tiles) * tiles


def _row_footprint(row: LayerRow, n_par: int, tiles: int,
                   pool: Optional[str] = None) -> Tuple[int, int, int]:
    """(X, W, Y) element counts one row needs, per cluster per tile."""
    s = _row_shape(row, n_par, tiles)
    c = s['cls']
    if c == 'gemm':
        # Sized on the padded M: that is the region the engine really walks.
        M = _gemm_rows(s['M'], pool) if pool else s['M']
        return M * s['N'], s['N'] * s['P'], M * s['P']
    if c in ('softmax', 'layernorm'):
        return s['M'] * s['N'], 0, s['M'] * s['N']
    if c in ('gelu', 'relu'):
        return 0, 0, s['size']
    if c == 'add':
        return s['len'], 0, s['len']
    if c == 'wait':
        return s['in'], 0, s['out']
    if c == 'free':
        # A view claims nothing: its operand is already resident, in the order wanted.
        return 0, 0, 0
    if c in ('permute', 'pool'):
        # A permutation that RUNS is not a view, and a reduction never was: each reads one
        # tensor and writes a second, so both ends are charged.
        return s['in'], 0, s['out']
    return 0, 0, 0


def _arena_holds_x(st: Stage, x_reg: str) -> bool:
    """Does this row's X operand sit inside the scratch arena?

    True when the row is unbound -- it reads at the arena base by definition -- and when
    it is bound to a region SCRATCH_OVER let the arena overlap.  False when it reads a
    live pinned region, which never overlaps the arena; such a row may still write AT the
    arena base, since nothing of its own is in it to clear.
    """
    return not x_reg or x_reg in SCRATCH_OVER.get(st.idx, ())


def _arena_out_off(item: Item, st: Stage, pool: str) -> int:
    """Element offset of a layer's output RELATIVE TO THE SCRATCH ARENA BASE.

    A layer whose input is in the arena writes at its own input size, rounded UP to a DMA
    line so the output never grows back into the input the kernel is still reading; one
    whose input is a live pinned region writes at the arena base itself (_arena_holds_x).
    A TILED item is the exception: its input region holds ALL its tiles at once, so its
    output has to clear the whole region.

    The rounding is not cosmetic: an unaligned DMA endpoint halves the leg's bandwidth.
    """
    x_reg, _y_reg = BIND.get((st.idx, item.excel_row), ('', ''))
    if not _arena_holds_x(st, x_reg):
        return 0
    x = _row_footprint(item.row, st.n_par, item.tiles, pool)[0]
    return _pad(x * item.tiles, DMA_LINE_ELEMS)


# The key _stage_layout uses for the scratch arena's base.  '@' keeps it out of any
# namespace a Tensor or Region name could collide with.
ARENA = '@arena'


def _arena_elems(st: Stage, pool: str) -> int:
    """How much scratch the rows BIND does not place need, per cluster.

    The arena holds one row's input and its output at a time, so it is sized to the
    largest (input + output) any row puts there.  Both halves are measured against the
    arena independently, since a row may have one bound and the other not.  This does not
    depend on the arena's BASE -- whether the arena overlaps a row's X is decided by
    SCRATCH_OVER -- so there is no circularity between sizing the arena and placing it.
    """
    need = 0
    for item in st.items:
        x_reg, y_reg = BIND.get((st.idx, item.excel_row), ('', ''))
        x, _w, y = _row_footprint(item.row, st.n_par, item.tiles, pool)
        end = 0
        if _arena_holds_x(st, x_reg):
            end = max(end, x * item.tiles)
        if not y_reg:
            end = max(end, _arena_out_off(item, st, pool) + y)
        need = max(need, end)
    return _pad(need, DMA_LINE_ELEMS)


def _stage_layout(st: Stage, pool: str) -> Dict[str, int]:
    """Every element offset inside one cluster's X_BUF, by name.

    THIS IS THE ONLY PLACE THAT DECIDES AN OFFSET.  The map holds, in layout order:

      * every INBOUND tensor, hand-off then LPDDR read, at the offsets the DMA writes them
        to.  Laying them out first and back to back makes the consumer's inbound region
        contiguous, which lets push_to_next derive its destinations from the sizes alone;
      * every RESIDENT region that is not already an inbound name -- one that IS rides
        that tensor's slot;
      * ARENA, the scratch base for every row BIND does not place;
      * '@end', one past the highest offset the stage uses -- what X_BUF must hold.

    The arena is FIRST-FIT into the gaps SCRATCH_OVER opens up, not appended: on the
    encoder stages the Q/K/V folds are dead after row 27 while Z stays live above them.
    Counts are PER CLUSTER.
    """
    off: Dict[str, int] = {}
    spans: List[Tuple[int, int, bool]] = []       # (start, end, live)
    at = 0
    dead = set(SCRATCH_OVER.get(st.idx, ()))

    # What each region has to hold for the rows BOUND to it, which is not always the
    # tensor's own size: a GEMM's M is rounded up to the tensor-engine count, so a row may
    # walk a few more rows than the template declares.  A WRITE past a region's end would
    # land in a live neighbour, so a region is sized to the larger of the two.  A READ that
    # overshoots is harmless but still has to stay inside X_BUF, which '@end' guarantees.
    writes: Dict[str, int] = {}
    reads: Dict[str, int] = {}
    for item in st.items:
        x_reg, y_reg = BIND.get((st.idx, item.excel_row), ('', ''))
        x, _w, y = _row_footprint(item.row, st.n_par, item.tiles, pool)
        if x_reg:
            reads[x_reg] = max(reads.get(x_reg, 0), x * item.tiles)
        if y_reg:
            writes[y_reg] = max(writes.get(y_reg, 0), y)

    def place(name: str, elems: int, grow: bool) -> None:
        nonlocal at
        size = _pad(elems // st.n_par, DMA_LINE_ELEMS)
        if grow:
            size = max(size, _pad(writes.get(name, 0), DMA_LINE_ELEMS))
        off[name] = at
        spans.append((at, at + size, name not in dead))
        at += size

    def place_inbound(t: Tensor) -> None:
        # An inbound region's size is the DMA's, not a kernel's: push_to_next lays the
        # consumer's tensors out by a running sum of the RAW per-lane byte counts, so a
        # region padded here and not there would put every following tensor at an address
        # the producer does not write.  Checked, not assumed.
        per_lane = t.elems // st.n_par
        if per_lane % DMA_LINE_ELEMS:
            raise ValueError(
                f"stage {st.idx + 1} receives '{t.name}' as {per_lane} elements per lane, "
                f"which is not a multiple of DMA_LINE_ELEMS ({DMA_LINE_ELEMS}); the "
                f"generator's inbound layout and push_to_next's running sum would "
                f"disagree from this tensor on")
        place(t.name, t.elems, grow=False)

    for t in (_tensors('hand', st.idx - 1) if st.idx else []):
        place_inbound(t)
    for t in _tensors('hin', st.idx):
        place_inbound(t)
    for r in RESIDENT.get(st.idx, ()):
        if r.name in off:
            continue                     # rides its inbound slot -- no copy, no space
        place(r.name, r.elems, grow=True)

    # First fit for the arena among the live spans.  `at` is the end of everything placed,
    # so appending is always available as the last candidate.
    need = _arena_elems(st, pool)
    live = sorted((s, e) for s, e, is_live in spans if is_live)
    cursor = 0
    for start, end in live:
        if start - cursor >= need:
            break
        cursor = max(cursor, end)
    off[ARENA] = cursor

    end = max(at, cursor + need)
    for name, size in list(reads.items()) + list(writes.items()):
        if name not in off:
            raise ValueError(
                f"stage {st.idx + 1} binds a row to region {name!r}, which is neither an "
                f"inbound tensor of that stage nor one of its RESIDENT regions; the "
                f"tables under THE DISTRIBUTED Q/K/V/Z BUFFER have drifted apart")
        end = max(end, off[name] + size)
    off['@end'] = _pad(end, DMA_LINE_ELEMS)
    return off


def _item_w_elems(item: Item, pool: str, n_par: int) -> int:
    """Elements one row claims in W_BUF: its operand B, or its parameter table.

      * the GEMM's SECOND OPERAND, N*P.  On a dense that is the weight matrix; on a matmul
        it is K, V or UG -- an ACTIVATION the template rightly scores as 0 weights, but
        which this lowering reads through W_BUF, so the buffer has to hold it.
      * the template's own Weights figure, the row's PARAMETERS.  Most emitted kernels
        take no pointer to these, but a real build holds them, so the space is reserved.

    Divided by n_rep, because a repeated row is placed on n_rep stages and each holds ONE
    weight set.  NOT divided by n_par: the lanes split rows, so every cluster holds the
    whole matrix.
    """
    gemm = _row_footprint(item.row, n_par, item.tiles, pool)[1]
    params = int(item.row.weights_bytes / item.row.n_rep) // BYTES_PER_ELEM
    return max(gemm, params)


def _stage_w_layout(st: Stage, pool: str) -> Dict[int, int]:
    """Element offset of each item's W_BUF region, keyed by position in st.items.

    EVERY ROW GETS ITS OWN REGION, and a stage's weights are resident rather than streamed
    -- loaded once and re-read on every pipeline item -- so what the stage needs is their
    SUM, not their max.
    """
    off: Dict[int, int] = {}
    at = 0
    for n, item in enumerate(st.items):
        need = _item_w_elems(item, pool, st.n_par)
        if not need:
            continue
        off[n] = at
        at += _pad(need, DMA_LINE_ELEMS)
    off[-1] = at                      # '@end', under a key no item index can take
    return off


def _region_off(name: str, st: Stage, pool: str) -> int:
    layout = st.layout or _stage_layout(st, pool)
    if name not in layout:
        raise ValueError(
            f"stage {st.idx + 1} names region {name!r}, which is neither an inbound "
            f"tensor of that stage nor one of its RESIDENT regions; the tables under THE "
            f"DISTRIBUTED Q/K/V/Z BUFFER have drifted apart")
    return layout[name]


def _item_offsets(item: Item, st: Stage, pool: str) -> Tuple[int, int]:
    """(X offset, Y offset) of one item inside X_BUF, in elements.

    A row BIND places reads and/or writes a named region; '' falls back to the scratch
    arena, which is the chain behaviour: read at the base, write one padded input past it.
    A row naming the same region twice is a VIEW -- both ends come back identical.
    """
    x_reg, y_reg = BIND.get((st.idx, item.excel_row), ('', ''))
    base = _region_off(ARENA, st, pool)
    x = _region_off(x_reg, st, pool) if x_reg else base
    y = (_region_off(y_reg, st, pool) if y_reg
         else base + _arena_out_off(item, st, pool))
    return x, y


def _stage_src_off(st: Stage, name: str, pool: str) -> int:
    """Element offset a stage's OUTBOUND tensor `name` is sent from.

    OUTBOUND_AT names the region for the tensors that do not simply come off the last
    layer; everything else is what the stage's last layer wrote.
    """
    entry = OUTBOUND_AT.get((st.idx, name))
    if entry is None:
        return _item_offsets(st.items[-1], st, pool)[1]
    region, within = entry
    return _region_off(region, st, pool) + within // st.n_par


def _row_latency(row: LayerRow, rate_gemm: float, rate_pe: float) -> float:
    """Modelled latency [ms] of ONE PLACEMENT of a row, across the WHOLE stage.

    The FLOPs column carries the xN_repetitions factor and a repeated row is placed on
    n_rep stages, so a placement is worth its share and the placements of a row sum back
    to the template figure exactly.
    """
    return row.flops / row.n_rep / (rate_gemm if row.is_gemm else rate_pe)


def _pool_cores(cfg: dict, pool: str) -> int:
    """Cores one cluster of this pool has -- num_core_per_cluster of its arch config.

    Only the permutation kernels need it.  Every other row is charged against a FLOPs peak
    with the core count already folded in; a permutation has no FLOPs to be charged
    against, so the width of the cluster IS the rate.
    """
    n = cfg.get('cores', {}).get(pool)
    if not n:
        raise ValueError(
            f'config gives no "cores" for pool {pool!r}; the permutation kernels are '
            f'charged per core, so each pool needs its num_core_per_cluster (see '
            f'software/runtime/multi_cluster/configs/arch_*.py)')
    return int(n)


def _layout_latency(item: Item, n_par: int, cores: int,
                    cycles_per_ms: float) -> float:
    """Modelled latency [ms] of ONE cluster's ONE tile of a zero-FLOP layout row.

    Applied to the DESTINATION element count: a permutation costs one pass over what it
    writes, spread over the cluster's cores.

    A CORE COUNT, NOT A FLOPs RATE.  A permutation carries no FLOPs, so there is no peak to
    charge it against; what sets the rate is how many cores are walking and how many cycles
    each element costs one of them.  See LAYOUT_CYCLES_PER_ELEM.
    """
    how = LAYOUT_LOWERING.get(item.row.excel_row)
    rate = LAYOUT_CYCLES_PER_ELEM.get(how)
    if rate is None:
        raise ValueError(
            f'row {item.row.excel_row} is lowered to {how!r}, which has no '
            f'cycles-per-element figure; LAYOUT_CYCLES_PER_ELEM needs one per '
            f'LAYOUT_LOWERING value')
    elems = _row_footprint(item.row, n_par, item.tiles)[2]
    return elems / max(1, cores) * rate / cycles_per_ms


def compute(layers: List[LayerRow], cfg: dict) -> Results:
    pools = list(cfg['PE_peak'].keys())
    pe_util = cfg['PE_utilization']
    te_util = cfg['TE_utilization']
    freq = cfg.get('frequency', 1_000_000_000)
    cycles_per_ms = freq / 1e3
    bw_c2c = cfg.get('bandwidth_c2c_gbs')
    bw_lpddr = _lpddr_bw(cfg)

    n_clusters = sum(s['n_par'] for s in STAGES)
    want = cfg.get('pipeline_clusters')
    if want is not None and int(want) != n_clusters:
        raise ValueError(
            f"config says pipeline_clusters={want} but the stage table sums to "
            f"{n_clusters} clusters ({' '.join(str(s['n_par']) for s in STAGES)}). "
            f"The partition is a property of the model: change STAGES, not the config.")

    flops_per_ms: Dict[str, Dict[str, float]] = {}
    row_lat: Dict[str, Dict[int, float]] = {}
    total_lat: Dict[str, float] = {}
    stages_per_pool: Dict[str, List[Stage]] = {}
    best: Dict[str, float] = {}
    best_ov: Dict[str, float] = {}

    for pool in pools:
        gemm_util = te_util if pool in TENSOR_ENGINE_POOLS else pe_util
        rate_pe = cfg['PE_peak'][pool] * pe_util * 1e9         # FLOPs per ms
        rate_gemm = cfg['GEMM_peak'][pool] * gemm_util * 1e9
        flops_per_ms[pool] = {'non_GEMM': rate_pe, 'GEMM': rate_gemm}

        stages = build_stages(layers)
        cores = _pool_cores(cfg, pool)
        lat: Dict[int, float] = {}

        for st in stages:
            # Every offset the stage uses, resolved once and carried on the Stage: the
            # emitters, the footprint and the LPDDR layout all read this map rather than
            # each deriving an offset of their own.
            st.layout = _stage_layout(st, pool)
            st.w_layout = _stage_w_layout(st, pool)
            x_max = w_max = y_max = 0
            for item in st.items:
                if item.row.flops:
                    whole = _row_latency(item.row, rate_gemm, rate_pe)
                    item.ms = whole / (st.n_par * item.tiles)
                elif item.row.excel_row in FREE_ROWS:
                    # A VIEW costs nothing, and saying so is the point: charging one a
                    # spin, as this file used to, reports a network slower than the one
                    # it runs.
                    item.ms = 0.0
                    whole = 0.0
                else:
                    # A zero-FLOP layout row a KERNEL runs: the figure is per cluster per
                    # tile, so the whole-stage number the report column wants is scaled up.
                    item.ms = _layout_latency(item, st.n_par, cores, cycles_per_ms)
                    whole = item.ms * st.n_par * item.tiles
                lat[item.excel_row] = lat.get(item.excel_row, 0.0) + whole
                x, w, y = _row_footprint(item.row, st.n_par, item.tiles, pool)
                x_max, w_max, y_max = max(x_max, x), max(w_max, w), max(y_max, y)

            # One tile's work, times the number of tiles that item makes.
            st.latency = sum(i.ms * i.tiles for i in st.items)

            # ---- inbound / outbound bytes, per cluster
            inbound = _tensors('hand', st.idx - 1) if st.idx else []
            outbound = _tensors('hand', st.idx)
            lpddr_in = _tensors('hin', st.idx)
            lpddr_out = _tensors('hout', st.idx)

            din_c2c = sum(_lane_bytes(t, st.n_par) for t in inbound)
            dout_c2c = sum(_lane_bytes(t, st.n_par) for t in outbound)
            din_lpddr = sum(_lane_bytes(t, st.n_par) for t in lpddr_in)
            dout_lpddr = sum(_lane_bytes(t, st.n_par) for t in lpddr_out)

            st.din_bytes = din_c2c + din_lpddr
            st.dout_bytes = dout_c2c + dout_lpddr

            # A stage pays for the hand-off it PRODUCES plus its own LPDDR legs; the
            # inbound c2c leg is the producer's bill, already counted there.
            xfer = (_transfer_ms(dout_c2c, bw_c2c)
                    + _lpddr_leg_ms(lpddr_in, 'hin', st, bw_lpddr)
                    + _lpddr_leg_ms(lpddr_out, 'hout', st, bw_lpddr))
            st.din_ms = _lpddr_leg_ms(lpddr_in, 'hin', st, bw_lpddr)

            if st.tiles > 1:
                # Self-draining: its transfers are serialised inside its own compute, so
                # they belong to `latency`.  Keep the amount in drain_ms so the reports
                # can show where that time went instead of burying it in the kernels.
                st.latency += xfer
                st.drain_ms = xfer
                st.transfer_ms = 0.0
                st.din_ms = 0.0
            else:
                st.transfer_ms = xfer

            # ---- buffers.  X receives every inbound tensor at once, hand-off and LPDDR
            # read alike; Y must hold every tensor the stage hands ON.  A tiled stage
            # drains one tile at a time, so its outbound region follows the tile.
            in_elems = int(sum(t.elems / st.n_par for t in inbound + lpddr_in))
            out_elems = int(sum(t.elems / st.n_par
                                for t in outbound + lpddr_out)
                            / max(1, st.tiles))
            # kernel_elems -- what this stage's LAYERS need: one X buffer holding a
            #                 layer's input and output, plus W.  A per-layer diagnostic.
            # x/w/y_elems  -- what the emitted app allocates: X and Y as separate static
            #                 arrays, both also DMA endpoints, so the hand-off and every
            #                 carried tensor land in them too.
            kx, kw = _stage_buffers(st, pool)
            st.kernel_elems = kx + kw
            # '@end' is the whole claim on X_BUF: pinned regions, the inbound landing zone
            # and the arena, with SCRATCH_OVER's overlaps already taken.  Only the
            # outbound region is still checked separately, because a tiled stage's
            # follows the tile.
            st.x_elems = max(x_max, in_elems, st.layout['@end'])
            st.w_elems = w_max
            st.y_elems = max(y_max, out_elems)

        _check_handoff_sizing(stages)
        row_lat[pool] = lat
        total_lat[pool] = sum(lat.values())
        stages_per_pool[pool] = stages
        best[pool] = _makespan(stages, overlap=False)
        best_ov[pool] = _makespan(stages, overlap=True)

    return Results(flops_per_ms, row_lat, total_lat, stages_per_pool, best, best_ov)


def _overlap_wall(st: Stage) -> float:
    """cwt wall time of one stage: the inbound leg cannot hide behind its own compute."""
    return st.din_ms + max(st.latency, st.transfer_ms - st.din_ms)


def _makespan(stages: List[Stage], overlap: bool) -> float:
    """Steady-state period of one pipeline step = what throughput is 1/of.

    Both generated loops rendezvous every cluster on a global barrier, so a step costs the
    slowest thing running inside each barriered phase:

    * ctt (overlap=False): compute -> barrier -> transfer -> barrier, TWO SUBSTAGES, so
      the step is max_s(compute_s) + max_s(transfer_s).
    * cwt (overlap=True): double buffering lets a cluster push item m-1 while it computes
      item m, one barrier a step, so the step is max_s(wall_s) -- see _overlap_wall.

    This is NOT the naive max_s(compute_s + transfer_s), and is >= it whenever the slowest
    compute and the slowest transfer sit on DIFFERENT stages.  Kept identical to
    cevit_pipeline._makespan so the two models stay comparable.
    """
    if overlap:
        return max(_overlap_wall(s) for s in stages)
    return max(s.latency for s in stages) + max(s.transfer_ms for s in stages)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    return f'{n / 1024:.0f} KiB' if n < 1024 * 1024 else f'{n / (1024 * 1024):.2f} MiB'


def print_results(layers: List[LayerRow], res: Results, cfg: dict,
                  l1_bytes: int) -> None:
    pools = list(res.flops_per_ms.keys())

    bw_c2c = cfg.get('bandwidth_c2c_gbs')
    bw_ch = cfg.get('bandwidth_lpddr_gbs')
    n_ch = cfg.get('lpddr_channels', 1)
    cum = (bw_ch or 0) * n_ch
    eff = _lpddr_bw(cfg)
    print(f'\nLinks: c2c {bw_c2c or 0:g} GB/s  |  '
          f'LPDDR {bw_ch or 0:g} GB/s x {n_ch} channel(s) = '
          f'{cum:g} GB/s cumulative')
    # Say what is CHARGED, not just what the parts are worth: past the c2c link the extra
    # channels buy nothing, and a reader reconciling a report against the model would
    # otherwise chase a figure never used.
    if eff is not None and eff < cum:
        print(f'       off-chip legs charged at {eff:g} GB/s -- capped by the c2c link')

    print('\nFLOPs per ms')
    print(f'  {"pool":<12}{"non-GEMM":>16}{"GEMM":>16}')
    for p in pools:
        f = res.flops_per_ms[p]
        print(f'  {p:<12}{f["non_GEMM"]:>16,.0f}{f["GEMM"]:>16,.0f}')

    print('\nPer-row latency [ms]  (whole stage, before the /n_par split; a repeated row')
    print('is summed over its placements, so the column sums to the template total)')
    head = f'  {"row":>4} {"xN":>3}  {"block":<20}{"op":<34}{"kernel":<14}'
    print(head + ''.join(f'{p:>13}' for p in pools))
    for r in layers:
        if r.excel_row in DROPPED_ROWS:
            continue
        cells = ''.join(f'{res.row_latencies[p].get(r.excel_row, 0.0):>13.5f}'
                        for p in pools)
        op = r.op if len(r.op) <= 33 else r.op[:30] + '...'
        print(f'  {r.excel_row:>4} {r.n_rep:>3}  {r.block[:19]:<20}{op:<34}'
              f'{r.kernel:<14}' + cells)
    print(f'  {"":>4} {"":>3}  {"TOTAL":<20}{"":<34}{"":<14}'
          + ''.join(f'{res.total_latency[p]:>13.5f}' for p in pools))

    geo_budget = {}
    for p in pools:
        geo = L1_GEOMETRY.get(p)
        geo_budget[p] = (min(l1_bytes, geo['total']) - geo['stacks'] - geo['runtime']
                         if geo else l1_bytes)

    for p in pools:
        stages = res.stages[p]
        idle = _stage_bases(stages)[1] - sum(s.n_par for s in stages)
        print(f'\n[{p}]  pipeline: {len(stages)} stages, '
              f'{sum(s.n_par for s in stages)} clusters'
              + (f' (+{idle} idle, so the LPDDR stages start on a column '
                 f'boundary)' if idle else ''))
        print(f'  {"st":>3} {"cl":>3} {"tl":>3}  {"work":<46}'
              f'{"compute":>10}{"xfer":>9}{"din KiB":>9}{"dout KiB":>9}{"X + W":>11}')
        for st in stages:
            work = ', '.join(i.label for i in st.items)
            if len(work) > 45:
                work = work[:42] + '...'
            need = st.kernel_elems * BYTES_PER_ELEM
            print(f'  {st.idx + 1:>3} {st.n_par:>3} {st.tiles:>3}  {work:<46}'
                  f'{st.latency:>10.5f}{st.transfer_ms:>9.5f}'
                  f'{st.din_bytes / 1024:>9.0f}{st.dout_bytes / 1024:>9.0f}'
                  f'{_fmt_bytes(need):>10}{"!" if need > geo_budget[p] else " "}')
        print('  X + W = the stage\'s LARGEST SINGLE LAYER: its input + output, plus its')
        print('          own operand B; per cluster.  A per-layer diagnostic -- what the')
        print('          app allocates is the app-wide line below, which is bigger.')
        nx, nw = _scratch_elems(stages, p)
        app_l1 = (nx + nw) * BYTES_PER_ELEM
        worst_x = max(stages, key=lambda s: s.layout['@end'])
        worst_w = max(stages, key=lambda s: s.w_layout[-1])
        print(f'  app-wide buffers: X {_fmt_bytes(nx * BYTES_PER_ELEM)} '
              f'(inbound + pinned regions + arena; peak on stage {worst_x.idx + 1}) + W '
              f'{_fmt_bytes(nw * BYTES_PER_ELEM)} (every operand B resident; peak on '
              f'stage {worst_w.idx + 1}) = {_fmt_bytes(app_l1)}  '
              f'(budget {_fmt_bytes(geo_budget[p])})')
        # W_BUF holds the GEMMs' second operand, which on a matmul row is an ACTIVATION
        # that also arrives in X_BUF -- so the buffer total is above the model's parameter
        # count by exactly those rows.  Both are printed.
        params = max(sum(it.row.weights_bytes / it.row.n_rep for it in st.items)
                     for st in stages)
        print(f'          of which parameters (template Weights column, worst stage): '
              f'{_fmt_bytes(params)}; the rest is matmul operands that ride W_BUF.')
        if app_l1 > geo_budget[p]:
            worst = max(stages, key=lambda s: s.kernel_elems)
            print(f'  WARNING: {_fmt_bytes(app_l1 - geo_budget[p])} over the .l1 budget '
                  f'-- largest stage by its own layers is {worst.idx + 1}.')
        # A self-draining stage is measured against ONE TILE: it writes tile t and re-uses
        # the same region for tile t+1, so the whole lane's share is never live at once.
        # That is the quantity _scratch_elems sized X from, which is what keeps this check
        # and the sizing from disagreeing.
        big = [s for s in stages
               if max(s.din_bytes, s.dout_bytes / s.tiles) > nx * BYTES_PER_ELEM]
        if big:
            raise ValueError(
                f'{p}: stage(s) {[s.idx + 1 for s in big]} receive or send more than X '
                f'holds ({_fmt_bytes(nx * BYTES_PER_ELEM)}); _scratch_elems is not '
                f'covering the hand-off regions')
        print(f'  ctt period {res.best_throughput[p]:.6g} ms  '
              f'(latency {res.best_throughput[p] * len(stages):.6g} ms)')
        print(f'  cwt period {res.best_throughput_overlap[p]:.6g} ms  '
              f'(latency {res.best_throughput_overlap[p] * 2 * len(stages):.6g} ms)')


# ---------------------------------------------------------------------------
# Buffer sizing
# ---------------------------------------------------------------------------

def _stage_buffers(st: Stage, pool: str) -> Tuple[int, int]:
    """(X, W) elements one cluster of this stage must declare for its LAYERS.

        X = max over layers of (input + output)
        W = max over layers of  weights

    with the shapes already per cluster per tile.  Weights do NOT divide: the split is
    over rows, so every cluster holds the whole matrix.  This counts a layer's own
    operands only, as if it ran alone in an empty buffer -- a diagnostic about the LAYERS,
    not the layout.  X_BUF is really sized from _stage_layout and _scratch_elems.
    """
    x = w = 0
    for it in st.items:
        _, wi, yi = _row_footprint(it.row, st.n_par, it.tiles, pool)
        x = max(x, _arena_out_off(it, st, pool) + yi)
        w = max(w, wi)
    return x, w


def _scratch_elems(stages: List[Stage], pool: str) -> Tuple[int, int]:
    """(X, W) elements the app declares -- one X_BUF and one W_BUF for every stage.

    X comes from _stage_layout's '@end', which is also what the emitted kernel calls and
    both DMA directions address through.  What it does NOT cover is where a stage's
    OUTBOUND tensors are read from -- a length starting at an offset the map only knows
    the base of -- so each is checked against the region behind it here.  A TILED item
    drains ONE TILE at a time, so its outbound LPDDR region is a tile.

    Rounded up to 32 elements (64 B) so the DMA sees an aligned base.
    """
    nx = nw = 1
    for st in stages:
        kx, _kw = _stage_buffers(st, pool)
        # W is the SUM of the stage's operand-B and parameter regions, not the largest of
        # them: a stage's weights are all resident at once.  See _stage_w_layout.
        nw = max(nw, (st.w_layout or _stage_w_layout(st, pool))[-1])
        nx = max(nx, kx, st.layout['@end'] if st.layout else _stage_layout(st, pool)['@end'])
        for x in _stage_xfers(st.idx):
            nx = max(nx, _stage_src_off(st, x.name, pool)
                     + x.nbytes // st.n_par // BYTES_PER_ELEM)
        for t in _tensors('hout', st.idx):
            nx = max(nx, _stage_src_off(st, t.name, pool)
                     + int(_lane_bytes(t, st.n_par)) // st.tiles // BYTES_PER_ELEM)
    rnd = lambda n: ((n + 31) // 32) * 32
    return rnd(nx), rnd(nw)


def _l1_geometry(num_cores: int, banking: int, bank_size: int,
                 stack: int = 512) -> Dict[str, int]:
    banks = banking * num_cores
    return {
        'total': banks * bank_size,
        'stacks': num_cores * stack,
        'runtime': 2 * banks * 4 + 2 * banks + 64,
        'align': banks * 4,
    }


L1_GEOMETRY = {
    'TensorPool': _l1_geometry(num_cores=256, banking=8, bank_size=2048),
    'Terapool': _l1_geometry(num_cores=1024, banking=4, bank_size=1024),
}


def _parity_elems(st: Stage, pool: str) -> int:
    """Elements of ONE stage that need a SECOND copy under cwt.

    ONLY WHAT CROSSES AN ITEM BOUNDARY, which is two things and not the whole buffer:

      * the stage's FIRST kernel's input -- the landing zone.  Under cwt the transfer that
        brings item i+1 runs while item i computes, so the bytes being written are not the
        bytes being read and the two cannot share a region.
      * the stage's LAST kernel's output -- whatever an outbound transfer reads.  Same
        argument the other way: item i is being shipped while item i+1 is being computed
        into the same place.

    EVERYTHING ELSE IS SINGLE.  The pinned intermediates and the scratch arena are written
    and read inside one item's compute, with a barrier either side, so no transfer is ever
    looking at them -- and the WEIGHTS are never written at all.  Doubling the whole of X,
    which this file used to do, charges the parity copy to regions no transfer touches; it
    is what put both cwt apps over budget by 943 KiB (TensorPool) and 1355 KiB (Terapool).
    """
    inbound = (_tensors('hand', st.idx - 1) if st.idx else []) + _tensors('hin', st.idx)
    landing = sum(int(_lane_bytes(t, st.n_par)) for t in inbound) // BYTES_PER_ELEM

    drain = sum(x.nbytes // st.n_par for x in _stage_xfers(st.idx))
    drain += sum(int(_lane_bytes(t, st.n_par)) // st.tiles
                 for t in _tensors('hout', st.idx))
    return landing + drain // BYTES_PER_ELEM


def _l1_footprint(stages: List[Stage], mode: str, pool: str) -> Tuple[int, str]:
    nx, nw = _scratch_elems(stages, pool)
    par = 0
    if mode == 'cwt':
        # ONE X_BUF AND ONE PARITY REGION SERVE EVERY STAGE, so the declaration has to
        # satisfy both maxima independently: X >= max(own) and parity >= max(parity).  That
        # is max(own) + max(parity), which is an upper bound on max(own + parity) and here
        # exceeds it by about 385 KiB -- the two maxima fall on different stages (stage 1
        # owns the largest working set, stage 5 the largest boundary pair).  Sizing to the
        # tighter number would need a per-stage buffer, which the app does not have.
        par = max((_parity_elems(st, pool) for st in stages), default=0)
        par = ((par + 31) // 32) * 32
    geo = L1_GEOMETRY.get(pool, {'stacks': 0, 'align': 64})
    sizes = [(nx + par) * BYTES_PER_ELEM, nw * BYTES_PER_ELEM]
    at = geo['stacks']
    for sz in sizes:
        at = ((at + geo['align'] - 1) // geo['align']) * geo['align'] + sz
    total = at - geo['stacks']
    parts = (f'X {nx * BYTES_PER_ELEM}B'
             + (f' + parity {par * BYTES_PER_ELEM}B' if par else '')
             + f' + W {nw * BYTES_PER_ELEM}B '
               f'+ {(total - sum(sizes)) / 1024:.0f} KiB align')
    return total, parts


# ---------------------------------------------------------------------------
# Shared code renderers
# ---------------------------------------------------------------------------

_KERNEL_INCLUDES = [
    'builtins_v2.h',
    'baremetal/mempool_redmule_f16.h',
    'baremetal/mempool_matmul_f16.h',
    'baremetal/mempool_softmax_f16.h',
    'baremetal/mempool_layernorm_f16.h',
    'baremetal/mempool_gelu_f16.h',
    'baremetal/mempool_axpy_f16.h',
    'baremetal/mempool_pool_f16.h',
    'baremetal/mempool_permute_f16.h',
]


# Stack a core gets, from stack_size in config/*.mk (512 on every pool this generates for).
# config/tensorpool256.mk sets stack_size = seq_mem_size = 512, so the core's whole
# sequential-memory slot IS its stack, and flex_start.S arms the stacklimit CSR from it.
PE_STACK_B = 512

# THE STAGE BODIES GET THEIR OWN FRAMES, and that is a change from how this file used to
# emit them.  Inlined, all five land in main()'s frame at once and it is sized by the SUM
# of their spills rather than the largest of them.  That was measured to fit -- 288 B on
# TensorPool against the 508 usable (STACK_SIZE - 4; _reset_vector starts sp four bytes
# down) -- under a stage table that lowered every layout row to a mempool_wait.
#
# IT NO LONGER HOLDS.  The permutation rows are real kernels now, and permute_f16p is
# exactly the case the old comment said to re-measure for: a six-level table with a live
# pointer chain across its asm blocks, 176 B of frame on its own.  main() measured 416 B
# once the two new kernels were in, up 128 B and leaving 92 B of the budget -- and the
# remaining work on this partition all lands in the same frame.
#
# So the depth is bounded instead: main + ONE stage + ONE kernel, which no longer grows
# with the number of stages.  Nothing in the toolchain checks stack depth, and an overrun
# is a store below __stack_start -- on hart 0 an access to 0xFFFFFFFC -- so the emitter
# keeps the depth down rather than relying on a measurement staying true.
#
# ONLY THE FUNCTIONS THAT SPILL ARE MARKED.  The leaf DMA wrappers (dma_2d and friends,
# lpddr_node_addr) stay inline: they marshal a few values into argument registers and call
# straight into flex_dma_pattern.h, whose entry points carry no frame at all, so
# un-inlining one would move no spill out of its caller and would only add a frame to the
# transfer chain.  cevit_pipeline.py draws the line in the same place.
_LEAF_T = lambda ret: f'static inline {ret}'   # leaf wrappers: no frame of their own
_INLINE_T = lambda ret: f'static NO_INLINE {ret}'
_INLINE = _INLINE_T('void')


def _render_stack_note(stages: List[Stage]) -> List[str]:
    """The NO_INLINE macro, and why the app has to care about its stack."""
    n = len(stages)
    return [
        '/* NO_INLINE: give this function ITS OWN STACK FRAME.',
        ' *',
        f' * A core gets STACK_SIZE bytes of stack ({PE_STACK_B} on both pools; see',
        ' * config/*.mk) and _reset_vector leaves STACK_SIZE - 4 of them usable -- it starts',
        ' * sp four bytes down and writes that limit into the stacklimit CSR.  Nothing in the',
        ' * toolchain checks the depth against it.',
        ' *',
        f' * INLINED, THE {n} STAGE BODIES SHARE ONE FRAME IN main(), and it is sized by the',
        ' * SUM of their spills rather than the largest of them.  That fit while every layout',
        ' * row was a mempool_wait; it stopped being comfortable once the permutations became',
        ' * real kernels -- permute_f16p alone carries 176 B, and main() measured 416 B of the',
        ' * 508 available.  A stage of its own frame bounds the depth at main + ONE stage +',
        ' * ONE kernel instead.',
        ' *',
        ' * ONLY THE FUNCTIONS THAT SPILL ARE MARKED.  The leaf wrappers below (dma_2d and',
        ' * friends) marshal a few values into argument registers and call straight into',
        ' * flex_dma_pattern.h, whose entry points carry no frame at all; un-inlining one of',
        ' * those would move no spill out of its caller and would only add a frame to the',
        ' * transfer chain.  noinline pays where several large bodies are folded into one',
        ' * frame, and nowhere else. */',
        '#define NO_INLINE __attribute__((noinline))',
    ]


def _c_comment(text: str) -> str:
    return text.replace('*/', '* /')


def _render_includes(wait: bool) -> List[str]:
    L = ['#include "flex_runtime.h"', '#include "flex_dma_pattern.h"']
    if not wait:
        L += [f'#include "{h}"' for h in _KERNEL_INCLUDES]
    return L


def _render_dma_helper(bench: bool) -> str:
    b0 = '    mempool_start_benchmark();\n' if bench else ''
    b1 = '    mempool_stop_benchmark();\n' if bench else ''
    return (
        'static inline void dma_2d_strided(uint64_t dst, uint64_t src, uint32_t chunk,\n'
        '                                  uint32_t dst_stride, uint32_t src_stride,\n'
        '                                  uint32_t reps)\n'
        '{\n'
        f'{b0}'
        '    flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);\n'
        '    flex_dma_async_wait_all();\n'
        f'{b1}'
        '}\n'
        '\n'
        'static inline void dma_2d(uint64_t dst, uint64_t src, uint32_t bytes)\n'
        '{\n'
        '    dma_2d_strided(dst, src, bytes, bytes, bytes, 1);\n'
        '}\n')


def _render_dma_issue_helper() -> str:
    return (
        'static inline void dma_2d_strided_issue(uint64_t dst, uint64_t src,\n'
        '                                        uint32_t chunk, uint32_t dst_stride,\n'
        '                                        uint32_t src_stride, uint32_t reps)\n'
        '{\n'
        '    flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);\n'
        '}\n'
        '\n'
        'static inline void dma_2d_issue(uint64_t dst, uint64_t src, uint32_t bytes)\n'
        '{\n'
        '    dma_2d_strided_issue(dst, src, bytes, bytes, bytes, 1);\n'
        '}\n')


# ---------------------------------------------------------------------------
# Hand-off tables
# ---------------------------------------------------------------------------
#
# The table stores only what does NOT follow from the two stages' lane counts.  Every
# offset, stride, repetition count and lane step is derived in the C, so no two stored
# numbers can contradict each other.

@dataclass(frozen=True)
class Xfer:
    """One tensor's leg of one stage hand-off, in bytes.

    The sizing rule holds for every shape: a producer cluster owns `nbytes // lanes_p` and
    a consumer needs `nbytes // lanes_c`.  Summed over a boundary those are exactly the
    two stages' dout_c2c and din_c2c, which is what makes "a stage DMAs the next stage's
    input, no more" true by construction; _check_handoff_sizing holds the generator to it.

    """
    name: str
    nbytes: int           # the WHOLE tensor, across every lane of both sides
    scatter: str = ''     # the layout row this leg performs; '' is a plain copy


def _stage_xfers(i: int) -> List[Xfer]:
    """The tensors stage i pushes to stage i+1, laid out back to back.

    A lane owns bytes/lanes contiguous bytes of it, and the side with fewer lanes holds
    several of the other's shares side by side; push_to_next picks fan-out, fan-in or a
    straight copy from the two lane counts alone.  The shape checks stay here so a
    mis-sized template row stops the generator rather than emitting a descriptor that
    walks off a buffer at run time.
    """
    if i + 1 >= len(STAGES):
        return []
    lanes_p = STAGES[i]['n_par']
    lanes_c = STAGES[i + 1]['n_par']
    hi, lo = max(lanes_p, lanes_c), min(lanes_p, lanes_c)
    out: List[Xfer] = []
    for t in _tensors('hand', i):
        if hi % lo:
            raise ValueError(
                f"stage {i + 1} -> {i + 2}: lane counts {lanes_p} and {lanes_c} are not "
                f"a multiple of one another, so no lane owns a whole number of the "
                f"other side's runs")
        for lanes, side in ((lanes_p, 'producing'), (lanes_c, 'consuming')):
            if t.elems % lanes:
                raise ValueError(
                    f"stage {i + 1} tensor '{t.name}': {t.elems} elements do not divide "
                    f"over the {lanes} lanes of the {side} stage, so no lane owns a "
                    f"whole number of them")
        out.append(Xfer(t.name, t.elems * BYTES_PER_ELEM, t.scatter))
    return out


def _check_handoff_sizing(stages: List[Stage]) -> None:
    """Hold the emitted tables to the rule the latency model already reports.

    compute() bills stage s for `dout_c2c` outbound bytes and stage s+1 for `din_c2c`
    inbound ones, both from _lane_bytes; the C tables are built from the same tensors by a
    different route.  This checks that both still come out of `nbytes // lanes`, and that
    the app moves exactly the next stage's input.
    """
    for i, st in enumerate(stages[:-1]):
        xs = _stage_xfers(i)
        lanes_p, lanes_c = st.n_par, stages[i + 1].n_par
        sent = sum(x.nbytes // lanes_p for x in xs)
        recv = sum(x.nbytes // lanes_c for x in xs)
        want_out = sum(_lane_bytes(t, lanes_p) for t in _tensors('hand', i))
        want_in = sum(_lane_bytes(t, lanes_c) for t in _tensors('hand', i))
        if sent != want_out or recv != want_in:
            raise ValueError(
                f"stage {i + 1} -> {i + 2}: the hand-off tables move {sent} B out and "
                f"{recv} B in, but the model bills {want_out:.0f} B and {want_in:.0f} B")


def _render_stage_tables(stages: List[Stage], lpddr: dict, pool: str) -> List[str]:
    n_st = len(stages)
    bases, n_phys = _stage_bases(stages)
    n_work = sum(s.n_par for s in stages)

    # Stage 0 marks an IDLE cid.  The tables are indexed BY CID, so they carry entries for
    # the padded cids too -- main() reads CID_STAGE[cid] before it knows whether anything
    # runs there.
    cid_stage: List[str] = []
    cid_lane: List[str] = []
    for s in stages:
        cid_stage += [str(s.idx + 1)] * s.n_par + ['0'] * s.pad
        cid_lane += [str(k) for k in range(s.n_par)] + ['0'] * s.pad

    L: List[str] = []
    A = L.append
    A(f'#define N_STAGES    {n_st}')
    A(f'#define N_CLUSTERS  {n_phys}  /* cids the pipeline OCCUPIES: {n_work} working')
    A(f'                          * + {n_phys - n_work} idle, see CID_STAGE */')
    A('')
    A('/* A stage may span several clusters; make a machine too small to host the')
    A(' * pipeline a compile error rather than a silent mis-run. */')
    A('_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER,')
    A('               "pipeline needs more clusters than this arch provides");')
    A('')
    A('/* Stages are numbered from 1, matching the console tables and the split plots.')
    A(' * Every per-stage table below is indexed by that number, so each carries an')
    A(' * unused row 0 -- cheaper to read than a -1 at every lookup. */')
    A(f'enum {{ {", ".join(f"STAGE_{i + 1}" + (" = 1" if i == 0 else "") for i in range(n_st))} }};')
    A('')
    A('/* cid -> (stage, lane).  STAGE 0 MEANS IDLE: the pipeline leaves cids unused so')
    A(' * that the stages touching LPDDR start on a column boundary.  See main(). */')
    A(f'static const uint8_t CID_STAGE[N_CLUSTERS] = {{ {", ".join(cid_stage)} }};')
    A(f'static const uint8_t CID_LANE [N_CLUSTERS] = {{ {", ".join(cid_lane)} }};')
    A('')
    A('/* stage -> first cid, how many clusters run it, how many passes each makes */')
    A(f'static const uint8_t STAGE_BASE [N_STAGES + 2] = '
      f'{{ 0, {", ".join(str(b) for b in bases)}, {n_phys} }};')
    A(f'static const uint8_t STAGE_LANES[N_STAGES + 1] = '
      f'{{ 0, {", ".join(str(s.n_par) for s in stages)} }};')
    A('')
    A('/* tiles > 1: the stage drains itself inside compute_stage_i(), so main() must')
    A(' * not push for it a second time. */')
    A(f'static const uint8_t STAGE_TILES[N_STAGES + 1] = '
      f'{{ 0, {", ".join(str(s.tiles) for s in stages)} }};')
    A('')
    A('/* One entry per tensor crossing a stage boundary, in the order the CONSUMER lays')
    A(' * them out.  A producer sends bytes/lanes and a consumer receives bytes/lanes_nx,')
    A(' * so every destination offset, stride, repetition count and lane step follows from')
    A(' * these and the two stages\' widths; push_to_next derives them.')
    A(' *')
    A(' * src_off is stored because it does NOT follow: a stage\'s outbound tensor is read')
    A(' * out of wherever its own buffer layout put it -- for the encoder stages a pinned')
    A(' * region, not the tail of the last kernel\'s output.')
    A(' *')
    A(' * A LEG IS ONE OF TWO THINGS.  With nseg == 0 it is a plain copy: the split axis')
    A(' * is outermost at both ends, so each share is one contiguous run and the two lane')
    A(' * counts decide everything -- fan-out, fan-in or straight across.')
    A(' *')
    A(' * nseg > 0 IS A LEG THAT PERFORMS A LAYOUT ROW.  Then the leg walks ONE BOX COLUMN')
    A(' * at a time -- `cols` of them per consumer, src_pitch and dst_pitch apart at the')
    A(' * two ends -- and issues SEG[seg0 .. seg0+nseg) inside each, which is where the')
    A(' * row\'s token-order map lives.  The two pitches differ exactly when the row adds or')
    A(' * drops rows: the lattice scatter\'s consumer column is one row longer than its')
    A(' * producer\'s, the observation gather\'s one shorter, and that difference IS the')
    A(' * insert and the crop.  nseg == 0 keeps the plain copy below. */')
    A('typedef struct {')
    A('    uint32_t chunk;      /* bytes of ONE run */')
    A('    uint32_t src_off, dst_off;      /* bytes into the box column, at each end */')
    A('    uint32_t src_stride, dst_stride, reps;')
    A('    uint32_t visits;     /* box columns this segment is REISSUED for; 1 when it')
    A('                          * had its repetition level spare and took the column')
    A('                          * stride into it instead */')
    A('} seg_t;')
    segs = _emit_handoff_segs(stages)
    A(f'static const seg_t SEG[{max(1, len(segs))}] = {{')
    L.extend(segs if segs else ['    { 0u, 0u, 0u, 0u, 0u, 0u, 0u },'])
    A('};')
    A('')
    A('typedef struct {')
    A('    uint32_t bytes;    /* the WHOLE tensor, across every lane of both sides */')
    A('    uint32_t src_off;  /* byte offset in the PRODUCER\'s X_BUF of its own share */')
    A('    uint32_t cols;     /* box columns ONE CONSUMER holds; 0 when nseg == 0 */')
    A('    uint32_t src_pitch, dst_pitch;  /* bytes of one box column, at each end */')
    A('    uint32_t seg0, nseg;            /* this leg\'s segments in SEG */')
    A('} xfer_t;')
    A('')
    first, rows = [0], []
    src = {(i, x.name): _stage_src_off(stages[i], x.name, pool) * BYTES_PER_ELEM
           for i in range(n_st) for x in _stage_xfers(i)}
    xfers = [(i, x) for i in range(n_st) for x in _stage_xfers(i)]
    wb = max((len(str(x.nbytes)) for _, x in xfers), default=1)
    ws = max((len(str(v)) for v in src.values()), default=1)
    tabs = _handoff_segments()
    seg0 = {}
    n = 0
    for i in range(n_st):
        for x in _stage_xfers(i):
            if x.scatter:
                seg0[(i, x.name)] = n
                n += len(tabs[x.scatter][2])
    for i in range(n_st):
        for x in _stage_xfers(i):
            if x.scatter:
                sr, dr, sg = tabs[x.scatter]
                # A consumer's box columns, and one column's bytes at each end.  The
                # column count is the CONSUMER's -- the producer sends that many per
                # consumer, whatever its own share holds.
                cols = _N_BOX // stages[i + 1].n_par
                pitch = _D_MODEL * BYTES_PER_ELEM
                tail = (f'{cols}u, {sr * pitch}u, {dr * pitch}u, '
                        f'{seg0[(i, x.name)]}u, {len(sg)}u')
            else:
                tail = '0u, 0u, 0u, 0u, 0u'
            rows.append(f'    {{ {x.nbytes:>{wb}}u, {src[(i, x.name)]:>{ws}}u, '
                        f'{tail} }},'
                        f'  /* stage {i + 1}: {_c_comment(x.name)} */')
        first.append(len(rows))
    A(f'static const xfer_t STAGE_XFER[{max(1, len(rows))}] = {{')
    L.extend(rows if rows else ['    { 0u, 0u, 0u, 0u, 0u, 0u, 0u },'])
    A('};')
    A(f'static const uint8_t STAGE_XFER_FIRST[N_STAGES + 2] = '
      f'{{ 0, {", ".join(str(f) for f in first)} }};')
    A('')
    A('/* LPDDR legs: whatever must cross the chip boundary.  An empty range means the')
    A(' * stage issues nothing.  These are stored in full, not derived: an LPDDR offset is')
    A(' * assigned by the layout and follows from nothing this file knows.')
    A(' *')
    A(' * `edge` picks hbm_south() or hbm_north(), because DIRECTION DOES NOT DETERMINE')
    A(' * EDGE here: stage 1 writes its window summaries back to the SOUTH node it read its')
    A(' * input from, and stage 4 reads the R3 base off a SOUTH node.  That node ends up')
    A(' * holding [model input | WIN], which is exactly what the R3 base is.')
    A(' *')
    A(' * lpddr_off is an offset INSIDE ONE NODE.  `nodes` and `lanes_per_node` say how the')
    A(' * stage covers the edge, and exactly one of them exceeds 1: a stage no wider than')
    A(' * the edge takes whole nodes and owns its channels, a wider one shares and')
    A(' * lane_step is where each sharer starts inside the node.  One rule covers both:')
    A(' *')
    A(' *     node = (lane / lanes_per_node) * nodes + n')
    A(' *     off  = lpddr_off + (lane % lanes_per_node) * lane_step')
    A(' *')
    A(' * Here the front end takes two nodes per lane, the head one each, and stage 4')
    A(' * shares, two lanes to a node -- reaching sideways to node k/2 rather than')
    A(' * straight down. */')
    A(f'#define LPDDR_EDGE_SOUTH  0u')
    A(f'#define LPDDR_EDGE_NORTH  1u')
    A('typedef struct {')
    A('    uint32_t lpddr_off, l1_off;')
    A('    uint32_t lane_step;  /* bytes into the node this sharer starts at; 0 when the')
    A('                          * lane does not share */')
    A('    uint32_t chunk, lpddr_stride, l1_stride, reps;  /* chunk is ONE descriptor */')
    A('    uint32_t edge;       /* LPDDR_EDGE_SOUTH / LPDDR_EDGE_NORTH */')
    A('    uint32_t nodes, lanes_per_node;')
    A('} lpddr_xfer_t;')
    for name in ('IN', 'OUT'):
        tbl = lpddr[name.lower()]
        rows, first = [], [0]
        for i in range(n_st):
            for x in tbl[i]:
                rows.append(f'    {{ {x.lpddr_off}u, {x.l1_off}u, {x.lane_step}u, '
                            f'{x.chunk}u, {x.lpddr_stride}u, {x.l1_stride}u, {x.reps}u, '
                            f'LPDDR_EDGE_{x.edge.upper()}, {x.nodes}u, '
                            f'{x.lanes_per_node}u }},'
                            f'  /* stage {i + 1}: {_c_comment(x.name)} */')
            first.append(len(rows))
        A(f'static const lpddr_xfer_t LPDDR_{name}_XFER[{max(1, len(rows))}] = {{')
        L.extend(rows if rows else
                 ['    { 0u, 0u, 0u, 0u, 0u, 0u, 0u, LPDDR_EDGE_SOUTH, 1u, 1u },'])
        A('};')
        A(f'static const uint8_t LPDDR_{name}_FIRST[N_STAGES + 2] = '
          f'{{ 0, {", ".join(str(f) for f in first)} }};')
    return L


def _emit_handoff_segs(stages: List[Stage]) -> List[str]:
    """The SEG[] rows, in the order STAGE_XFER's seg0 indices assume.

    Token rows become bytes here and nowhere else: _handoff_segments states each map in
    the unit the model thinks in, and d_model is a property of the tensor, not of the map.
    """
    tabs = _handoff_segments()
    out: List[str] = []
    pitch = _D_MODEL * BYTES_PER_ELEM
    for i in range(len(stages)):
        for x in _stage_xfers(i):
            if not x.scatter:
                continue
            sr, dr, segs = tabs[x.scatter]
            cols = _N_BOX // stages[i + 1].n_par
            for (lab, chunk, so, do, ss, ds, reps) in segs:
                # A SEGMENT THAT DOES NOT USE ITS LEVEL TAKES THE COLUMN AXIS INSTEAD.
                # The descriptor has exactly one repetition level; a rectangle with reps
                # = 1 has it spare, so the column stride goes there and one descriptor
                # covers every column the consumer holds.  A rectangle already walking a
                # fold axis cannot -- the two ends' column pitches DIFFER, by the row this
                # leg inserts or drops, so no single stride pair reaches across a column
                # boundary and the fold axis at once.
                if reps == 1 and cols > 1:
                    reps, ss, ds, visits = cols, sr, dr, 1
                else:
                    visits = cols
                out.append(f'    {{ {chunk * pitch}u, {so * pitch}u, {do * pitch}u, '
                           f'{ss * pitch}u, {ds * pitch}u, {reps}u, {visits}u }},'
                           f'  /* stage {i + 1}: {_c_comment(lab)} */')
    return out


def _render_push_helper(issue: bool = False) -> str:
    sfx = '_issue' if issue else ''
    d2 = f'dma_2d{sfx}'
    ds = f'dma_2d_strided{sfx}'
    return '\n'.join([
        f'{_INLINE} push_to_next{sfx}(uint32_t stage, uint32_t lane,',
        # Line the second parameter row up under the first, derived rather than a literal
        # so it survives the next edit to _INLINE.
        f'{" " * (len(_INLINE) + len(" push_to_next(") + len(sfx))}'
        f'uint32_t dst_base, uint64_t src_base)',
        '{',
        '    uint32_t next     = stage + 1u;',
        '    uint32_t lanes    = STAGE_LANES[stage];',
        '    uint32_t lanes_nx = STAGE_LANES[next];',
        '    uint32_t base_nx  = STAGE_BASE[next];',
        '',
        '    /* The DESTINATION region is filled tensor by tensor in the table\'s order --',
        '     * the consumer\'s inbound tensors are laid out contiguously and in exactly',
        '     * this order, so a running sum reaches each one.  The SOURCE is not: a',
        '     * producer reads each tensor out of wherever its own buffer layout put it,',
        '     * so src_off is carried per tensor and dst_off is derived here. */',
        '    uint32_t dst_off  = 0u;',
        '',
        '    for (uint32_t e = STAGE_XFER_FIRST[stage]; e < STAGE_XFER_FIRST[next]; e++) {',
        '        const xfer_t *x = &STAGE_XFER[e];',
        '        uint32_t src_off = x->src_off;            /* where THIS stage holds it */',
        '        uint32_t own_src = x->bytes / lanes;      /* what this cluster holds */',
        '        uint32_t own_dst = x->bytes / lanes_nx;   /* what one consumer needs */',
        '',
        '        if (x->nseg) {',
        '            /* A LEG THAT PERFORMS A LAYOUT ROW.  One consumer takes x->cols box',
        '             * columns, and a fan-out producer feeds r of them back to back out',
        '             * of its own share -- so consumer k\'s first column is column',
        '             * k * cols of what this lane holds.  Inside a column the segments',
        '             * are the row\'s token-order map, and the two pitches carry the row',
        '             * it inserts or drops. */',
        '            uint32_t r = lanes_nx > lanes ? lanes_nx / lanes : 1u;',
        '            for (uint32_t k = 0; k < r; k++)',
        '                for (uint32_t s = 0; s < x->nseg; s++)',
        '                    for (uint32_t c = 0; c < SEG[x->seg0 + s].visits; c++) {',
        '                        const seg_t *g = &SEG[x->seg0 + s];',
        f'                        {ds}((uint64_t)remote_cid(base_nx + lane * r + k,',
        '                                  dst_base + dst_off + c * x->dst_pitch',
        '                                  + g->dst_off),',
        '                             src_base + (uint64_t)(src_off',
        '                                  + (k * x->cols + c) * x->src_pitch',
        '                                  + g->src_off),',
        '                             g->chunk, g->dst_stride, g->src_stride, g->reps);',
        '                    }',
        '        } else if (lanes_nx > lanes) {',
        '            /* Fan-out: this lane\'s share covers r consecutive consumers back',
        '             * to back, so consumer k takes the k-th own_dst of it.  They are',
        '             * disjoint per producer, so no rotation is needed -- every lane is',
        '             * already writing a different set of clusters. */',
        '            uint32_t r = lanes_nx / lanes;',
        '            for (uint32_t k = 0; k < r; k++)',
        f'                {d2}((uint64_t)remote_cid(base_nx + lane * r + k,',
        '                                          dst_base + dst_off),',
        '                     src_base + (uint64_t)(src_off + k * own_dst), own_dst);',
        '        } else if (lanes > lanes_nx) {',
        '            /* Fan-in: r producers share one consumer, and this lane lands in',
        '             * slot (lane % r) of its inbound region. */',
        '            uint32_t r = lanes / lanes_nx;',
        f'            {d2}((uint64_t)remote_cid(base_nx + lane / r,',
        '                                      dst_base + dst_off',
        '                                      + (lane % r) * own_src),',
        '                 src_base + (uint64_t)src_off, own_src);',
        '        } else {                                    /* straight hand-off */',
        f'            {d2}((uint64_t)remote_cid(base_nx + lane, dst_base + dst_off),',
        '                 src_base + (uint64_t)src_off, own_src);',
        '        }',
        '        dst_off += own_dst;',
        '    }',
        '}',
    ])


# ---------------------------------------------------------------------------
# Per-layer emission
# ---------------------------------------------------------------------------

# ---- Level tables for the permutation kernels ------------------------------
#
# permute_f16p walks a table of (extent, src_step, dst_step) levels, fastest first, with a
# contiguous RUN at the bottom.  This is where each lowered row's index map is written
# down, cut to ONE LANE's share.
#
# WHAT IS PINNED AND WHAT IS CHOSEN.  Every extent, every run and every count here follows
# from the template -- from the row's own dims, from `out_width = pspan * n_bins * S2 * 2`
# and from row 44's `span_prbs * n_bins * ups_in`, which are two independent statements of
# the same 384 and agree.  What the template does NOT pin is the ORDER of the two
# two-dimensional indices: which of (bv, bh) is outer in the box index, and which of
# (kv, kh) is outer within a patch.  A convention is chosen here and used consistently.
# It changes no cycle -- the run length, the run count and both footprints are identical
# either way -- and this app moves placeholder data and checks no result, so what a
# different convention would change is a numerical answer nobody reads.
#
# ROW 3, the front-end unpack, is the one that needs all six levels: a 2-D tile index over
# a 2-D antenna plane cannot be one uniform stride, so each half of each pair is its own
# level.  Its run is P * S2 * 2 = 48 and is CONTIGUOUS AT BOTH ENDS -- the pilot, symbol
# and re/im axes are adjacent in the received grid and adjacent again in the patch payload.
_CELERE_N_FULL = 11        # n_full = floor(n_span / W)
_CELERE_L_L = 13           # L_l = W + 1, spans plus their own summary
_CELERE_L_REM = 6          # L_rem = n_span - n_full*W + 1
_RA_U_V, _RA_U_H = 4, 16                 # 4 x 16 dual-pol UPA, pol already on the batch axis
_RA_G_V, _RA_G_H = 2, 2                  # antennas per patch = g_v * g_h = 4
_PILOTS_PER_PRB, _SYMS, _RI = 12, 2, 2   # P, S2, re/im
_SPAN_PRBS = 2

# ---- Segment tables for the layout rows a HAND-OFF performs ----------------
#
# A leg's descriptor walks ONE BOX COLUMN at a time, because that is the axis every stage
# splits and the axis a lane stores outermost.  Inside a column the unit is a TOKEN ROW of
# d_model, and a layout row is then a map from the producer's token order to the
# consumer's -- which is what these segments are.  Each is one dma_2d_strided: `reps` runs
# of `chunk` token rows, walking the source by src_stride and the destination by
# dst_stride, both from the segment's own base.  Everything is in TOKEN ROWS here; the
# emitter scales by d_model and the element size.
#
# A SEGMENT PER RECTANGLE, for the same reason _permute_levels needs one: a level walks at
# ONE uniform stride, and neither of these maps is uniform end to end.
_D_MODEL, _N_BOX = 256, 16


def _handoff_segments() -> Dict[str, Tuple[int, int, List[Tuple[str, int, int, int, int, int, int]]]]:
    """name -> (producer token rows per column, consumer's, [segments]).

    A segment is (label, chunk, src_off, dst_off, src_stride, dst_stride, reps).
    """
    n_full, l_l, l_rem = _CELERE_N_FULL, _CELERE_L_L, _CELERE_L_REM
    spans, summ = l_l - 1, l_rem - 1        # spans in a whole window, in the short one
    n_summ = n_full + 1                     # every window contributes one summary
    n_span = n_full * spans + summ          # 137
    return {
        # ROW 9, the lattice scatter.  Row 7 embedded PAT and WIN CONCATENATED on i_2, so
        # the producer holds [every span | every summary]; row 9's own description asks for
        # window-major [w0 spans | w0 summary | w1 ... | spell].  Interleaving one summary
        # after every window's spans is the whole of it, and it takes four rectangles: the
        # whole windows' spans and summaries walk at a uniform stride, the short last
        # window's do not.  THE SPELL ROW HAS NO SOURCE -- it is learned, one per box
        # column -- so 149 rows cross and the consumer's 150th is left alone.
        'row9_lattice': (n_span + n_summ, n_span + n_summ + 1, [
            ('whole windows, spans',    spans, 0,             0,     spans, l_l,   n_full),
            ('short window, spans',     summ,  n_full * spans, n_full * l_l, summ, summ, 1),
            ('whole windows, summaries', 1,    n_span,        spans, 1,     l_l,   n_full),
            ('short window, summary',    1,    n_span + n_full, n_full * l_l + summ, 1, 1, 1),
        ]),
        # ROW 33, the observation-carrier gather.  It drops the spell row, which row 9 put
        # LAST in the column, so what crosses is each column's leading n_obs rows -- and
        # the crop lives in the PITCH DIFFERENCE, the producer's column being one row
        # longer than the consumer's.  One rectangle, and it is only not a plain copy
        # because a lane holds more than one column.
        'row33_crop': (n_span + n_summ + 1, n_span + n_summ, [
            ('drop the spell row', n_span + n_summ, 0, 0, n_span + n_summ, n_span + n_summ, 1),
        ]),
    }


# One permute_f16p call: (label, run, levels fastest-first, src_off, dst_off).  A level is
# (count, src_step, dst_step) in ELEMENTS, and the two offsets are where the call starts in
# the row's source and destination buffers -- both 0 for a row that takes one call.
PermCall = Tuple[str, int, List[Tuple[int, int, int]], int, int]


def _permute_levels(row: LayerRow, n_par: int) -> List[PermCall]:
    """One (label, run, levels fastest-first, src_off, dst_off) per CALL, for ONE LANE.

    A row takes more than one call for either of two reasons.  Its shape may not be one
    rectangle -- rows 22 and 28 have the n_full whole folds and one short L_rem fold, row
    44 a cropped last token -- and emitting each rectangle separately is what keeps every
    level's stride uniform.  Or its op may name SEVERAL TENSORS (PERMUTE_TENSORS): row 15
    gathers Q, K and U out of one QKV and row 22 folds Q and K, and each of those is its
    own walk.  The two multiply: row 22 is 2 tensors x 2 rectangles = 4 calls.

    EVERY STRIDE IS THIS LANE'S, NOT THE MODEL'S.  _row_shape divides every extent by
    n_par * tiles, so the buffer these walk holds n_box / n_par boxes; a stride built from
    the full box grid steps n_par times too far and runs off the end of X_BUF.  The lane's
    own counts (box, bh_lane, bv_lane) are what every stride is built from.

    THE OFFSETS ARE WHAT PLACE A CALL, and a call that is not the first of its row needs
    them: the remainder fold starts where the whole folds stopped, and tensor t starts one
    whole tensor past tensor t-1.  Without them every call would restart at the row's base
    and overwrite the one before it.
    """
    n = row.excel_row
    ind, outd = row.in_dims, row.out_dims

    if n == 3:
        n_bv, n_bh = _RA_U_V // _RA_G_V, _RA_U_H // _RA_G_H
        n_box = n_bv * n_bh
        n_span = outd[1]
        fp = ind[1]                                   # padded pilot subcarriers
        pspan = fp // n_span                          # 24 = span_prbs * P
        per_ant = fp * _SYMS * _RI                    # elements one antenna contributes
        run = _PILOTS_PER_PRB * _SYMS * _RI           # 48, contiguous both ends
        n_bins = _RA_G_V * _RA_G_H
        ow = pspan * n_bins * _SYMS * _RI             # out_width = 384
        # This lane owns n_box / n_par boxes.  The split takes the HORIZONTAL tile axis
        # first and only spills into the vertical one once it is exhausted, so a lane's
        # boxes stay a rectangle and both levels keep a uniform stride.
        bh_lane = n_bh // n_par if n_par <= n_bh else 1
        bv_lane = n_bv if n_par <= n_bh else max(1, n_bv // (n_par // n_bh))
        # The destination holds THIS LANE'S boxes back to back, so a row of them is
        # bh_lane wide and a span is bh_lane * bv_lane of them -- not n_bh and n_box, which
        # are the whole grid's and would leave n_par - 1 boxes' worth of hole per step.
        return [('whole', run, [(_RA_G_H, _RA_U_V * per_ant, run),
                     (_RA_G_V, per_ant, _RA_G_H * run),
                     (_SPAN_PRBS, _PILOTS_PER_PRB * _SYMS * _RI, n_bins * run),
                     (bh_lane, _RA_G_H * _RA_U_V * per_ant, ow),
                     (bv_lane, _RA_G_V * per_ant, bh_lane * ow),
                     (n_span, pspan * _SYMS * _RI, bh_lane * bv_lane * ow)], 0, 0)]

    if n == 15:
        # QG[b, h, t, d] = QKV[t, b, stripe*d_model + h*d_head + d].  Q, K and V are the
        # three stripes of ONE QKV, so this is three calls: stripe s reads at column
        # s*d_model and writes the s-th gathered tensor.
        #
        # THE THREE TOGETHER FILL QKV, WHICH IS ALSO WHAT THEY READ (BIND pins both ends
        # there), so a gathered block lands on source the other stripes have not read yet.
        # The apps move placeholder data and check no result, so what this costs is a
        # numerical answer nobody reads; making it exact means staging through the arena,
        # which is a partition decision -- see the module header's L1 budget.
        n_box, n_tok, d_model = outd[0], outd[2], ind[2]
        heads, d_head = outd[1], outd[3]
        box = max(1, n_box // n_par)
        row_b = box * 3 * d_model            # one token of QKV IN THIS LANE, q|k|v
        tens = box * heads * n_tok * d_head  # one gathered tensor
        levels = [(n_tok, row_b, d_head),
                  (heads, d_head, n_tok * d_head),
                  (box, 3 * d_model, heads * n_tok * d_head)]
        return [(f'{t}G', d_head, levels, s * d_model, s * tens)
                for s, t in enumerate(('Q', 'K', 'U'))]

    if n == 22:
        # QF[a, h, m, d] = Q[w*L_l + m, b, h*d_head + d], a = w*box + b.  TWO tensors, Q
        # and K -- the first two stripes of QKV -- and TWO rectangles each, the n_full
        # whole folds and the short L_rem one, so four calls.  The remainder starts where
        # the whole folds stopped at BOTH ends: fold n_full of the source, and the
        # n_full * bulk elements of the destination they filled.
        n_box, d_model = outd[0], outd[2]
        heads, d_head = 2, d_model // 2
        box = max(1, n_box // n_par)
        row_b = box * 3 * d_model
        n_full, l_l, l_rem = _CELERE_N_FULL, _CELERE_L_L, _CELERE_L_REM
        bulk = box * heads * l_l * d_head             # one whole fold, in the destination
        tens = n_full * bulk + box * heads * l_rem * d_head        # one folded tensor
        out: List[PermCall] = []
        for t, name in enumerate(('Q', 'K')):
            out.append((f'{name}F full folds', d_head,
                        [(l_l, row_b, d_head),
                         (heads, d_head, l_l * d_head),
                         (box, 3 * d_model, heads * l_l * d_head),
                         (n_full, l_l * row_b, bulk)],
                        t * d_model, t * tens))
            out.append((f'{name}F remainder fold', d_head,
                        [(l_rem, row_b, d_head),
                         (heads, d_head, l_rem * d_head),
                         (box, 3 * d_model, heads * l_rem * d_head)],
                        t * d_model + n_full * l_l * row_b, t * tens + n_full * bulk))
        return out

    if n == 28:
        # UD[b, w*L_l + m, h*d_head + d] = UL[(w*box + b), h, m, d].  Two rectangles again.
        # THE DESTINATION GAP IS THE PROOF THE SECOND ONE IS PLACED: the box stride is the
        # whole n_tok * d_model while the whole folds cover only n_full * l_l * d_model of
        # it, and what is left is exactly the remainder fold's l_rem tokens.
        heads, l_l, d_head = ind[1], ind[2], ind[3]
        n_box, n_tok, d_model = outd[0], outd[1], outd[2]
        box = max(1, n_box // n_par)
        n_full, l_rem = _CELERE_N_FULL, _CELERE_L_REM
        return [('full folds', d_head,
                 [(heads, l_l * d_head, d_head),
                  (l_l, d_head, d_model),
                  (n_full, box * heads * l_l * d_head, l_l * d_model),
                  (box, heads * l_l * d_head, n_tok * d_model)], 0, 0),
                ('remainder fold', d_head,
                 [(heads, l_rem * d_head, d_head),
                  (l_rem, d_head, d_model),
                  (box, heads * l_rem * d_head, n_tok * d_model)],
                 n_full * box * heads * l_l * d_head, n_full * l_l * d_model)]

    if n == 40:
        # ZS: fold order -> [physical | summary] on o_2, boxes on o_3.  Run = d_model, and
        # both ends are contiguous over it, so this is one level plus the run.
        n_tok, n_box, d_model = outd[1], outd[2], outd[3]
        box = max(1, n_box // n_par)
        return [('whole', d_model,
                 [(n_tok, box * d_model, box * d_model),
                  (box, d_model, d_model)], 0, 0)]

    if n == 44:
        # BLK: every payload row of out_width ungroups into span_prbs*n_bins per-PRB blocks
        # of ups_in, and the physical segment is CROPPED by a contiguous tail -- 1188 kept
        # of the 1192 the tokens make, so the last token keeps only 4 of its 8 blocks.  The
        # bulk and that short last token are two calls, the same shape as the two static
        # folds rows 22 and 28 take.
        n_tok, n_box, ow = ind[1], ind[2], ind[3]
        blocks_out, ups_in = outd[1], outd[3]
        per_row = ow // ups_in                        # span_prbs * n_bins = 8
        box = max(1, n_box // n_par)
        full, tail = divmod(blocks_out, per_row)
        out = [('bulk', ups_in, [(per_row, ups_in, box * ups_in),
                                 (box, ow, ups_in),
                                 (full, box * ow, per_row * box * ups_in)], 0, 0)]
        if tail:
            out.append(('crop tail', ups_in, [(tail, ups_in, box * ups_in),
                                              (box, ow, ups_in)],
                        full * box * ow, full * per_row * box * ups_in))
        return out

    raise ValueError(f'row {n} is in LAYOUT_LOWERING but has no level table; '
                     f'_permute_levels must gain one')


def _walk_bounds(run: int, extents: List[Tuple[int, int]], base: int) -> Tuple[int, int]:
    """[lowest, one-past-highest] element a levelled walk touches, from `base`.

    Steps are SIGNED -- a level that unfolds an axis walks backwards -- so the two ends
    accumulate separately; a plain sum would net a backward level against a forward one and
    report a span narrower than the walk really covers.
    """
    lo = hi = base
    for n, step in extents:
        span = (n - 1) * step
        if span < 0:
            lo += span
        else:
            hi += span
    return lo, hi + run


def _check_permute_extents(stages: List[Stage], pool: str) -> None:
    """Every permute call must stay inside X_BUF.  Raises if one does not.

    WHY THIS EXISTS.  A level table is data, and a stride built from the model's extents
    rather than the lane's overruns silently: the cores walk past X_BUF, through W_BUF and
    over the barrier counter, and the app hangs tens of thousands of cycles later with all
    of them asleep in wfi and nothing in the trace pointing back here.  The walk is
    arithmetic the generator already has, so it can say so before the app is written.
    """
    nx, _nw = _scratch_elems(stages, pool)
    for st in stages:
        for item in st.items:
            if LAYOUT_LOWERING.get(item.row.excel_row) != 'permute':
                continue
            base_x, base_y = _item_offsets(item, st, pool)
            for label, run, levels, s_off, d_off in _permute_levels(item.row, st.n_par):
                for end, base, col in (('source', base_x + s_off, 1),
                                       ('destination', base_y + d_off, 2)):
                    lo, hi = _walk_bounds(run, [(l[0], l[col]) for l in levels], base)
                    if lo >= 0 and hi <= nx:
                        continue
                    raise ValueError(
                        f'stage {st.idx + 1} row {item.row.excel_row} '
                        f'({label!r}): its {end} walk covers [{lo}, {hi}) of an X_BUF of '
                        f'{nx} elements.  A permute level built from the model\'s extents '
                        f'instead of this lane\'s (n_par = {st.n_par}) steps too far -- '
                        f'see _permute_levels.')


def _layer_cycles(ms: float, cycles_per_ms: float) -> int:
    return max(1, int(ms * cycles_per_ms))


def _render_item(item: Item, idx: int, st: Stage, pool: str, bench: bool, dbl: bool,
                 cycles_per_ms: float, x_off: str = '') -> List[str]:
    """C statement(s) for one item of a stage.

    x_off shifts the X operand -- used on the first item of a tiled run, which reads its
    own tile out of the whole region the previous layer wrote.

    Both activation addresses come from _item_offsets, which resolves the row against the
    stage's buffer layout; the offset a transfer later reads the result from goes through
    the same map, so the two cannot drift.  The B operand comes from _stage_w_layout,
    which gives every row its OWN region of W_BUF.
    """
    base = 'X_BUF[buf]' if dbl else 'X_BUF'
    x_elems, y_elems = _item_offsets(item, st, pool)
    off = lambda n: (f' + {n}u' if n else '')
    xbuf = base + off(x_elems) + (f' + {x_off}' if x_off else '')
    ybuf = base + off(y_elems)
    w_layout = st.w_layout or _stage_w_layout(st, pool)
    wbuf = 'W_BUF' + off(w_layout.get(idx, 0))
    work: List[str] = []

    row = item.row
    sh = _row_shape(row, st.n_par, item.tiles)
    c = sh['cls']
    x_reg, y_reg = BIND.get((st.idx, item.excel_row), ('', ''))
    where = ''
    if x_reg or y_reg:
        # Say which pinned region the operands are, so the emitted call can be read
        # against the model without going back to the generator's tables.
        where = (f'; in place in {x_reg}' if x_reg and x_reg == y_reg else
                 '; ' + ', '.join(p for p in (f'X from {x_reg}' if x_reg else '',
                                              f'Y into {y_reg}' if y_reg else '') if p))
    tag = _c_comment(f'{row.op} [{row.kernel}]{where}')
    if c == 'gemm':
        N, P = sh['N'], sh['P']
        M = _gemm_rows(sh['M'], pool)
        note = ('' if M == sh['M'] else
                f': M {sh["M"]} padded to {M}, a multiple of the '
                f'{POOL_REDMULE_TILES[pool]} tensor engines')
        if pool in TENSOR_ENGINE_POOLS:
            call = (f'redmule_synch_parallel({xbuf}, {wbuf}, {ybuf}, '
                    f'{M}u, {N}u, {P}u, GEMM, false, 0);')
        else:
            call = (f'matmul_4x2_parallel_f16vec({xbuf}, {wbuf}, {ybuf}, '
                    f'{M}u, {N}u, {P}u, cid, nt);')
        work.append(f'{call}  /* {tag}{_c_comment(note)} */')
        work.append('flex_intra_cluster_sync();')
    elif c in ('softmax', 'layernorm'):
        note = ('' if sh['N'] == sh['N_true'] else
                f': N {sh["N_true"]} padded to {sh["N"]}, the kernel steps 4 columns '
                f'and needs an even row stride for its v2h accesses')
        work.append(f'{c}_parallel_2x4_f16vec({xbuf}, {ybuf}, '
                    f'{sh["M"]}u, {sh["N"]}u, cid, nt);  '
                    f'/* {tag}{_c_comment(note)} */')
        work.append('flex_intra_cluster_sync();')
    elif c in ('gelu', 'relu'):
        work.append(f'{c}_f16({ybuf}, {sh["size"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'add':
        work.append(f'axpy_f16vecp_local_unrolled4(0x3C00u, {xbuf}, {ybuf}, '
                    f'{sh["len"]}u);  /* {tag}: Y += X */')
    elif c == 'free':
        # A VIEW: its operand is already in the order the consumer wants, so there is
        # nothing to emit.  The comment stays, because the row is a real step of the
        # network and a reader looking for it should find it here rather than conclude it
        # was forgotten.
        work.append(f'/* {tag}: {_c_comment(sh["why"])} -- no code, no cycles. */')
    elif c == 'pool':
        work.append(f'pool_mean_f16p({xbuf}, {ybuf}, {sh["n_in"]}u, {sh["n_out"]}u, '
                    f'{sh["win"]}u, {sh["lane"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'permute':
        # One call per rectangle -- see _permute_levels for why some rows need two -- and
        # one level table each, emitted beside the call so the map can be read here.
        # A SYNC AFTER EVERY RECTANGLE: the kernel ends with no barrier of its own, so
        # without one a core would enter the next call while another is still in this
        # one, and the two rectangles of a split row address the same buffer.
        calls = _permute_levels(row, st.n_par)
        for j, (label, run, levels, s_off, d_off) in enumerate(calls):
            nm = f'lv{item.excel_row}_{idx}_{j}'
            work.append(f'static const perm_level_t {nm}[] = {{')
            for (cnt, ss, ds) in levels:
                work.append(f'    {{ {cnt}u, {ss}, {ds} }},')
            work.append(f'}};  /* {_c_comment(label)} */')
            work.append(f'permute_f16p({xbuf}{off(s_off)}, {ybuf}{off(d_off)}, {run}u, '
                        f'{nm}, {len(levels)}u, cid, nt);  /* {tag} */')
            work.append('flex_intra_cluster_sync();')
    elif c == 'wait':
        # The stand-in keeps the rendezvous the kernel it replaces would have ended on:
        # every core spins the same count, so the barrier is nearly free.
        work.append(f'mempool_wait({_layer_cycles(item.ms, cycles_per_ms)}UL);  '
                    f'/* {tag}: timing stand-in -- {_c_comment(sh["why"])} */')
        work.append('flex_intra_cluster_sync();')
    else:
        raise ValueError(f"row {row.excel_row}: nothing emits '{row.kernel}'")

    if not bench:
        return work
    # Trace exactly this item: two CSR writes around the work, no barrier added.
    return (['{', '    mempool_start_benchmark();']
            + ['    ' + w for w in work]
            + ['    mempool_stop_benchmark();', '}'])


def _tile_drain_off(st: Stage, pool: str) -> int:
    """Element offset a self-draining stage writes its tiles from.

    A tiled stage has exactly one outbound LPDDR tensor and drains it itself, so the
    address is that tensor's source, resolved through the layout map everything else uses.
    """
    outs = _tensors('hout', st.idx)
    if len(outs) != 1:
        raise ValueError(
            f'stage {st.idx + 1} is tiled and drains itself, so it must have exactly one '
            f'outbound LPDDR tensor; it has {len(outs)}')
    return _stage_src_off(st, outs[0].name, pool)


def _render_tile_drain(ind: str, dbl: bool, out_off: int) -> List[str]:
    """The per-tile LPDDR write a self-draining stage issues from inside its own compute.

    Emitted in --wait too: that build is the timing twin of the real one, so it has to
    move the same bytes.

    The cluster's OWN position picks the node and only `t` walks inside it, so each
    cluster owns one channel and reaches it over one hop: hbm_north() indexes the north
    edge by cluster x and the stage spans the whole top row, which the prologue's
    _Static_asserts hold the machine to.
    """
    # This source is the one DMA endpoint in the app whose alignment does not follow from
    # a tensor size, so it is checked rather than trusted; unaligned, the leg runs at half
    # rate (see DMA_LINE_ELEMS).
    if out_off % DMA_LINE_ELEMS:
        raise ValueError(
            f'the tiled stage drains from X_BUF + {out_off} elements '
            f'({out_off * BYTES_PER_ELEM} B), which is '
            f'{out_off * BYTES_PER_ELEM % (DMA_LINE_ELEMS * BYTES_PER_ELEM)} mod '
            f'{DMA_LINE_ELEMS * BYTES_PER_ELEM} -- the leg would run at half rate. '
            f'_out_off should have padded it; check that it did not get bypassed.')
    ybuf = f'({"X_BUF[buf]" if dbl else "X_BUF"} + {out_off}u)'
    return [
        f'{ind}if (flex_is_dm_core())',
        f'{ind}    dma_2d(hbm_{LPDDR_OUT_EDGE}(hbm_node,',
        f'{ind}                    LPDDR_OUTPUT_OFFSET + t * TILE_OUT_BYTES),',
        f'{ind}           (uint64_t)(uintptr_t){ybuf}, TILE_OUT_BYTES);',
        f'{ind}flex_intra_cluster_sync();',
    ]


# The local _render_tile_drain reads: this cluster's own node on LPDDR_OUT_EDGE.
_DRAIN_NODE_DECL = (
    'uint32_t hbm_node = get_pos(flex_get_cluster_id()).x;'
    f'  /* this cluster\'s {LPDDR_OUT_EDGE} neighbour */')


def _tile_in_elems(st: Stage, pool: str) -> int:
    """One tile's input footprint for the first item of the tiled run, or 0.

    The whole region arrives at once and each pass reads its own slice; only the first
    tiled item touches it, the rest chain off intermediates.
    """
    if st.tiles <= 1:
        return 0
    return _row_footprint(st.items[st.tile_from].row, st.n_par,
                          st.items[st.tile_from].tiles, pool)[0]


def _render_compute_body_wait(st: Stage, cycles_per_ms: float, dbl: bool,
                              pool: str) -> List[str]:
    L: List[str] = []
    if st.n_par > 1:
        L.append(f'    /* {st.n_par} lanes split {_c_comment(st.split)}; each spin '
                 f'below is this cluster\'s 1/{st.n_par} share */')
    for n, item in enumerate(st.items):
        if st.tiles > 1 and n == st.tile_from:
            L.append(f'    {_DRAIN_NODE_DECL}')
            L.append(f'    for (uint32_t t = 0; t < {st.tiles}u; t++) {{')
        ind = '        ' if (st.tiles > 1 and n >= st.tile_from) else '    '
        L.append(f'{ind}mempool_wait({_layer_cycles(item.ms, cycles_per_ms)}UL);  '
                 f'/* {_c_comment(item.label)} */')
    if st.tiles > 1:
        L += _render_tile_drain('        ', dbl, _tile_drain_off(st, pool))
        L.append('    }')
    return L


def _render_compute_body_real(st: Stage, pool: str, bench: bool, dbl: bool,
                              cycles_per_ms: float) -> List[str]:
    L: List[str] = []
    if st.n_par > 1:
        L.append(f'    /* {st.n_par} lanes split {_c_comment(st.split)}; each kernel '
                 f'below is this cluster\'s 1/{st.n_par} share */')
    tile_in = _tile_in_elems(st, pool)
    for n, item in enumerate(st.items):
        if st.tiles > 1 and n == st.tile_from:
            L.append(f'    /* {st.tiles} tiles from here on: one tile\'s activations are')
            L.append('     * live at a time, and the stage writes each tile out itself. */')
            L.append(f'    for (uint32_t t = 0; t < {st.tiles}u; t++) {{')
        ind = '        ' if (st.tiles > 1 and n >= st.tile_from) else '    '
        off = f't * {tile_in}u' if (n == st.tile_from and st.tiles > 1 and tile_in) else ''
        for line in _render_item(item, n, st, pool, bench, dbl, cycles_per_ms, off):
            L.append(ind + line)
    if st.tiles > 1:
        L += _render_tile_drain('        ', dbl, _tile_drain_off(st, pool))
        L.append('    }')
    return L


def _render_compute_stage_fns(stages: List[Stage], pool: str, wait: bool,
                              cycles_per_ms: float,
                              bench: bool, dbl: bool) -> List[str]:
    L: List[str] = []
    if wait:
        L.append('/* ---- Per-cluster compute: one mempool_wait() per layer (timing only). */')
    else:
        L.append('/* ---- Per-cluster compute: real baremetal kernel calls, X -> Y.')
        L.append('   ROPE IS THE ONLY STAND-IN LEFT: it alone has no kernel in the tree,')
        L.append('   and it alone is a mempool_wait() here.  pool runs pool_mean_f16p and')
        L.append('   every layout row is a permute_f16p, a hand-off descriptor or a view')
        L.append('   -- see LAYOUT ROWS in the generator\'s docstring for which is which. */')
    arg = 'uint32_t buf' if dbl else 'void'
    for st in stages:
        work = ', '.join(i.label for i in st.items)
        L.append(f'/* stage {st.idx + 1} : [{_c_comment(work)}] */')
        L.append(f'{_INLINE} compute_stage_{st.idx + 1}({arg})')
        L.append('{')
        if wait:
            if dbl:
                L.append('    (void)buf;   /* nothing is touched: this is the timing twin */')
            L.extend(_render_compute_body_wait(st, cycles_per_ms, dbl, pool))
        else:
            L.append('    uint32_t cid = mempool_get_core_id();    /* in-cluster core id */')
            L.append('    uint32_t nt  = mempool_get_core_count();  /* cores per cluster */')
            L.append('    (void)cid; (void)nt;')
            if st.tiles > 1:
                L.append(f'    {_DRAIN_NODE_DECL}')
            L.extend(_render_compute_body_real(st, pool, bench, dbl, cycles_per_ms))
        L.append('}')
        L.append('')
    return L


def _render_scratch_buffers(stages: List[Stage], pool: str, dbl: bool) -> List[str]:
    """The two L1 buffers: one for activations, one for weights.

    X is sized to the largest '@end' any stage's layout reports; every call site carries
    its own offset out of that same map, so there is no app-wide split point to keep in
    step.
    """
    nx, nw = _scratch_elems(stages, pool)
    d = '[2]' if dbl else ''
    par = ('  Index [2] is the ITEM PARITY (compute uses one, the DMA the other).'
           if dbl else '')
    attr = '__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")))'
    attr_w = ('__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), '
              'section(".l1"), used))')
    return [
        '/* Per-cluster L1 (placeholder data -- only cycles are measured).',
        ' *',
        ' * X_BUF is the one activation buffer, shared by every stage and carved differently',
        ' * by each: inbound tensors first, then the regions the stage pins for rows that',
        ' * read an operand produced several rows earlier, then the scratch arena.  It is',
        ' * also the DMA endpoint at both ends.',
        ' * W_BUF holds operand B for every GEMM, resident and one region PER ROW, so the',
        ' * buffer is their SUM.  It does not divide by the cluster count -- the split is',
        ' * over rows -- and it is not doubled in cwt, because nothing writes it.  On a',
        ' * matmul row operand B is an ACTIVATION (K, V or UG) that this lowering reads',
        f' * through W_BUF, so it is reserved here as well as arriving in X_BUF.{par} */',
        f'#define X_ELEMS     {nx}U',
        f'#define W_ELEMS     {nw}U',
        f'static __fp16 X_BUF{d}[X_ELEMS] {attr};',
        f'static __fp16 W_BUF[W_ELEMS] {attr_w};',
    ]


def _render_buffer_init(dbl: bool) -> List[str]:
    xb = 'X_BUF[b][i]' if dbl else 'X_BUF[i]'
    ind = '    ' if dbl else ''
    L = ['    /* Prime the buffers with fp16 1.0 before timing (avoids NaN/subnormal FP',
         '     * stalls; gives the normal-path cycle counts).  All cores, untimed. */']
    if dbl:
        L.append('    for (uint32_t b = 0; b < 2u; b++) {')
    L += [f'{ind}    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS; '
          f'i += mempool_get_core_count()) {xb} = (__fp16)1.0f;']
    if dbl:
        L.append('    }')
    L += ['    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; '
          'i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;',
          '    flex_intra_cluster_sync();']
    return L


# ---------------------------------------------------------------------------
# LPDDR layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LpddrXfer:
    """One LPDDR leg, in bytes.  Unlike a hand-off this cannot be derived: an LPDDR offset
    is assigned by the layout below and follows from nothing the C knows, so it is stored.

    `lpddr_off`/`l1_off` name the two ends whichever way the copy runs.
    """
    name: str
    lpddr_off: int
    l1_off: int
    lane_step: int        # bytes added to the LPDDR end, times this cluster's lane
    chunk: int            # bytes moved PER NODE
    lpddr_stride: int
    l1_stride: int
    reps: int
    edge: str             # 'south' / 'north' -- which hbm_*() the emitted leg calls
    nodes: int            # consecutive nodes one lane covers
    lanes_per_node: int   # lanes sharing one node; exactly one of the two exceeds 1


def _leg_edge(t: Tensor, direction: str) -> str:
    """Which LPDDR edge one leg addresses.  A Tensor may name it; otherwise the
    direction's default applies."""
    return t.edge or (LPDDR_IN_EDGE if direction == 'hin' else LPDDR_OUT_EDGE)


def _leg_fanout(n_par: int, edge: str, what: str) -> Tuple[int, int]:
    """(nodes per lane, lanes per node) for a stage of `n_par` lanes on `edge`.

    Exactly one of the two is greater than 1.  A stage no wider than the edge takes whole
    nodes and owns its channels; a wider one shares, and the sharing is charged for in
    _lpddr_leg_ms.  A ratio that does not divide would leave a lane owning a fraction of a
    node that no descriptor can express, so it is refused here rather than emitted.
    """
    nodes = HBM_CHAN_PLACEMENT[edge]
    if not nodes:
        raise ValueError(f"{what}: the {edge} edge has no LPDDR nodes")
    if n_par <= nodes:
        if nodes % n_par:
            raise ValueError(
                f"{what}: {n_par} lanes do not divide the {edge} edge's {nodes} nodes, "
                f"so a lane would own a fractional number of them")
        return nodes // n_par, 1
    if n_par % nodes:
        raise ValueError(
            f"{what}: the {edge} edge's {nodes} nodes do not divide {n_par} lanes, so a "
            f"node would hold a fractional number of lanes' shares")
    return 1, n_par // nodes


def _lpddr_layout(stages: List[Stage],
                  pool: str) -> Tuple[Dict[str, List[List[LpddrXfer]]], dict]:
    """Assign an LPDDR offset to every tensor that leaves or enters the chip.

    Returns ({'in': [[LpddrXfer, ...] per stage], 'out': ...}, #defines).

    THE OFFSETS ARE PER NODE and THE NODE IS THE UNIT OF ACCOUNTING, not the lane: a lane
    covers `t.nodes` consecutive nodes from lane*nodes, so a node holds
    elems / (n_par * nodes).  That is what lets the 2-lane front end and the 8-lane FFN
    stage place the SAME tensor names at the SAME node offsets and have both be right.

    ONE OFFSET SPACE PER EDGE, not per direction: the south edge carries the model input
    in, the WIN spill out and the R3 base back in.
    """
    align = lambda n: (n + 63) & ~63
    offsets: Dict[str, int] = {}
    ends: Dict[str, int] = {}                 # per-EDGE running offset, inside one node

    def own(t: Tensor, key: str) -> int:
        """Bytes ONE NODE holds of this tensor.

        Independent of the stage's lane count on purpose: a node's contents do not depend
        on who reads them.
        """
        return t.elems * BYTES_PER_ELEM // HBM_CHAN_PLACEMENT[_leg_edge(t, key)]

    for i in range(len(stages)):
        for key in ('hin', 'hout'):
            for t in _tensors(key, i):
                # Refuses a lane/node ratio no descriptor can express, whichever way it
                # goes.  Called for its check even when the tensor is already placed.
                _leg_fanout(stages[i].n_par, _leg_edge(t, key),
                            f"stage {i + 1} tensor '{t.name}'")
                if t.name in offsets:
                    continue
                edge = _leg_edge(t, key)
                offsets[t.name] = ends.get(edge, 0)
                ends[edge] = align(offsets[t.name] + own(t, key))

    def leg(i: int, t: Tensor, key: str, local_off: int) -> LpddrXfer:
        """One LPDDR transfer: one contiguous burst per descriptor.

        `chunk` is what ONE descriptor moves -- a whole node when the lane owns nodes
        outright, a node divided by the sharers when it does not.  Nothing strides inside
        a burst, so reps = 1; `lane_step` is where a sharing lane starts, 0 otherwise.
        """
        nodes, lanes = _leg_fanout(stages[i].n_par, _leg_edge(t, key), t.name)
        chunk = own(t, key) // lanes
        return LpddrXfer(t.name, offsets[t.name], local_off,
                         chunk if lanes > 1 else 0,
                         chunk, chunk, chunk, 1,
                         _leg_edge(t, key), nodes, lanes)

    tables: Dict[str, List[List[LpddrXfer]]] = {'in': [], 'out': []}
    for i, st in enumerate(stages):
        # Both ends of every leg now come from the stage's own buffer layout: an inbound
        # tensor lands in its named slot, an outbound one is read from wherever the layout
        # put it -- a pinned region for the WIN spill, the last layer's output otherwise.
        ins = [leg(i, t, 'hin', st.layout[t.name] * BYTES_PER_ELEM)
               for t in _tensors('hin', i)]
        # A tiled stage's output is written by its own tile loop, not from this table.
        outs = ([] if st.tiles > 1 else
                [leg(i, t, 'hout', _stage_src_off(st, t.name, pool) * BYTES_PER_ELEM)
                 for t in _tensors('hout', i)])
        tables['in'].append(ins)
        tables['out'].append(outs)

    last = stages[-1]
    out_t = LPDDR_OUT[len(stages) - 1][0]
    defines = {
        'LPDDR_OUTPUT_OFFSET': offsets[out_t.name],
        'TILE_OUT_BYTES': out_t.elems * BYTES_PER_ELEM // (last.n_par * last.tiles),
    }
    return tables, defines


# ---------------------------------------------------------------------------
# C renderers (one per mode)
# ---------------------------------------------------------------------------

def _render_lpddr_asserts(stages: List[Stage]) -> List[str]:
    """Compile-time checks for every stage that touches LPDDR.

    What an LPDDR leg needs is that the stage lies wholly in ONE grid row, so that
    get_pos(cid).x gives each of its lanes a DIFFERENT node -- not that it lies in a
    particular row.  hbm_south() and hbm_north() build global addresses, so reading the
    south edge from the top row is legal; it costs hops, which the model does not charge.

    The arch header is the authority on the grid and these hold the emitted addresses to
    it.  Its hbm_chan_placement is unreadable from C, so this script carries
    HBM_CHAN_PLACEMENT and the two are kept in step by hand.
    """
    L: List[str] = []
    bases, _total = _stage_bases(stages)
    touches = sorted(set(_TABLES['hin']) | set(_TABLES['hout']))
    for i in touches:
        st = stages[i]
        base = bases[i]
        # A lane's whole claim on its node: every leg it issues in either direction, plus
        # the tiled drain when the stage writes its own output.
        per_node: Dict[str, int] = {}
        nodes_per_lane = lanes_per_node = 1
        for key in ('hin', 'hout'):
            for t in _tensors(key, i):
                edge = _leg_edge(t, key)
                nd, ln = _leg_fanout(st.n_par, edge, t.name)
                nodes_per_lane, lanes_per_node = max(nodes_per_lane, nd), max(
                    lanes_per_node, ln)
                per_node[edge] = (per_node.get(edge, 0)
                                  + t.elems * BYTES_PER_ELEM
                                  // HBM_CHAN_PLACEMENT[edge])
        legs = ', '.join(f'{e} {n} B' for e, n in sorted(per_node.items()) if n)
        share = (f'{lanes_per_node} lanes share each node'
                 if lanes_per_node > 1 else
                 f'each lane owns {nodes_per_lane} node(s)')
        L += [
            '',
            f'/* Stage {i + 1} touches LPDDR ({legs} per node; {share}).',
            ' * The node index is derived from the lane, not read off the cluster\'s',
            ' * position, so the stage must START AT A COLUMN BOUNDARY for the two to',
            ' * agree -- and it must not claim more nodes than the edge has.  It does NOT',
            ' * have to lie in one grid row, and does not have to COVER the edge: only a',
            ' * stage that owns its nodes one-per-lane gets that guarantee, and the check',
            ' * below is emitted only for those. */',
            f'#define S{i + 1}_BASE_CID  {base}U',
            f'#define S{i + 1}_LANES     {st.n_par}U',
            f'#define S{i + 1}_NODES     {nodes_per_lane}U  /* nodes per lane */',
            f'#define S{i + 1}_SHARERS   {lanes_per_node}U  /* lanes per node */',
            # `lane` counts from the stage's base cid, so the derived node index only
            # lines up with the physical columns when the stage starts on one.
            f'_Static_assert(S{i + 1}_BASE_CID % ARCH_NUM_CLUSTER_X == 0U,',
            f'               "stage {i + 1} touches LPDDR but does not start at a column '
            f'boundary, so its derived node index is not its column");',
            f'_Static_assert(S{i + 1}_LANES * S{i + 1}_NODES / S{i + 1}_SHARERS'
            ' <= ARCH_NUM_CLUSTER_X,',
            f'               "stage {i + 1} addresses more LPDDR nodes than the edge '
            f'has");',
        ]
        if lanes_per_node == 1:
            # Only a stage with a node per lane can promise that each of its clusters
            # reaches the node in its OWN column, and only then if it fits in one row.
            L += [
                f'_Static_assert(S{i + 1}_BASE_CID / ARCH_NUM_CLUSTER_X',
                f'                   == (S{i + 1}_BASE_CID + S{i + 1}_LANES - 1U) '
                f'/ ARCH_NUM_CLUSTER_X,',
                f'               "stage {i + 1} owns its LPDDR nodes one per lane but '
                f'does not lie in one grid row");',
            ]
        for edge, nbytes in sorted(per_node.items()):
            if not nbytes:
                continue         # this edge carries nothing for this stage
            L += [
                f'_Static_assert({nbytes}U <= ARCH_HBM_NODE_ADDR_SPACE,',
                f'               "stage {i + 1}\'s per-node {edge} traffic does not fit '
                f'one LPDDR node");',
            ]
    last = stages[-1]
    if last.tiles > 1:
        L += [
            '',
            f'/* The last stage drains itself, one tile at a time, into its own node on',
            f' * the {LPDDR_OUT_EDGE.upper()} edge -- see _render_tile_drain.  Lying in the',
            f' * {LPDDR_OUT_EDGE} ROW is not required for correctness (the address is',
            ' * global) but it is what makes the drain a single hop, so it is asserted as',
            ' * the locality claim the timing model rests on. */',
            f'_Static_assert(S{len(stages)}_BASE_CID / ARCH_NUM_CLUSTER_X'
            ' == ARCH_NUM_CLUSTER_Y - 1U,',
            f'               "the self-draining last stage is not in the '
            f'{LPDDR_OUT_EDGE} row, so its drain is not one hop");',
            f'_Static_assert(LPDDR_OUTPUT_OFFSET + {last.tiles}U * TILE_OUT_BYTES',
            '                   <= ARCH_HBM_NODE_ADDR_SPACE,',
            '               "the last stage\'s per-node output does not fit one LPDDR node");',
        ]
    return L


def _render_prologue(pool: str, stages: List[Stage], cycles_per_ms: float,
                     n_iter: int, bench: bool, wait: bool, mode: str,
                     dbl: bool) -> List[str]:
    lpddr, defines = _lpddr_layout(stages, pool)
    kind = 'mempool_wait timing twin' if wait else (
        'RedMulE tensor engines' if pool in TENSOR_ENGINE_POOLS else 'PE matmul')
    L = [
        f'/* Auto-generated by celere_pipeline.py -- '
        f'{"compute-then-transfer" if mode == "ctt" else "compute-while-transfer"} '
        f'pipeline for {pool}.',
        f' * {len(stages)} stages over {sum(s.n_par for s in stages)} clusters'
        + (f' (+{_stage_bases(stages)[1] - sum(s.n_par for s in stages)} idle)'
           if _stage_bases(stages)[1] != sum(s.n_par for s in stages) else '')
        + f', {n_iter} item(s) streamed.',
        f' * Compute: {kind}.',
        ' *',
        ' * Dataflow: ONE activation buffer, carved into named regions per stage.  X_BUF',
        ' * opens with the stage\'s INBOUND tensors, at the offsets the hand-off and the',
        ' * LPDDR read write them to; then the RESIDENT regions the stage pins -- the',
        ' * encoder\'s Q|K|V block, its lattice Z, its attention output U -- which stay live',
        ' * for the whole stage; then the SCRATCH ARENA, where a row with no pinned operand',
        ' * reads at the base and writes one padded input past it.  Where an inbound tensor',
        ' * is dead before the arena is first used, the arena sits ON TOP OF it.',
        ' *',
        ' * Stage 3 repeats stage 2\'s rows: the encoder runs twice with a second weight',
        ' * set, unrolled onto its own clusters rather than looped.  A WHOLE pass is one',
        ' * stage, so nothing crosses between clusters inside one of these functions.',
        ' *',
        ' * ROPE IS THE ONLY TIMING STAND-IN: no kernel for it exists in the tree, so it',
        ' * is a mempool_wait() in every mode.  Every other row runs -- pool on the cores,',
        ' * and each layout row as a permute_f16p, as the address pattern of a hand-off',
        ' * descriptor, or as a view that was already in the order its consumer wanted. */',
        '',
    ]
    L += _render_includes(wait)
    L.append('')
    L += _render_stack_note(stages)
    L.append('')
    if wait:
        L.append(f'#define CYCLES_PER_MS  {int(cycles_per_ms)}UL')
    L.append(f'#define N_ITER  {n_iter}U')
    for k, v in defines.items():
        L.append(f'#define {k}  {v}U')
    L += _render_lpddr_asserts(stages)
    L.append('')
    L += _render_stage_tables(stages, lpddr, pool)
    L.append('')
    L.append(_render_dma_helper(bench))
    L.append(_render_push_helper())
    # The async twins only have a caller when consecutive items overlap; emitting them
    # at N_ITER == 1 would be dead code the compiler warns about.
    if dbl and n_iter > 1:
        L.append(_render_dma_issue_helper())
        L.append(_render_push_helper(issue=True))
    L.append('')
    L += _render_lpddr_helper()
    L.append('')
    L += _render_scratch_buffers(stages, pool, dbl)
    L.append('')
    if not wait:
        # The bodies below are what would carry a level table off the end of X_BUF, so the
        # walk is checked here, before any of them is written.
        _check_permute_extents(stages, pool)
    L += _render_compute_stage_fns(stages, pool, wait, cycles_per_ms, bench, dbl)
    return L


def _render_lpddr_helper() -> List[str]:
    return [
        '/* The LPDDR legs a stage owns: its inbound reads and its outbound writes, as',
        ' * listed in LPDDR_IN_XFER / LPDDR_OUT_XFER above.  A stage with no legs has an',
        ' * empty range and issues nothing.',
        ' *',
        ' * The node index is DERIVED, not the cluster\'s column: a lane may own several',
        ' * nodes or share one, so it is (lane / lanes_per_node) * nodes + n.  hbm_south()',
        ' * and hbm_north() build GLOBAL addresses, so reaching a node outside this',
        ' * cluster\'s column, or on an edge several rows away, is legal; it costs hops,',
        ' * which the timing model does not charge, and a shared channel, which it does.',
        ' *',
        ' * `lane` counts from the stage\'s base cid, so this only lines up with the',
        ' * physical columns when the stage starts on a column boundary -- which the',
        ' * prologue _Static_asserts.',
        ' *',
        ' * hbm_addr() is deliberately NOT used: it is west node 0, and the west edge holds',
        ' * no nodes (see HBM_CHAN_PLACEMENT). */',
        # A LEAF, and inline on purpose: it folds an edge, a node and an offset into one
        # address and spills nothing, so a frame of its own would only deepen the transfer
        # chain.  See the NO_INLINE note for where the line is drawn.
        f'{_LEAF_T("uint64_t")} lpddr_node_addr(uint32_t edge, uint32_t node,',
        f'{" " * (len(_LEAF_T("uint64_t")) + len(" lpddr_node_addr("))}uint32_t off)',
        '{',
        '    return edge == LPDDR_EDGE_NORTH ? hbm_north(node, off)',
        '                                    : hbm_south(node, off);',
        '}',
        '',
        f'{_INLINE} lpddr_pull(uint32_t stage, uint32_t lane, uint64_t dst_base)',
        '{',
        '    for (uint32_t e = LPDDR_IN_FIRST[stage]; e < LPDDR_IN_FIRST[stage + 1u]; e++) {',
        '        const lpddr_xfer_t *x = &LPDDR_IN_XFER[e];',
        '        uint32_t node = (lane / x->lanes_per_node) * x->nodes;',
        '        uint32_t off  = x->lpddr_off + (lane % x->lanes_per_node) * x->lane_step;',
        '        for (uint32_t n = 0; n < x->nodes; n++)',
        '            dma_2d_strided(dst_base + (uint64_t)(x->l1_off + n * x->chunk),',
        '                           lpddr_node_addr(x->edge, node + n, off),',
        '                           x->chunk, x->l1_stride, x->lpddr_stride, x->reps);',
        '    }',
        '}',
        '',
        f'{_INLINE} lpddr_push(uint32_t stage, uint32_t lane, uint64_t src_base)',
        '{',
        '    for (uint32_t e = LPDDR_OUT_FIRST[stage]; e < LPDDR_OUT_FIRST[stage + 1u]; e++) {',
        '        const lpddr_xfer_t *x = &LPDDR_OUT_XFER[e];',
        '        uint32_t node = (lane / x->lanes_per_node) * x->nodes;',
        '        uint32_t off  = x->lpddr_off + (lane % x->lanes_per_node) * x->lane_step;',
        '        for (uint32_t n = 0; n < x->nodes; n++)',
        '            dma_2d_strided(lpddr_node_addr(x->edge, node + n, off),',
        '                           src_base + (uint64_t)(x->l1_off + n * x->chunk),',
        '                           x->chunk, x->lpddr_stride, x->l1_stride, x->reps);',
        '    }',
        '}',
        '',
    ]


def _render_ctt_c(pool: str, stages: List[Stage], cycles_per_ms: float,
                  n_iter: int, bench: bool, wait: bool) -> str:
    L = _render_prologue(pool, stages, cycles_per_ms, n_iter, bench, wait,
                         'ctt', dbl=False)
    L += [
        'int main()',
        '{',
        '    uint32_t cid     = flex_get_cluster_id();',
        '    uint32_t eoc_val = 0;',
        '    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;',
        '    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;',
        '    /* Stage numbers run 1..N_STAGES; the wavefront wants the 0-based position.',
        '     *',
        '     * STAGE 0 IS AN IDLE CLUSTER -- the pipeline leaves cids unused so that the',
        '     * stages touching LPDDR start on a column boundary.  There stage - 1u',
        '     * UNDERFLOWS to ~0u, so `phase <= step` is false at every step and the cluster',
        '     * never computes or transfers.  It still reaches both global barriers, which',
        '     * it MUST: they are collective over every cluster on the machine.  The row-0',
        '     * entry in each stage-indexed table is the second net -- STAGE_XFER_FIRST[0]',
        '     * == STAGE_XFER_FIRST[1] and STAGE_TILES[0] == 0, so even a spuriously active',
        '     * idle cluster would move nothing. */',
        '    uint32_t phase   = stage - 1u;',
        '    const uint32_t buf_off = (uint32_t)(uintptr_t)X_BUF;',
        '',
        '    flex_barrier_xy_init();',
        '    flex_global_barrier_xy();',
    ]
    if not wait:
        L += _render_buffer_init(False)
    L += [
        '    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();',
        '    flex_global_barrier_xy();',
        '',
        '    for (uint32_t step = 0; step < N_ITER + N_STAGES - 1; step++) {',
        '        uint32_t active = (cid < N_CLUSTERS) && (phase <= step)',
        '                          && (step - phase < N_ITER);',
        '        if (active) {',
        '            if (flex_is_dm_core())',
        '                lpddr_pull(stage, lane, (uint64_t)(uintptr_t)X_BUF);',
        '            flex_intra_cluster_sync();   /* worker cores wait for the LPDDR load */',
        '            switch (stage) {',
    ]
    for st in stages:
        L.append(f'            case STAGE_{st.idx + 1}: '
                 f'compute_stage_{st.idx + 1}(); break;')
    L += [
        '            default: break;',
        '            }',
        '        }',
        '        flex_global_barrier_xy();   /* compute done before ANY transfer */',
        '',
        '        if (flex_is_dm_core() && active) {',
            '            /* Both helpers carry their own source offsets -- STAGE_XFER has',
            '             * src_off per tensor and every LPDDR leg has l1_off -- so what',
            '             * they are handed is the BUFFER BASE. */',
        '            uint64_t out = (uint64_t)(uintptr_t)X_BUF;',
        '            lpddr_push(stage, lane, out);',
        '            if (stage < N_STAGES && STAGE_TILES[stage] == 1u)',
        '                push_to_next(stage, lane, buf_off, out);',
        '        }',
        '        flex_global_barrier_xy();   /* all transfers done before next compute */',
        '    }',
        '',
        '    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();',
        '    flex_global_barrier_xy();',
        '    flex_eoc(eoc_val);',
        '    return 0;',
        '}',
    ]
    return '\n'.join(L) + '\n'


def _render_cwt_c(pool: str, stages: List[Stage], cycles_per_ms: float,
                  n_iter: int, bench: bool, wait: bool) -> str:
    overlap = n_iter > 1
    L = _render_prologue(pool, stages, cycles_per_ms, n_iter, bench, wait,
                         'cwt', dbl=True)
    L += [
        'int main()',
        '{',
        '    uint32_t cid     = flex_get_cluster_id();',
        '    uint32_t eoc_val = 0;',
        '    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;',
        '    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;',
        '    /* Stage numbers run 1..N_STAGES; the wavefront wants the 0-based position.',
        '     *',
        '     * STAGE 0 IS AN IDLE CLUSTER -- the pipeline leaves cids unused so that the',
        '     * stages touching LPDDR start on a column boundary.  There stage - 1u',
        '     * UNDERFLOWS to ~0u, so `phase <= step` is false at every step and the cluster',
        '     * never computes or transfers.  It still reaches both global barriers, which',
        '     * it MUST: they are collective over every cluster on the machine.  The row-0',
        '     * entry in each stage-indexed table is the second net -- STAGE_XFER_FIRST[0]',
        '     * == STAGE_XFER_FIRST[1] and STAGE_TILES[0] == 0, so even a spuriously active',
        '     * idle cluster would move nothing. */',
        '    uint32_t phase   = stage - 1u;',
        '    const uint32_t buf_off = (uint32_t)(uintptr_t)X_BUF;',
        '',
        '    flex_barrier_xy_init();',
        '    flex_global_barrier_xy();',
    ]
    if not wait:
        L += _render_buffer_init(True)
    L += [
        '    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();',
        '    flex_global_barrier_xy();',
        '',
        '    /* Consecutive stages are offset by TWO steps so producer and consumer agree',
        '     * on the item parity; the outbound transfer of one item then runs against the',
        '     * compute of the next. */',
        '    uint32_t base = 2u * phase;',
        '    for (uint32_t step = 0; step < 2u * (N_STAGES - 1u) + N_ITER + 1u; step++) {',
        '        uint32_t cbuf = step & 1u;',
        '        uint32_t obuf = cbuf ^ 1u;',
        '        uint32_t compute_active  = (cid < N_CLUSTERS) && (step >= base)',
        '                                   && (step - base < 2u * N_ITER)',
        '                                   && (((step - base) & 1u) == 0u);',
        '        uint32_t outbound_active = (cid < N_CLUSTERS) && (step > base)',
        '                                   && (step - base <= 2u * N_ITER)',
        '                                   && (((step - base) & 1u) == 1u);',
        '',
        '        if (compute_active && flex_is_dm_core())',
        '            lpddr_pull(stage, lane, (uint64_t)(uintptr_t)X_BUF[cbuf]);',
        '        flex_intra_cluster_sync();',
        '',
    ]
    if overlap:
        L += [
            '        if (outbound_active && flex_is_dm_core()) {',
            '            uint64_t out = (uint64_t)(uintptr_t)X_BUF[obuf];',
            '            lpddr_push(stage, lane, out);',
            '            if (stage < N_STAGES && STAGE_TILES[stage] == 1u)',
            '                push_to_next_issue(stage, lane, buf_off, out);',
            '        }',
            '',
        ]
    L.append('        if (compute_active) {')
    L.append('            switch (stage) {')
    for st in stages:
        L.append(f'            case STAGE_{st.idx + 1}: '
                 f'compute_stage_{st.idx + 1}(cbuf); break;')
    L += [
        '            default: break;',
        '            }',
        '        }',
    ]
    if overlap:
        L += [
            '        if (flex_is_dm_core()) flex_dma_async_wait_all();',
        ]
    else:
        L += [
            '        flex_global_barrier_xy();',
            '        if (outbound_active && flex_is_dm_core()) {',
            '            uint64_t out = (uint64_t)(uintptr_t)X_BUF[obuf];',
            '            lpddr_push(stage, lane, out);',
            '            if (stage < N_STAGES && STAGE_TILES[stage] == 1u)',
            '                push_to_next(stage, lane, buf_off, out);',
            '        }',
        ]
    L += [
        '        flex_global_barrier_xy();',
        '    }',
        '',
        '    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();',
        '    flex_global_barrier_xy();',
        '    flex_eoc(eoc_val);',
        '    return 0;',
        '}',
    ]
    return '\n'.join(L) + '\n'


_C_RENDERERS = {'ctt': _render_ctt_c, 'cwt': _render_cwt_c}


# ---------------------------------------------------------------------------
# App generation
# ---------------------------------------------------------------------------

_APPS_DIR = (Path(__file__).resolve().parent.parent.parent.parent
             / 'software' / 'apps' / 'multi_cluster')


def app_name(mode: str, pool: str) -> str:
    return f'celere_{mode}_{pool.replace(" ", "_")}'


def generate_apps(res: Results, apps_dir: str, cycles_per_ms: float, mode: str,
                  n_iter: int, bench: bool, wait: bool, l1_bytes: int) -> None:
    print(f'\nGenerating {mode} apps:')
    for pool, stages in res.stages.items():
        app_dir = Path(apps_dir) / app_name(mode, pool)
        app_dir.mkdir(parents=True, exist_ok=True)
        src = _C_RENDERERS[mode](pool, stages, cycles_per_ms,
                                 n_iter, bench, wait)
        (app_dir / 'main.c').write_text(src)
        print(f'  [{pool}]  {app_dir / "main.c"}')
        l1, parts = _l1_footprint(stages, mode, pool)
        geo = L1_GEOMETRY.get(pool)
        budget = (min(l1_bytes, geo['total']) - geo['stacks'] - geo['runtime']
                  if geo else l1_bytes)
        print(f'  [{pool}]  .l1 {l1 / 1024:.0f} KiB of {budget / 1024:.0f} KiB  '
              f'({parts})')
        worst = max(stages, key=lambda s: s.kernel_elems)
        kern = worst.kernel_elems * BYTES_PER_ELEM * (2 if mode == 'cwt' else 1)
        print(f'  [{pool}]  largest stage by its own layers: {kern / 1024:.0f} KiB '
              f'(stage {worst.idx + 1})')
        if l1 > budget:
            print(f'  [{pool}]  WARNING: {(l1 - budget) / 1024:.0f} KiB over budget -- '
                  f'this app will not link.')
        print(f'  [{pool}]  build with:  make -C {apps_dir} '
              f'{app_name(mode, pool)} COMPILER=llvm config=tensorpool256')


# ---------------------------------------------------------------------------
# Summary + plots
# ---------------------------------------------------------------------------

def write_summary(res: Results, path: str, csv_name: str) -> None:
    lines = ['# CELERE pipeline summary  (read by latency_throughput_bars.py)',
             f'# csv: {csv_name}',
             '# columns: pool  mode  throughput_ms  n_stages  n_clusters  latency_ms']
    for mode, best, steps in (('ctt', res.best_throughput, 1),
                              ('cwt', res.best_throughput_overlap, 2)):
        for pool, stages in res.stages.items():
            n_st = len(stages)
            n_cl = sum(s.n_par for s in stages)
            lines.append(f'{pool}\t{mode}\t{best[pool]:.6g}\t{n_st}\t{n_cl}'
                         f'\t{steps * n_st * best[pool]:.6g}')
    Path(path).write_text('\n'.join(lines) + '\n')
    print(f'\nSummary written to {path}')


def _fit_yaxis(fig, ax, top: float, label_pt: float = 10.0) -> None:
    """Scale y to the data, leaving exactly the room the bar labels need.

    The headroom is the annotation's height, a fixed number of POINTS whatever the data is
    worth in ms; converting it through the axes' pixel height gives the fraction of the
    axes it occupies, so bars fill (1 - f) of the height at any magnitude.  A fixed
    percentage cannot do that.
    """
    if top <= 0.0:
        ax.set_ylim(0.0, 1.0)
        return
    fig.canvas.draw()                       # realise the layout before measuring it
    ax_px = ax.get_window_extent().height
    need_px = label_pt * fig.dpi / 72.0
    f = min(0.25, need_px / ax_px) if ax_px > 0 else 0.08
    ax.set_ylim(0.0, top / (1.0 - f))


def _plot_split(plt, pool: str, stages: List[Stage], overlap: bool,
                out_path: Path) -> None:
    """Per-cluster wall time: compute (red) against transfer (green).

    One x slot per STAGE, one BAR GROUP per CLUSTER inside it, which shows directly that a
    stage's wall is the height of one lane and not the sum of them.

    ctt -- stacked bars: compute and transfer are barriered substages, so the wall is the
           sum.  cwt -- grouped pair: they overlap, so the wall is the taller, standing on
           the hatched pedestal of the inbound LPDDR read (_overlap_wall).

    A self-draining stage carries its transfer INSIDE its compute term, so its green bar
    is legitimately zero; the tick label says so.
    """
    tag = 'CWT' if overlap else 'CTT'
    n_st = len(stages)
    n_phys = sum(s.n_par for s in stages)
    bases, at = [], 0
    for st in stages:
        bases.append(at)
        at += st.n_par
    walls = [(_overlap_wall(s) if overlap else s.latency + s.transfer_ms)
             for s in stages]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * n_phys + 3.0), 4.6))
    span = 0.82                      # width of a stage slot
    first = True                     # label only the first bars, for one legend entry
    first_din = True
    first_drain = True
    tick_pos, tick_lab = [], []

    for i, (st, base) in enumerate(zip(stages, bases)):
        p = st.n_par
        slots = 2 * p if overlap else p
        bw = span / slots
        # keep a lone cluster's bar from ballooning to the full slot width
        draw = min(bw * 0.9, 0.30 if overlap else 0.34)
        left = i - span / 2

        if p > 1:
            ax.axvspan(left - 0.03, left + span + 0.03, color='0.92', zorder=0)
        for lane in range(p):
            if overlap:
                xmid = left + (2 * lane + 1.0) * bw
                dout = st.transfer_ms - st.din_ms
                if st.din_ms > 0.0:
                    for x in (xmid - bw / 2, xmid + bw / 2):
                        ax.bar(x, st.din_ms, width=draw, color='green', hatch='//',
                               edgecolor='white', zorder=2,
                               label=('data_in (not overlapped)'
                                      if first_din and x < xmid else None))
                    first_din = False
                ax.bar(xmid - bw / 2, st.latency - st.drain_ms, width=draw,
                       bottom=st.din_ms, color='red', zorder=2,
                       label='compute' if first else None)
                if st.drain_ms > 0.0:
                    ax.bar(xmid - bw / 2, st.drain_ms, width=draw,
                           bottom=st.din_ms + st.latency - st.drain_ms, color='green',
                           hatch='\\\\', edgecolor='white', zorder=2,
                           label=('transfer (drained inside compute)'
                                  if first_drain else None))
                    first_drain = False
                ax.bar(xmid + bw / 2, dout, width=draw, bottom=st.din_ms,
                       color='green', zorder=2, label='transfer' if first else None)
            else:
                xmid = left + (lane + 0.5) * bw
                ax.bar(xmid, st.latency - st.drain_ms, width=draw, color='red',
                       zorder=2, label='compute' if first else None)
                if st.drain_ms > 0.0:
                    ax.bar(xmid, st.drain_ms, width=draw,
                           bottom=st.latency - st.drain_ms, color='green', hatch='\\\\',
                           edgecolor='white', zorder=2,
                           label=('transfer (drained inside compute)'
                                  if first_drain else None))
                    first_drain = False
                ax.bar(xmid, st.transfer_ms, width=draw, bottom=st.latency,
                       color='green', zorder=2, label='transfer' if first else None)
            first = False
            # cluster id above its own bar(s), clear of the stage ticks below the axis
            ax.annotate(f'c{base + lane}', xy=(xmid, walls[i]), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, color='0.35')

        tick_pos.append(i)
        extra = []
        if p > 1:
            extra.append(f'{p} clusters ||')
        if st.tiles > 1:
            extra.append(f'{st.tiles} tiles, self-drains')
        tick_lab.append(f'stage {st.idx + 1}'
                        + (f'\n({", ".join(extra)})' if extra else ''))

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_xlim(-0.6, n_st - 0.4)
    ax.set_xlabel('pipeline stage  (clusters of a stage run in parallel)')
    ax.set_ylabel('wall time [ms]')
    ax.set_title(f'CELERE {pool} -- {tag}  |  {n_st} stages over {n_phys} clusters')
    # Legend outside, to the right: inside the data area the y range would have to dodge
    # it.  bbox_inches='tight' grows the canvas instead of stealing plot area.
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small',
              frameon=False)
    fig.tight_layout()
    _fit_yaxis(fig, ax, max(walls))
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def generate_plots(res: Results, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available -- skipping plots')
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for pool, stages in res.stages.items():
        for overlap in (False, True):
            tag = 'CWT' if overlap else 'CTT'
            path = out_dir / f'CELERE_{tag}_split_{pool.replace(" ", "_")}.png'
            _plot_split(plt, pool, stages, overlap, path)
            print(f'  {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('template', help='Path to the CELERE template (.csv/.xlsx/.ods)')
    ap.add_argument('config', help='Path to the JSON hardware config')
    ap.add_argument('--mode', choices=['ctt', 'cwt'], metavar='{ctt,cwt}', default=None,
                    help='Generate only this flavour; omit for both')
    ap.add_argument('--iterations', type=int, default=1, metavar='N',
                    help='Items streamed through the pipeline (default 1)')
    ap.add_argument('--out', default=str(_APPS_DIR),
                    help='Apps root under which celere_<mode>_<pool>/main.c is written')
    ap.add_argument('--summary', default='pipeline_summary.txt',
                    help='Machine-readable per-(pool, mode) summary')
    ap.add_argument('--bench', action='store_true',
                    help='Bracket every layer with mempool_start/stop_benchmark()')
    ap.add_argument('--wait', action='store_true',
                    help='Emit the timing-only pipeline: every layer a mempool_wait() '
                         'spin sized to the modelled latency, instead of a real kernel')
    ap.add_argument('--l1-bytes', type=int, default=4 * 1024 * 1024, metavar='N',
                    help='TCDM budget for the over-budget warning (default 4 MiB)')
    ap.add_argument('--plot', action='store_true',
                    help='Save per-stage bar plots under ./celere_split_plots/')
    args = ap.parse_args()

    if args.iterations < 1:
        ap.error('--iterations must be >= 1')

    cfg = load_config(args.config)
    layers = load_template(args.template)
    print(f'Parsed {len(layers)} kernel rows from {Path(args.template).name}; '
          f'dropped row(s) {list(DROPPED_ROWS)} (they ride a DMA).')

    res = compute(layers, cfg)
    print_results(layers, res, cfg, args.l1_bytes)
    write_summary(res, args.summary, Path(args.template).name)

    cycles_per_ms = cfg.get('frequency', 1_000_000_000) / 1e3
    for m in ([args.mode] if args.mode else ['ctt', 'cwt']):
        generate_apps(res, args.out, cycles_per_ms, m, args.iterations, args.bench,
                      args.wait, args.l1_bytes)

    if args.plot:
        print('\nPlots:')
        generate_plots(res, Path.cwd() / 'celere_split_plots')


if __name__ == '__main__':
    main()
