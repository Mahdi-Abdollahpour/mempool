#!/usr/bin/env python3
"""
CEViT performance model and GVSOC pipeline app generator.

Reads the model's layer template (xlsx/ods/csv), costs it against a JSON hardware config, and
emits one multi-cluster app per (mode, pool) under
software/apps/multi_cluster/cevit_<mode>_<pool>/main.c.

    python cevit_pipeline.py CEViT_273PRBs.csv config.json
    python cevit_pipeline.py CEViT_273PRBs.csv config.json --iterations 8 --bench
    python cevit_pipeline.py CEViT_273PRBs.csv config.json --wait      # timing-only twin
    python cevit_pipeline.py CEViT_273PRBs.csv config.json --plot

    ctt  compute-then-transfer: stage s is SKEWED BY s STEPS, so it works on item m at step
         m + s.  TWO barriers a step, one L1 buffer set, compute and DMA serialized.  Period =
         slowest compute + slowest transfer over the whole partition, because the barrier
         stops the two covering for each other.
    cwt  compute-while-transfer: stage s is SKEWED BY 2s STEPS, so it works on item m at step
         m + 2s.  That leaves a spare step between a producer and its consumer for the
         transfer to run in, and needs ONE barrier a step.  A step reads item i's OWN input,
         waits for it, then issues item i-1's drain beside the compute, so the wall is
         data_in + max(compute, transfer): flex_dma_async_wait_all() is the only wait the DMA
         shim has, so the read cannot be hidden as well.

SCOPE.  THIS PARTITION IS TUNED FOR TENSORPOOL.  The stage widths were chosen against its
RedMulE GEMM peak, where the period is 0.086 ms against Terapool's 0.526; on Terapool a GEMM is
a PE matmul and the balance moves -- the attention core becomes the wall instead of the FFN
block -- so a Terapool-first partition would want different widths, not just a different
config.  The Terapool numbers are honest output of the same model, they are simply not what the
shape was optimised for.  All four apps do fit L1, between 2206 and 2581 KiB.

Both are generated, for both pools -- four apps.  Under cwt only what crosses an item boundary
takes a second copy (a hand-off's landing zone and whatever an outbound transfer reads), and
only for the stage a cluster is on.

THE PARTITION IS NOT SEARCHED FOR.  It lives in STAGES below: THREE stages of TWO, SIX and TWO
clusters, ten working clusters on a 4 x 3 grid of twelve.

    row 2 (NORTH edge)   cid  8 [s3]  cid  9 [s3]  cid 10 [--]  cid 11 [--]
    row 1                cid  4 [s2]  cid  5 [s2]  cid  6 [s2]  cid  7 [s2]
    row 0 (SOUTH edge)   cid  0 [s1]  cid  1 [s1]  cid  2 [s2]  cid  3 [s2]

    stage 1   cids 0-1   2 lanes -- the front end and the Q|K|V projection, 546 tokens each
    stage 2   cids 2-7   6 lanes -- the attention core, 2 HEADS x 364 QUERIES each
    stage 3   cids 8-9   2 lanes -- the FFN block and the output projection, 546 tokens each

No stage needs `pad`: the two that touch LPDDR land on the rows with nodes beside them by
themselves, and stage 2 touches none.  DISTANCE IS NOT CHARGED -- every leg is billed its bytes
and nothing more.

WHERE THE PARALLELISM COMES FROM.  Attention is independent across HEADS and across QUERIES
(softmax normalises over i_4, the keys) but NOT over the token axis of K and V: a query scores
against every key and A*V sums over every one, so a lane needs K and V for all 1092 tokens of
whatever heads it owns.  Everything outside the attention core is row-wise over tokens, so any
number of lanes divides the two end stages with nothing replicated.

WHAT THE QUERY SPLIT COSTS IS K AND V ON THE WIRE.  The three lanes of one head group each need
that group's whole K and whole V, so both cross boundary 1 -> 2 three times; Q crosses once.
That is charged, not hidden (_handoff_qkv builds the copies, _handoff_replication tells the
sizing check to expect them), and it makes that boundary the largest transfer in the model --
955 KiB a lane, about three times the residual round trip.  The period sits on stage 3, at
0.069 ms of compute on TensorPool against stage 2's 0.050 and stage 1's 0.029; it divides
freely, so a taller grid is the fix.  On Terapool the balance inverts and stage 2 is the wall.

THE TWO BOUNDARIES

  * 1 -> 2, THE PROJECTION, cut by head AND by token at once.  Row 10's output is
    [tokens, 3*d_model] row-major, so head h of stripe s is column s*d_model + h*d_head of
    every row: a d_head RUN at a 3*d_model STRIDE.  That addressing IS rows 11 and 12, and it
    is free -- d_head is 32 halfwords, exactly one 64 B DMA line.  BOTH ENDS FAN: a consumer's
    K slab is assembled from both producers, and Q is the intersection of a 546-token producer
    slab with a 364-query consumer slab.  Thirty-two descriptors per producer.
  * 2 -> 3, A*V, six lanes to two.  ITS DESTINATION STRIDE IS THE HEAD MERGE: one d_head run
    per token at column h*d_head of a d_model-wide row is exactly rows 17 and 18.  Sixteen
    descriptors, because 364 does not divide 546.  Each source overlaps only its own
    destinations, so the tensor moves EXACTLY ONCE.

The score matrix, the largest tensor in the model, never crosses anything: it is produced,
softmaxed and consumed inside one lane of one stage.

LAYOUT ROWS.  The template has seven and none is a kernel -- nothing in
software/kernels/baremetal permutes a tensor.  A transfer can perform one only when it was
crossing that boundary anyway, and here every one has a transfer to ride (DROPPED_ROWS), so
nothing is left to spin for on the PE path:

    row 2             patchify                       the inbound LPDDR read
    rows 11, 12       split q,k,v + split heads      the 1 -> 2 hand-off's source addressing
    row 13            transpose K                    ASSUMED onto the 1 -> 2 hand-off
    rows 17, 18       transpose + merge heads        the 2 -> 3 hand-off's destination stride
    row 33            merge patches                  the outbound LPDDR write

ROW 13 IS THE ONE OPTIMISM IN THIS FILE.  Row 14 contracts against K TRANSPOSED and the
partition assumes the hand-off produces K^T on the way, which a DMA cannot do: K^T is
[d_head, tokens], so a descriptor building it walks 2-byte runs against a 64 B line.  What is
emitted is the plain strided slab, delivering K.  The CYCLES are unaffected -- row 14's GEMM is
issued at the same M, N and P, and the app checks no result -- so what is lost is a numerical
answer nobody reads.  Putting row 13 back in STAGES buys a real transpose, as a PE wait on
stage 1 or 2.

CAVEAT ON PATCHIFY.  It is charged as a plain contiguous read and it is not one: [Nf, Nt, C]
interleaves the two channels and the channel is part of the output row index, so one patch row
walks 84 elements with two different strides against a 64 B line -- a three-level nest no
single flex_dma_async_2d expresses.  Treat that leg as optimistic until it is measured.

THE ATTENTION RESIDUAL SPILLS, AND NOTHING ELSE DOES.  A residual has to leave the chip exactly
when it SKIPS a stage: row 9 is on stage 1 and row 19 on stage 3, and a boundary only ever
joins adjacent stages.  It goes to LPDDR on a rotating set of slots (see LPDDR_LEGS), cut into
TWO PIECES indexed by PIECE rather than by column, because stage 3 sits on the north row and
its nearest nodes are the wrong ones.  At ~0.0044 ms a lane on TensorPool it is a third of
boundary 1 -> 2, so raising lpddr_channels buys little.  The FFN residual never spills: row 19
-> row 28 stays on one lane.

L1 BUDGET.  Stage 2 sets it and its score matrix sets stage 2: [364 queries x 1092 keys],
written once by Q*K^T and once by the softmax, is 1.6 MiB of the 1.9 MiB the stage declares.
About 2.2 MiB against TensorPool's 3948 KiB and Terapool's 3544, with no tiling.  Two heads per
lane is a BATCH loop and the emitter will not nest a tiling inside one, so `tiles` must stay 1
here; a larger PRB count needs a wider stage 2, a tiling, or both.

Every layer is lowered to a REAL baremetal kernel call operating on placeholder L1 scratch --
these kernels' cycle counts are data-independent, so the apps need no HBM preload.  Pass --wait
for the timing-only twin, where every layer is a spin sized to the modelled latency.

The template is parsed without markers: the header row (labelling N_repetitions / FLOPs /
Weights / Data_in / Data_out) is located automatically and layers are read to the trailing
parameter block; block name -> column A, operation -> column C, the rest by header label, all
traffic in elements.  STAGES names template ROW NUMBERS and build_stages checks the table
places every row N_repetitions times and every dropped row zero times, so a template whose rows
move stops the generator instead of costing the wrong network.  The token count, head count and
d_head are read from the rows, so a different PRB count is a different CSV and nothing else.

Hardware peaks, utilisations and the link bandwidths come from a JSON config; see config.json.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Logical field -> accepted header names in the header row
HEADER_ALIASES: Dict[str, List[str]] = {
    'n_rep':   ['N_repetitions', 'N_rep', 'Repetitions'],
    'kernel':  ['Kernel'],
    'flops':   ['FLOPs', 'Flops', 'FLOPS'],
    'weights': ['Weights'],
    'din':     ['Data_in'],
    'dout':    ['Data_out'],
}

# Fallback positions (0-based) for columns that don't carry a header label
DEFAULT_COLS: Dict[str, int] = {'block': 0, 'op': 2}

# Tensor-shape columns, located by their header label.  OPTIONAL: only real-kernel mode
# needs them.  `in_*` = input operand dims, `out_*` = output dims; there are no
# second-operand columns, since a matmul's/dense's B is (in_dims[-1], out_dims[-1]).
DIM_HEADERS: Dict[str, str] = {
    'in_1': 'i_1', 'in_2': 'i_2', 'in_3': 'i_3', 'in_4': 'i_4',
    'out_1': 'o_1', 'out_2': 'o_2', 'out_3': 'o_3', 'out_4': 'o_4',
}

BYTES_PER_ELEM = 2                       # FP16

# The DMA moves whole 64 B lines.  A descriptor whose runs do not START on a line
# boundary straddles two lines on every repetition and the leg goes at HALF RATE -- which
# _transfer_ms cannot see, because it charges bytes and nothing else.  So every placed
# region is aligned to a line, and _check_alignment holds every emitted descriptor to it.
DMA_LINE_B     = 64
DMA_LINE_ELEMS = DMA_LINE_B // BYTES_PER_ELEM


def _line(n: int) -> int:
    """Round an ELEMENT count up to a whole DMA line."""
    return ((n + DMA_LINE_ELEMS - 1) // DMA_LINE_ELEMS) * DMA_LINE_ELEMS

# Kernels dispatched to the GEMM path; every other kernel runs on the PE path.
GEMM_KERNELS = {'dense', 'matmul', 'conv1d'}
LAYOUT_KERNELS = {'reshape', 'split', 'transpose'}

# Pools with a RedMulE tensor engine.  Only here is GEMM_peak derated by TE_utilization,
# and only here does the generated C call redmule_synch_parallel -- so the model and the
# code stay on one fact.
TENSOR_ENGINE_POOLS = {'TensorPool'}

# RedMulE tiles per cluster: redmule_synch_parallel splits M over them with two integer
# divisions, so an M that is not a multiple of the tile count drops the remainder and
# misaligns every engine's slice.  The emitted M is rounded up; the COST model keeps the
# true M, since the padding is an artefact of the kernel's row split.
POOL_REDMULE_TILES = {'TensorPool': 16}

# The grid the pipeline occupies, and the LPDDR nodes on its edges.  Must match
# num_cluster_x / num_cluster_y and hbm_chan_placement in
# software/runtime/multi_cluster/configs/arch_*.py -- the generated app _Static_asserts
# the first two out of flex_cluster_arch.h, but nothing in the C can see the placement,
# so the node counts below are the app's only view of it and are kept in step by hand.
GRID_X, GRID_Y = 4, 3
HBM_CHAN_PLACEMENT = {'west': 0, 'north': GRID_X, 'east': 0, 'south': GRID_X}

# Traffic enters on the south edge and leaves on the north one; a tensor may name its own
# edge, so these are only the defaults.  West and east hold nothing, so hbm_addr() (west
# node 0) must never appear in a generated app: there is no node behind it.
LPDDR_IN_EDGE = 'south'
LPDDR_OUT_EDGE = 'north'


# PE-run layout cost, the same fit celere_pipeline.py uses: a strided halfword permutation
# with no arithmetic to hide the addressing behind.  DORMANT ON THIS PARTITION -- every
# layout row rides a transfer (see DROPPED_ROWS), so nothing reaches this fit.  Kept
# because it is what would cost a layout row stranded mid-stage, and because it is the
# honest alternative DROPPED_ROWS' row 13 is measured against.  It is a timing model, not
# a lowering: a row costed here would spin in the emitted app and move no bytes.  The fit
# was taken on patchify_f16, a halfword-strided gather; re-fit from an RTL run with
# --bench.
PE_LAYOUT_FLOPS_PER_ELEM = 8.9
PE_LAYOUT_FIXED_CYCLES = 1029            # per-invocation entry cost (cluster rendezvous)

# Layout rows no kernel runs and no local copy moves: a TRANSFER CARRIES EACH OF THEM, so
# the row is the address pattern of a descriptor and costs nothing extra.
#
#   row 2         patchify                    the inbound LPDDR read on stage 1
#   rows 11, 12   split q,k,v / split heads   the 1 -> 2 hand-off's source addressing:
#                                             head h's Q slab is a d_head run at a
#                                             3*d_model stride out of row 10's output
#   rows 17, 18   transpose + merge heads     the 2 -> 3 hand-off's destination stride
#   row 33        merge patches               the outbound LPDDR write on stage 3
#
# ROW 13 IS HERE ON AN ASSUMPTION, AND IT IS THE ONE OPTIMISM IN THIS FILE: row 14
# contracts against K TRANSPOSED, and a DMA cannot build K^T -- a descriptor for
# [d_head, tokens] walks 2-byte runs against a 64 B line, which _check_alignment would
# refuse.  What is emitted is the plain contiguous slab.  It costs NOTHING IN CYCLES (same
# M, N and P) and the app checks no result, so only a numerical answer is lost.  Put row 13
# back in STAGES to buy a real transpose, as a wait on stage 1 or 2.
#
# All seven rows carry 0 FLOPs, so nothing leaves the latency totals with them.
DROPPED_ROWS = (2, 11, 12, 13, 17, 18, 33)

# Rows whose "weights" are indexed by TOKEN and so split with the tokens.  Row 6 adds the
# positional-encoding table, one row per token, so a lane holding 546 tokens holds 546
# rows of it and not the whole [1092, 128].  Every other row's weights are a contraction
# matrix and are replicated on every lane of the stage.
SPLIT_WEIGHT_ROWS = {6}


#
# One entry per stage, in order.  `n_par` is how many clusters run it, `pad` how many cids
# it leaves IDLE after itself, `split` the axis the lanes divide (documentation, printed
# in the tables and the emitted comments), and `parts` the stage's ordered work -- see
# Part.
#
# Ten working clusters over the twelve the 4 x 3 grid provides; cids 10-11 are idle and NO
# STAGE NEEDS `pad`, because the widths land the two stages that touch LPDDR on the rows
# with nodes beside them by themselves -- stage 1 on columns 0-1 of the SOUTH edge row
# where the model input arrives and the residual spills, stage 3 on columns 0-1 of the
# NORTH one where the channel estimate leaves.  Stage 2 straddles rows 0 and 1 and does
# not care.
#
# EVERY STAGE DIVIDES REAL WORK; nothing is replicated on a lane, and NOTHING CROSSES
# INSIDE A STAGE -- every cut is over an axis its rows are independent along.  So each
# stage is emitted as ONE compute_stage_N() and a step needs exactly the two global
# barriers the wavefront already has.
STAGES: List[dict] = [
    # THE FRONT END AND THE Q|K|V PROJECTION, on TWO clusters split over the TOKENS: 546
    # each, all four heads on both.  ONE PART AND FOUR KERNELS, one per row: embed, add
    # the positional table, layernorm, project.  n_par = 2 with head_groups = 1 gives
    # q_split = 2, so row 10 is a [546, 128] x [128, 384] dense on each lane.
    #
    # THE PROJECTION IS NOT SLICED BY HEAD HERE: a lane computes all of Q|K|V for all four
    # heads of its own tokens in one call, and ROWS 11 AND 12 ARE PERFORMED BY THE HAND-OFF
    # INSTEAD -- head h's Q slab is a 64 B run at a 768 B stride, one whole DMA line either
    # way (see _handoff_qkv).  Row 6's weights split with the tokens; see
    # SPLIT_WEIGHT_ROWS, which this partition is the first to make use of.
    dict(n_par=2,
         split='i_2 1092/2 tokens per lane; all 4 heads on both',
         parts=[dict(rows=[5, 6, 9, 10])]),
    # THE ATTENTION CORE, on 6 lanes = 2 head groups x 3 query slabs, so a lane owns TWO
    # HEADS and 364 of their 1092 queries.  Lane c owns head group c / 3 and query slab
    # c % 3.
    #
    # TWO HEADS ON A LANE IS A BATCH LOOP -- each head contracts against its own K, so the
    # two cannot be one GEMM and _part_loop emits `for (h = 0; h < 2; h++)` with operand B
    # advancing per pass.  That is also why `tiles` is 1 and must be: the emitter writes
    # ONE loop, not a tiling nested inside a batch, and refuses the combination.  It does
    # not need the tiling either: a lane's score matrix is [364, 1092] written twice,
    # ~1.6 MiB against a 4 MiB TCDM.
    dict(n_par=6,
         split='i_2 4/2 (2 heads) per lane, i_3 1092/3 queries per lane',
         parts=[dict(rows=[14, 15, 16], head_groups=2)]),
    # THE FFN BLOCK AND THE OUTPUT PROJECTION, on TWO clusters holding 546 tokens each.
    # Every row here is row-wise over tokens, so any number of lanes divides it and the
    # width is a free choice; two is what keeps the ~310 MFLOP of GEMM from being the
    # pipeline's wall, which it is at one lane (0.183 ms against stage 2's 0.075).
    dict(n_par=2,
         split='i_2 1092/2 tokens per lane',
         parts=[dict(rows=[19, 22, 23, 24, 25, 26, 27, 28, 30])]),
]

# WHAT THE SHAPE COSTS.  The cuts follow from the model; the widths do not.  Modelled ctt
# periods for the shapes this script has been through:
#
#                                     clusters   ctt TensorPool   ctt Terapool   cluster-ms
#   2 / 6 / 2, this one                  10       0.0863 ms        0.5262 ms        0.863
#   1 / 4 / 2, stage 3 doubled            7       0.1546           0.5944           1.082
#   6 + 2, replicated front end           8       0.0896           0.5564           0.717
#   8 + 2, replicated front end          10       0.0730           0.5079           0.730
#   1 / 8 / 1, front and tail serial     10       0.1272           0.8044           1.272
#
# Removing the two single-cluster stages was the cheap half; the expensive half was stage
# 2, where splitting the QUERIES three ways is the only way to get six lanes out of a
# four-head model without replicating the front end -- and it makes boundary 1 -> 2 the
# largest transfer in the model.  Stage 2's 2 x 3 is a FOUR-HEAD answer at SIX LANES:
# head_groups must divide both the head count and the width.  More heads would buy lanes
# with head groups instead, which costs nothing on the wire; a deeper pre-attention block
# argues for a wider stage 1, which is free.


# WHERE EACH ROW FINDS ITS OPERANDS.  A row addresses the base of its buffer unless one of
# the three tables below sends it to a NAMED REGION; a value may name the buffer as well,
# so 'Y:SCORES' is the SCORES region of Y_BUF and 'X:QKV_K' the QKV_K region of X_BUF.
# That cross-buffer form is what stops a tensor being declared twice: a cluster plays ONE
# role, so the score matrix row 14 writes and the one row 15 reads are the same claim on
# the same silicon.
#
#   STAGE 1 runs out of Y_BUF after its first row; only the patch tensor arrives from
#   outside, so X_BUF holds that and nothing else:
#
#     row  A operand                       B operand              output
#      5   X:PATCH (the LPDDR read)        W: embedding matrix    Y:EMB
#      6   the positional table, in W      --                     Y:EMB, in place
#      9   Y:EMB                           --                     Y:RESIDUAL
#     10   Y:RESIDUAL                      W: W_qkv, whole        Y:QKV  [546, 3*d_model]
#
#   STAGE 2 takes ALL THREE of its operands from the boundary, the one place here where
#   operand B is neither a weight nor an earlier row's output.  Row 14's B is K, not K^T
#   (see DROPPED_ROWS); the GEMM's shape is the same either way.
#
#     14   X:QKV_Q                         X:QKV_K (as K^T)       Y:SCORES
#     15   Y:SCORES                        --                     Y:SCRATCH
#     16   Y:SCRATCH                       X:QKV_V                Y:ATTNH
#
#   STAGE 3 takes the head-merged attention output into Y:FFN_RESID, because row 19's axpy
#   computes Y += X; the residual arrives separately, off LPDDR, into X:RESIDUAL.
BIND_X: Dict[int, str] = {9: 'Y:EMB', 10: 'Y:RESIDUAL',
                          14: 'X:QKV_Q', 15: 'Y:SCORES', 16: 'Y:SCRATCH',
                          19: 'RESIDUAL', 22: 'Y:FFN_RESID', 28: 'Y:FFN_RESID'}

# Rows whose OUTPUT has to land in a named region rather than at the base of Y_BUF, because
# something later in the STEP still has to read it -- a later row, a hand-off, or an LPDDR
# write.  Everything unnamed writes the base and is scratch the next row overwrites.
#
# A LAYOUT ROW MAY ALSO CLAIM ONE (see _fp): a layout row is normally a VIEW, but one that
# really does produce a second tensor needs somewhere to put it.  Nothing on this partition
# is in that position -- every layout row here is dropped, row 13 included.
BIND_Y: Dict[int, str] = {5: 'EMB', 6: 'EMB', 9: 'RESIDUAL', 10: 'QKV',
                          14: 'SCORES', 16: 'ATTNH',
                          19: 'FFN_RESID', 30: 'OUTPUT'}

# Rows whose operand B is not a weight, so _w_layout must not reserve room for it in W_BUF;
# the value says where it lives instead.  Both of stage 2's attention matmuls are like
# that and both name X_BUF: a lane's K and V arrive over the boundary from stage 1, so
# they land in the inbound buffer and are contracted against where they lie.  A 'Y:'
# prefix works too, for a lane that projected its own K and V.
BIND_W: Dict[int, str] = {14: 'X:QKV_K', 16: 'X:QKV_V'}

# The three buffers a bind value may name, and the one a bare value means for each table.
_BIND_BUFS = ('X', 'W', 'Y')


def _bind(value: Optional[str], own: str) -> Tuple[str, str]:
    """Split a BIND_X / BIND_W value into (buffer, region name).

    A bare value names a region of the table's OWN buffer -- X_BUF for BIND_X, W_BUF for
    BIND_W -- and an 'X:' / 'W:' / 'Y:' prefix sends the operand somewhere else.  Returning
    the buffer explicitly lets one predicate stand for every case, rather than a chain of
    startswith tests that has to be extended each time a partition moves a tensor.
    """
    if not value:
        return own, ''
    if len(value) > 1 and value[1] == ':' and value[0] in _BIND_BUFS:
        return value[0], value[2:]
    return own, value


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bandwidths:
    """Per-link bandwidths [GB/s].  An unset (None) or 0 link costs no time."""
    c2c:   Optional[float] = None   # on-chip cluster-to-cluster hand-off
    lpddr: Optional[float] = None   # cluster <-> LPDDR, per cluster, already capped


@dataclass
class LayerRow:
    """One kernel row of the template."""
    excel_row:     int
    block:         str
    n_rep:         int
    op:            str
    kernel:        str
    flops:         float
    weights_bytes: float
    din_bytes:     float
    dout_bytes:    float
    is_gemm:       bool
    in_dims:       List[int] = field(default_factory=list)
    out_dims:      List[int] = field(default_factory=list)


@dataclass
class Item:
    """One placement of one template row on one stage."""
    row:   LayerRow
    part:  int = 0            # which part of the stage it belongs to -- see Part
    ms:    float = 0.0        # latency of ONE cluster's share of it
    x_off: int = 0            # element offset of its A operand inside its A buffer
    w_off: int = 0            # element offset of its B operand inside its B buffer
    y_off: int = 0            # element offset of its OUTPUT inside Y_BUF
    x_buf: str = 'X'          # which buffer its A operand is in -- see BIND_X
    w_buf: str = 'W'          # which buffer its B operand is in -- see BIND_W
    # The NAMED region each operand is bound to, '' for one that just addresses the buffer
    # base as scratch.  Only cwt reads these, and only to decide which operands move to the
    # second copy on an odd item: a scratch read is not part of the dataflow and must NOT be
    # shifted, because it can legitimately walk further than the region it starts in.
    x_reg: str = ''
    w_reg: str = ''


@dataclass(frozen=True)
class Cut:
    """How one part of a stage divides its rows over the stage's lanes.

    THE ONE PLACE the partition's arithmetic lives, and neither half of the script
    computes a lane count of its own: the cost model divides a row's FLOPs by `lanes_eff`,
    the shape builder cuts its dims by `q_split`, `groups` and `tiles`, and _row_footprint
    sizes the buffers from the shape.
    """
    n_par:  int
    groups: int = 1           # lanes dividing the LEADING axis (the attention heads)
    tiles:  int = 1           # serial passes ONE lane takes over its own share
    heads:  int = 1           # the model's head count, from Geometry -- see hpl

    @property
    def hpl(self) -> int:
        """Heads ONE lane owns.

        Not always 1: `groups` divides the heads over the lanes, so a lane count that is
        not a multiple of the head count leaves several heads on a lane -- six lanes over
        four heads can only be 2 groups of 2.  Everything that has to be issued ONCE PER
        HEAD reads it from here.
        """
        return max(1, self.heads // max(1, self.groups))

    @property
    def q_split(self) -> int:
        """How many ways the lanes divide the ROW axis (queries, or tokens elsewhere)."""
        return max(1, self.n_par // max(1, self.groups))

    @property
    def lanes_eff(self) -> float:
        """How many ways the row's WHOLE work is divided across the stage.

        `tiles` is NOT in here.  Tiling is a serial loop on one lane -- the same work in
        more passes -- so it shrinks the footprint and never the latency.
        """
        return float(self.n_par)


@dataclass
class Part:
    """One run of a stage's rows that is divided over the lanes in one particular way.

    A stage is a list of these.  Splitting a stage into parts is NOT a rendezvous and
    costs nothing at run time -- nothing crosses between them, so the emitted C runs them
    back to back inside one compute_stage_N() -- it only says that the rows in one part
    share a shape rule and, where `tiles` > 1, one enclosing loop.
    """
    cut:   Cut
    items: List[Item] = field(default_factory=list)
    lo:    int = 0            # index of its first item in the stage's flat item list
    hi:    int = 0            # one past its last


@dataclass
class Stage:
    idx:        int
    n_par:      int
    split:      str
    parts:      List[Part] = field(default_factory=list)
    pad:        int = 0       # cids left IDLE after this stage; see _stage_bases
    latency:  float = 0.0     # compute [ms] of ONE cluster
    dout_ms:  float = 0.0     # outbound c2c hand-off [ms], per cluster
    din_ms:   float = 0.0     # inbound LPDDR read [ms], per cluster (before the compute)
    lout_ms:  float = 0.0     # outbound LPDDR write [ms], per cluster
    din_c2c:  float = 0.0     # inbound hand-off bytes, per cluster
    dout_c2c: float = 0.0     # outbound hand-off bytes, per cluster
    layout:   Dict[str, int] = field(default_factory=dict)   # X regions, in elements
    y_layout: Dict[str, int] = field(default_factory=dict)   # Y regions, in elements
    x_elems:  int = 0
    w_elems:  int = 0
    y_elems:  int = 0

    @property
    def items(self) -> List[Item]:
        """The stage's rows in order, flattened across its parts."""
        return [it for p in self.parts for it in p.items]

    def part_for(self, k: int) -> Part:
        """The part item `k` of the flat list belongs to."""
        return self.parts[self.items[k].part]

    def cut_for(self, k: int) -> Cut:
        return self.part_for(k).cut

    @property
    def transfer_ms(self) -> float:
        """Everything the TRANSFER phase of a step costs this stage."""
        return self.dout_ms + self.lout_ms

    @property
    def compute_ms(self) -> float:
        """Everything a step costs this stage before the transfer phase.

        The inbound LPDDR read is serialized in front of the compute -- the generated loop
        issues it and holds the worker cores until it lands -- so a step's compute half is
        the two of them added.
        """
        return self.din_ms + self.latency


@dataclass
class Geometry:
    """The model's shape, read from the template rather than hard-coded."""
    tokens:  int      # i_2 of the token stream (1092)
    d_model: int      # 128
    heads:   int      # 4
    d_head:  int      # 32
    patch:   int      # patch length, the token matrix's row (84)


@dataclass
class Results:
    flops_per_ms:            Dict[str, Dict[str, float]] = field(default_factory=dict)
    row_latencies:           Dict[str, Dict[int, float]] = field(default_factory=dict)
    block_latencies:         Dict[str, List[Dict]]       = field(default_factory=dict)
    total_latency:           Dict[str, float]            = field(default_factory=dict)
    stages:                  Dict[str, List[Stage]]      = field(default_factory=dict)
    best_throughput:         Dict[str, float]            = field(default_factory=dict)
    best_throughput_overlap: Dict[str, float]            = field(default_factory=dict)
    geom:                    Optional[Geometry]          = None


# ---------------------------------------------------------------------------
# Cell value helpers
# ---------------------------------------------------------------------------

EXCEL_ERRORS = ('#NAME?', '#REF!', '#DIV/0!', '#VALUE!', '#N/A', '#NULL!', '#NUM!', '#ERROR!')


def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip() == ''


def _is_excel_error(x) -> bool:
    return isinstance(x, str) and x.strip() in EXCEL_ERRORS


def _to_str(x) -> str:
    return '' if _is_blank(x) else str(x).strip()


def _to_float(x) -> float:
    if _is_blank(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in EXCEL_ERRORS:
        raise ValueError(f"cell holds Excel error {s!r}")
    return float(s)


def _to_int(x) -> int:
    return int(_to_float(x))


def col_letter(idx: int) -> str:
    """0-based column index -> Excel letter (A, B, ..., Z, AA, AB, ...)."""
    s = ''
    n = idx
    while True:
        s = chr(ord('A') + n % 26) + s
        n = n // 26 - 1
        if n < 0:
            return s


def _prod(xs) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _pad(n: int, mult: int) -> int:
    """Round n up to a multiple of `mult`.

    The baremetal kernels are vectorised and unrolled and none of them guards its tail: a
    dimension that is not a multiple of the kernel's step is read and written past its end,
    and for a 2-D kernel an ODD row length also puts every other row on a 2-byte boundary,
    where a 32-bit v2h access is misaligned.  Padding the dimension pads the buffer with
    it, so the accesses stay legal; the trip counts are unchanged, so no cycles move.
    """
    return ((n + mult - 1) // mult) * mult


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def read_grid(path: str) -> List[List]:
    """Read a template into a row-major 2D list, returning calculated values for formulas."""
    ext = Path(path).suffix.lower()

    if ext == '.csv':
        with open(path, newline='') as f:
            return list(csv.reader(f))

    if ext in ('.xlsx', '.xlsm'):
        from openpyxl import load_workbook
        wb = load_workbook(path, data_only=True)
        ws = wb.active
        return [[ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
                for r in range(1, ws.max_row + 1)]

    if ext == '.ods':
        import pandas as pd
        df = pd.read_excel(path, engine='odf', sheet_name=0, header=None)
        df = df.where(df.notnull(), None)
        return df.values.tolist()

    raise ValueError(f"Unsupported file extension: {ext}. Use .csv, .xlsx, .xlsm or .ods")


def find_header_row(grid: List[List]) -> int:
    """Return the 0-based index of the header row.

    The header row is the first row in which *every* named field of HEADER_ALIASES has a
    matching label cell (the same test build_column_map uses).  Secondary sub-header rows
    (e.g. "Data_in (Bytes)") do not match, so they are skipped automatically.
    """
    for r, row in enumerate(grid):
        if all(
            any(_to_str(v) in aliases for v in row)
            for aliases in HEADER_ALIASES.values()
        ):
            return r
    raise ValueError(
        "Header row not found: no row labels all of "
        + " / ".join(a[0] for a in HEADER_ALIASES.values())
        + f". (accepted aliases: {HEADER_ALIASES})"
    )


def find_data_end(grid: List[List], cols: Dict[str, int], first_row: int) -> int:
    """Return the 0-based row index where the layer table ends (exclusive).

    Reading stops at the first row whose block-name cell (column A) is filled but whose
    operation cell (column C) is empty.  A real block header always carries its first
    layer's operation on the same row, so this only fires on the bottom parameter table.
    """
    bcol, ocol = cols['block'], cols['op']
    for r in range(first_row, len(grid)):
        row = grid[r]
        block = _to_str(row[bcol]) if bcol < len(row) else ''
        op    = _to_str(row[ocol]) if ocol < len(row) else ''
        if block and not op:
            return r
    return len(grid)


def build_column_map(header_row: List) -> Dict[str, int]:
    """Build {field -> column index}, header-based for named fields, defaults for the rest.

    The i_*/o_* tensor-shape columns are OPTIONAL: located when present (real-kernel
    emission needs them), omitted otherwise, so --wait-only templates still parse.
    """
    cols = dict(DEFAULT_COLS)
    for field_name, aliases in HEADER_ALIASES.items():
        match = next((c for c, v in enumerate(header_row) if _to_str(v) in aliases), None)
        if match is None:
            raise ValueError(
                f"Header for {field_name!r} not found (expected one of {aliases}). "
                f"Got header row: {[_to_str(x) for x in header_row]}"
            )
        cols[field_name] = match
    for field_name, label in DIM_HEADERS.items():
        match = next((c for c, v in enumerate(header_row) if _to_str(v) == label), None)
        if match is not None:
            cols[field_name] = match
    return cols


def check_for_errors(grid: List[List], cols: Dict[str, int],
                     first_row: int, last_row: int) -> None:
    """Raise a clear ValueError listing any Excel error cells in the data columns."""
    bad: List[str] = []
    for r in range(first_row, last_row + 1):
        if r >= len(grid):
            break
        row = grid[r]
        for field_name, c in cols.items():
            if c < len(row) and _is_excel_error(row[c]):
                bad.append(f"  {col_letter(c)}{r+1}  ({field_name}) = {row[c]!r}")
    if bad:
        raise ValueError(
            "Template contains unrecalculated formula errors in the data region:\n"
            + "\n".join(bad)
            + "\n\nOpen the file in Excel/LibreOffice, recalculate (Ctrl+Shift+F9), "
              "save, and try again."
        )


def parse_layers(grid: List[List], cols: Dict[str, int],
                 first_row: int, last_row: int) -> List[LayerRow]:
    """Parse data rows in the inclusive 0-based range [first_row, last_row]."""
    max_col = max(cols.values())
    layers: List[LayerRow] = []
    for r in range(first_row, last_row + 1):
        if r >= len(grid):
            break
        row = list(grid[r])
        if len(row) <= max_col:
            row += [None] * (max_col + 1 - len(row))

        op = _to_str(row[cols['op']])
        if not op or _is_blank(row[cols['flops']]):
            continue  # skip blank/spacer rows inside the region

        def dims(prefix: str) -> List[int]:
            """Read prefix_1..prefix_4 as ints, dropping trailing empty cells."""
            vals: List[Optional[int]] = []
            for k in range(1, 5):
                key = f'{prefix}_{k}'
                if key not in cols:
                    break
                cell = row[cols[key]] if cols[key] < len(row) else None
                vals.append(None if _is_blank(cell) else _to_int(cell))
            while vals and vals[-1] is None:
                vals.pop()
            return [v for v in vals if v is not None]

        kernel = _to_str(row[cols['kernel']])
        layers.append(LayerRow(
            excel_row     = r + 1,  # back to 1-indexed for messages / printing
            block         = _to_str(row[cols['block']]),
            n_rep         = _to_int(row[cols['n_rep']]),
            op            = op,
            kernel        = kernel,
            flops         = _to_float(row[cols['flops']]),
            weights_bytes = _to_float(row[cols['weights']]) * BYTES_PER_ELEM,
            din_bytes     = _to_float(row[cols['din']])     * BYTES_PER_ELEM,
            dout_bytes    = _to_float(row[cols['dout']])    * BYTES_PER_ELEM,
            is_gemm       = kernel.lower() in GEMM_KERNELS,
            in_dims       = dims('in'),
            out_dims      = dims('out'),
        ))
    return layers


def load_template(path: str) -> List[LayerRow]:
    grid       = read_grid(path)
    header_row = find_header_row(grid)
    cols       = build_column_map(grid[header_row])
    data_start = header_row + 1
    data_end   = find_data_end(grid, cols, data_start)
    check_for_errors(grid, cols, data_start, data_end - 1)
    layers = parse_layers(grid, cols, data_start, data_end - 1)
    if not layers:
        raise ValueError("No data rows found below the header row")
    # The template fills the block name and N_repetitions only on the row that OPENS a
    # block; its continuation rows leave both blank.  Carry them down, so every row knows
    # which block it belongs to and how many times it runs -- build_stages' placement
    # invariant is stated per row, and a blank would read as "runs zero times".
    block, n_rep = '', 1
    for r in layers:
        if r.block:
            block, n_rep = r.block, r.n_rep
        r.block, r.n_rep = block, n_rep
    return layers


# ---------------------------------------------------------------------------
# Model geometry
# ---------------------------------------------------------------------------

def read_geometry(layers: List[LayerRow]) -> Geometry:
    """Read the model's shape off the rows the partition already names.

    Nothing here is a constant of this script: the hand-off descriptors, the lane maps and
    the buffer layouts all follow from these five numbers, so a different PRB count is a
    different CSV.  Row 10 gives the token count and d_model, row 14 the head count and
    d_head, row 5 the patch length.
    """
    by_row = {r.excel_row: r for r in layers}
    qkv    = by_row.get(10)
    scores = by_row.get(14)
    front  = by_row.get(5)
    if not (qkv and scores and front):
        missing = [n for n, r in ((10, qkv), (14, scores), (5, front)) if not r]
        raise ValueError(
            f'template is missing row(s) {missing}; the geometry is read from row 5 (the '
            f'first dense), row 10 (the QKV projection) and row 14 (Q*K^T)')
    if not (qkv.in_dims and scores.in_dims and front.in_dims):
        raise ValueError(
            'the template has no i_*/o_* tensor-shape columns; they are needed to derive '
            'the hand-off descriptors, so this partition cannot be built without them')

    g = Geometry(tokens  = qkv.in_dims[-2],
                 d_model = qkv.in_dims[-1],
                 heads   = scores.in_dims[1],
                 d_head  = scores.in_dims[-1],
                 patch   = front.in_dims[-1])
    if g.heads * g.d_head != g.d_model:
        raise ValueError(
            f'row 14 says {g.heads} heads x {g.d_head} d_head = {g.heads * g.d_head}, but '
            f'row 10 says d_model = {g.d_model}')
    if qkv.out_dims[-1] != 3 * g.d_model:
        raise ValueError(
            f"row 10 ('{qkv.op}') projects to {qkv.out_dims[-1]}, not 3 x d_model = "
            f'{3 * g.d_model}; the 1 -> 2 hand-off reads Q, K and V out of that one output')
    return g


# ---------------------------------------------------------------------------
# Stage construction
# ---------------------------------------------------------------------------

def _stage_bases() -> Tuple[List[int], int]:
    """Each stage's first cid, and one past the last cid the pipeline OCCUPIES.

    NOT a running sum of n_par: a stage may leave cids IDLE after it (`pad`) so that the
    next one starts on a column boundary, which the stages touching LPDDR need since they
    derive their node from the lane.  The emitted tables are read BY CID, so they span the
    gaps even though nothing runs in them.  THE ONLY PLACE cid BASES ARE COMPUTED.
    """
    bases, at = [], 0
    for spec in STAGES:
        bases.append(at)
        at += spec['n_par'] + spec.get('pad', 0)
    return bases, at


def build_stages(layers: List[LayerRow], geom: Geometry) -> List[Stage]:
    """Bind STAGES to the parsed rows, checking that the two stay in step."""
    by_row = {r.excel_row: r for r in layers}
    stages: List[Stage] = []
    placed: Dict[int, int] = {}

    for i, spec in enumerate(STAGES):
        st = Stage(idx=i, n_par=spec['n_par'], split=spec['split'],
                   pad=spec.get('pad', 0))
        at = 0
        for j, pspec in enumerate(spec['parts']):
            cut  = Cut(n_par=spec['n_par'], groups=pspec.get('head_groups', 1),
                       tiles=pspec.get('tiles', 1), heads=geom.heads)
            part = Part(cut=cut, lo=at)
            for rn in pspec['rows']:
                if rn not in by_row:
                    raise ValueError(
                        f'stage {i + 1} names template row {rn}, which the parser did not '
                        f'read as a layer (kernel column empty, or outside the data region)')
                part.items.append(Item(row=by_row[rn], part=j))
                placed[rn] = placed.get(rn, 0) + 1
                at += 1
            part.hi = at
            st.parts.append(part)
        stages.append(st)

    # The placement invariant: the FLOPs column already carries the xN_repetitions factor
    # and each placement is costed against it, so the stage table costs a repeated row
    # correctly exactly when it places it n_rep times.  Catches both halves of the likely
    # mistake -- a row left out (its work vanishes) and a dropped row placed anyway
    # (charged for a transfer that already carries it).
    problems = []
    for r in layers:
        want = 0 if r.excel_row in DROPPED_ROWS else r.n_rep
        got  = placed.get(r.excel_row, 0)
        if got != want:
            why = (' (it is in DROPPED_ROWS: a transfer carries it)'
                   if r.excel_row in DROPPED_ROWS else '')
            problems.append(
                f'  row {r.excel_row} ({r.kernel} {r.op[:40]!r}): placed {got} time(s), '
                f'wanted {want}{why}')
    if problems:
        raise ValueError(
            'the stage table does not place every template row N_repetitions times:\n'
            + '\n'.join(problems)
            + f'\nDROPPED_ROWS = {list(DROPPED_ROWS)} must be placed 0 times.')

    # The lane arithmetic every hand-off descriptor and every LPDDR leg is derived from.
    _check_lane_geometry(stages, geom)
    _check_handoff_sizing(stages, geom)
    _check_alignment(stages, geom)
    return stages


def _check_alignment(stages: List[Stage], geom: Geometry) -> None:
    """Every descriptor must start, step and run on whole DMA lines.

    A run that starts mid-line straddles two of them and the leg goes at HALF RATE, while
    _transfer_ms, which charges bytes and nothing else, reports the same number either
    way.  That is a class of error the cost model is structurally blind to, so this is the
    only place it can be caught: the region placer (_place, _line) is what makes it hold,
    and this is what proves it did.
    """
    def audit(what: str, xs: List[Xfer]) -> None:
        for x in xs:
            bad = {n: v for n, v in (('src_off', x.src_off), ('dst_off', x.dst_off),
                                     ('chunk', x.chunk), ('src_stride', x.src_stride),
                                     ('dst_stride', x.dst_stride))
                   # A stride is only walked when there is more than one run.
                   if v % DMA_LINE_B and not (x.reps == 1 and n.endswith('stride'))}
            if bad:
                raise ValueError(
                    f'{what}: the {x.name} descriptor is not {DMA_LINE_B} B aligned -- '
                    + ', '.join(f'{n} = {v}' for n, v in sorted(bad.items()))
                    + f'.  Every run would straddle two DMA lines and the leg would go at '
                      f'half rate, which the byte-based cost model cannot see.')

    for i, st in enumerate(stages):
        for p in range(st.n_par):
            audit(f'stage {i + 1} lane {p} hand-off', _handoff(stages, geom, i, p))
        for leg in _lpddr_legs(i):
            reg = _leg_region(stages, geom, i, leg, 'TensorPool')
            # Both the L1 end and the PIECE stride.  The piece length is a stride only
            # when something walks by it -- a leg with several pieces, or several item
            # slots -- so a single-piece, single-slot leg is one contiguous descriptor and
            # only its start matters, which is why the patch tensor may be an odd
            # 183,456 B.
            walks = leg.shares > 1 or leg.slots > 1
            bad = {n: v for n, v in
                   (('L1 offset', reg.off * BYTES_PER_ELEM),
                    ('piece', _leg_piece_bytes(stages, geom, i, leg, 'TensorPool')))
                   if v % DMA_LINE_B and not (n == 'piece' and not walks)}
            if bad:
                raise ValueError(
                    f'stage {i + 1}: the {leg.name} LPDDR leg is not {DMA_LINE_B} B '
                    f'aligned -- ' + ', '.join(f'{n} = {v}' for n, v in sorted(bad.items())))


def _check_lane_geometry(stages: List[Stage], geom: Geometry) -> None:
    """Refuse a partition whose lanes do not divide the model's axes.

    Every one of these is a divisibility the descriptor builders assume: a lane owning a
    fractional number of tokens or heads would need a descriptor that cannot be written, so
    it is caught here -- naming the axis -- rather than emitted.
    """
    for i, st in enumerate(stages):
        for j, part in enumerate(st.parts):
            cut = part.cut
            # `groups` cuts the HEAD axis and what is left of the lanes cuts the ROW axis
            # (queries on the attention rows, tokens everywhere else).  Both must come out
            # whole.
            if geom.heads % cut.groups or st.n_par % cut.groups:
                raise ValueError(
                    f'stage {i + 1} part {j + 1} declares head_groups={cut.groups}, which '
                    f'does not divide both the {geom.heads} heads and its {st.n_par} lanes')
            if geom.tokens % cut.q_split:
                raise ValueError(
                    f'stage {i + 1} part {j + 1} splits the row axis {cut.q_split} ways '
                    f'({st.n_par} lanes / {cut.groups} head groups), which does not divide '
                    f'the {geom.tokens} tokens')
    # THE STAGES MAY HAVE DIFFERENT WIDTHS -- a boundary fans in or out, and _handoff
    # writes the descriptors for it.  A leg's `shares` must still divide every touching
    # stage's width, because a cluster owns a WHOLE number of pieces; that is what lets two
    # stages of different widths name the same bytes.
    for i, st in enumerate(stages):
        for leg in _lpddr_legs(i):
            if not HBM_CHAN_PLACEMENT[leg.edge]:
                raise ValueError(f"the {leg.edge} edge carries no LPDDR nodes")
            if leg.node == 'share' and leg.shares % st.n_par:
                raise ValueError(
                    f'the LPDDR tensor {leg.name!r} is cut into {leg.shares} pieces and '
                    f'touched by stage {i + 1}, which is {st.n_par} lanes wide; a cluster '
                    f'owns a whole number of pieces, so the widths must divide the pieces')
    _check_lpddr_slots(stages)
    bases, occupied = _stage_bases()
    if occupied > GRID_X * GRID_Y:
        raise ValueError(
            f'the partition occupies {occupied} cids (widths plus padding), more than the '
            f'{GRID_X} x {GRID_Y} grid provides')


def _slots_needed(w: int, r: int, mode: str) -> int:
    """Item-rotation depth a tensor written by stage `w` and read by stage `r` needs.

    ctt, WAVEFRONT m + s.  Item m is written in the TRANSFER phase of step m + w and read
    in the COMPUTE phase of step m + r, so the writes strictly between are items
    m+1 .. m+r-w-1 and r - w slots suffice.  Item m + (r - w) does reuse slot m % slots,
    but in that step's transfer phase, after the barrier that ended the read's compute
    phase.

    cwt, WAVEFRONT m + 2s.  Item m is drained in step m + 2w + 1 and read at the TOP of
    step m + 2r.  THE WRITE IN THE SAME STEP AS THE READ HAS TO MISS TOO, and that is the
    whole difference: cwt has ONE barrier a step and nothing orders two different clusters
    inside it.  Hence 2(r - w).

    A tensor written and read inside one step needs 1 either way.
    """
    gap = r - w
    if gap <= 0:
        return 1
    return max(1, 2 * gap if mode == 'cwt' else gap)


def _check_lpddr_slots(stages: List[Stage]) -> None:
    """Hold a spilled tensor's rotation depth to the distance it has to survive.

    Checked against the DEEPER of the two modes, because LPDDR_LEGS declares one number and
    both apps read the same table.  See _slots_needed.

    Both ends must also agree on the cut, or they name different bytes.
    """
    ends: Dict[str, Dict[str, List[int]]] = {}
    legs: Dict[str, Leg] = {}
    for i in range(len(stages)):
        for leg in _lpddr_legs(i):
            ends.setdefault(leg.name, {'in': [], 'out': []})[leg.direction].append(i)
            prev = legs.setdefault(leg.name, leg)
            if (prev.shares, prev.slots, prev.node, prev.edge) != \
               (leg.shares, leg.slots, leg.node, leg.edge):
                raise ValueError(
                    f'the LPDDR tensor {leg.name!r} is declared two different ways -- '
                    f'{prev} against {leg}; both ends must cut and rotate it identically or '
                    f'they name different bytes')
    for name, e in ends.items():
        for w in e['out']:
            for r in e['in']:
                need = max(_slots_needed(w, r, m) for m in ('ctt', 'cwt'))
                if legs[name].slots < need:
                    raise ValueError(
                        f'the LPDDR tensor {name!r} is written by stage {w + 1} and read by '
                        f'stage {r + 1}, but its leg declares slots={legs[name].slots} and '
                        f'the schedules need {need} '
                        f'(ctt {_slots_needed(w, r, "ctt")}, cwt {_slots_needed(w, r, "cwt")}'
                        f'); a later item would overwrite it before it is read')


def _check_handoff_sizing(stages: List[Stage], geom: Geometry) -> None:
    """Hold the descriptor tables to what the cost model reports about them.

    Two things have to be true once a stage's lanes may send different amounts: the figure
    billed per cluster has to BOUND every lane, and everything sent has to arrive
    somewhere.  Lanes do not differ on this partition -- both boundaries are uniform --
    but a stage's dout is billed at the WORST-OFF lane all the same, because an average
    would bound nothing the moment a cut stops dividing evenly.
    """
    for i in range(len(stages) - 1):
        per_lane = [sum(x.nbytes for x in _handoff(stages, geom, i, p))
                    for p in range(stages[i].n_par)]
        billed = max(per_lane, default=0)
        senders = [p for p, nb in enumerate(per_lane) if nb]
        if not senders:
            raise ValueError(
                f'stage {i + 1} -> {i + 2}: no lane sends anything across the boundary')
        # Consumers may legitimately differ from one another too; what must hold is that
        # everything sent arrives somewhere.
        recv = [sum(x.nbytes
                    for p in senders
                    for x in _handoff(stages, geom, i, p) if x.dst_lane == c)
                for c in range(stages[i + 1].n_par)]
        if sum(per_lane) != sum(recv):
            raise ValueError(
                f'stage {i + 1} -> {i + 2}: {sum(per_lane)} B are sent and {sum(recv)} B '
                f'arrive; a descriptor is going somewhere nothing reads')
        if any(nb > billed for nb in per_lane):
            raise ValueError(
                f'stage {i + 1} -> {i + 2}: lanes move {per_lane} bytes, more than the '
                f'{billed} B the model bills per cluster')
        for p in senders:
            for x in _handoff(stages, geom, i, p):
                span  = x.dst_off + (x.reps - 1) * x.dst_stride + x.chunk
                limit = {'X': _end(_x_regions(stages, geom, i + 1)),
                         'Y': _end(_y_regions(stages, geom, i + 1, 'TensorPool')),
                         }[x.dst_buf] * BYTES_PER_ELEM
                if span > limit:
                    raise ValueError(
                        f'stage {i + 1} lane {p}: the {x.name} descriptor reaches {span} B '
                        f"into the consumer's {x.dst_buf}_BUF, past the {limit} B of named "
                        f'regions there')
        # EACH BOUNDARY MOVES EACH TENSOR EXACTLY AS MANY TIMES AS ITS CONSUMERS NEED IT,
        # AND NO MORE.  _handoff_replication says how many copies the consuming stage's cut
        # demands, and the check has to be exact in both directions: a MISSING copy starves
        # a lane, and a SPURIOUS one is a silent wrong answer rather than a crash, because
        # the byte totals would still add up.
        whole = geom.tokens * geom.d_model * BYTES_PER_ELEM
        for name in HANDOFF_TENSORS[i]:
            want  = whole * _handoff_replication(stages, i, name)
            moved = sum(x.nbytes
                        for p in senders
                        for x in _handoff(stages, geom, i, p) if x.name == name)
            if moved != want:
                raise ValueError(
                    f'stage {i + 1} -> {i + 2}: the boundary moves {moved} B of the '
                    f'{name} tensor, not the {want} B its consumers need '
                    f'({whole} B x {_handoff_replication(stages, i, name)} copies)')


# ---------------------------------------------------------------------------
# Hand-off descriptors
# ---------------------------------------------------------------------------
#
# One descriptor is one flex_dma_async_2d: `reps` runs of `chunk` bytes, walking the source
# by src_stride and the destination by dst_stride.  They are built HERE, per producing
# cluster, and the emitted C walks the same constants -- which is what gives
# _check_handoff_sizing something to check.

@dataclass(frozen=True)
class Xfer:
    """One 2D transfer from one producing lane to one consuming lane."""
    name:       str    # the tensor it carries, for the emitted comment
    dst_lane:   int    # lane WITHIN the consuming stage
    dst_buf:    str    # 'X' or 'Y' -- which of the consumer's buffers it lands in
    dst_off:    int    # bytes
    src_off:    int    # bytes, from the producer's Y_BUF
    chunk:      int    # bytes per run
    dst_stride: int
    src_stride: int
    reps:       int

    @property
    def nbytes(self) -> int:
        return self.chunk * self.reps


def _overlap(a0: int, a1: int, b0: int, b1: int) -> Tuple[int, int]:
    """The intersection of two half-open ranges, empty when lo >= hi."""
    return max(a0, b0), min(a1, b1)


def _cut_of(stages: List[Stage], i: int) -> Cut:
    """The cut stage `i` produced its outbound tensor under.

    A hand-off carries what the LAST part of the producing stage wrote, so that part's
    `groups` and `q_split` are what say which head and which rows a lane holds.  Read
    rather than restated, so changing a stage's width changes the boundary it feeds.
    """
    return stages[i].parts[-1].cut




# The tensors each boundary carries, checked in _check_handoff_sizing against the count
# _handoff_replication gives.
#
#   boundary 0 (1 -> 2)   Q, K and V.  Three tensors of [tokens, d_model] between them, i.e.
#                         the whole projection output.
#   boundary 1 (2 -> 3)   A*V, head-merged on the way: [tokens, d_model].
HANDOFF_TENSORS: Dict[int, Tuple[str, ...]] = {
    0: ('QKV_Q', 'QKV_K', 'QKV_V'),
    1: ('ATTN',),
}


def _handoff_replication(stages: List[Stage], i: int, name: str) -> int:
    """How many copies of `name` boundary `i` must carry.

    ONE, EXCEPT WHERE THE CONSUMER'S CUT REALLY NEEDS SEVERAL, and only the attention keys
    and values do: a stage-2 lane owns a slice of the QUERIES and reduces over ALL the
    keys, so the `q_split` lanes of one head group each need that group's whole K and whole
    V.  Q does not -- the lanes take disjoint thirds of it -- and A*V on boundary 1 is cut
    by token and crosses once as well.

    THIS IS THE COST OF SPLITTING THE QUERIES and it is meant to be visible.  Derived from
    the consuming stage's cut rather than tabulated, so a partition that stops splitting
    them comes back to 1 everywhere on its own.
    """
    if i == 0 and name in ('QKV_K', 'QKV_V'):
        return _cut_of(stages, i + 1).q_split
    return 1


def _handoff_qkv(stages: List[Stage], geom: Geometry, i: int, lane: int) -> List[Xfer]:
    """Boundary 1 -> 2: the projection output, cut by HEAD and by TOKEN at once.

    THIS IS ROWS 11 AND 12, and the SOURCE ADDRESSING performs them: row 10's output is
    [tokens, 3*d_model] row-major, so head h of stripe s is column s*d_model + h*d_head of
    every row -- a d_head RUN at a 3*d_model STRIDE, once per token, and free, because
    d_head is exactly one 64 B DMA line.  THE DESTINATION IS CONTIGUOUS: a consumer wants
    each head's Q, K and V as plain [rows, d_head] tensors.

    BOTH SIDES CUT THE TOKENS, AND NOT THE SAME WAY.  A producing lane holds a token slab
    and owes every consumer only that much, so the boundary FANS IN; a consuming lane
    holds a query slab and WHOLE keys, so Q is intersected with its slab and crosses once
    while K and V go to ALL the lanes of a head group.  So K AND V ARE REPLICATED
    `q_split` TIMES ON THE WIRE -- charged, not hidden: _check_handoff_sizing expects that
    multiple rather than treating it as an overlap.  Thirty-two descriptors per producer.

    Head and token ranges are INTERSECTED rather than assumed to line up, so a producer
    slab straddling two consumer slabs falls out of the same loop.
    """
    cut  = _cut_of(stages, i)
    nxt  = stages[i + 1].n_par
    cut2 = _cut_of(stages, i + 1)
    hpl  = cut.hpl                                   # heads on this producing lane
    hpn  = cut2.hpl                                  # heads on one consumer
    dh   = geom.d_head * BYTES_PER_ELEM              # one run, one 64 B line
    dm   = geom.d_model * BYTES_PER_ELEM
    row  = len(HANDOFF_TENSORS[i]) * dm              # 3*d_model: one row of the projection
    tt   = geom.tokens // cut.q_split                # rows one PRODUCING lane holds
    tq   = geom.tokens // cut2.q_split               # rows one CONSUMING lane scores
    hp, qp = (lane // cut.q_split) * hpl, lane % cut.q_split
    src0 = _region(_y_regions(stages, geom, i, 'TensorPool'), 'QKV').off * BYTES_PER_ELEM
    x2   = {r.name: r for r in _x_regions(stages, geom, i + 1)}

    out: List[Xfer] = []
    for k in range(nxt):
        hk, qk = (k // cut2.q_split) * hpn, k % cut2.q_split
        h0, h1 = _overlap(hp, hp + hpl, hk, hk + hpn)
        for s, nm in enumerate(HANDOFF_TENSORS[i]):
            # Q lands in this consumer's query slab; K and V land in a whole-token slab, at
            # the row this producer's own tokens start on.  One `slab` and one row range
            # per tensor is the only difference between the two cases.
            queries = nm == 'QKV_Q'
            lo, hi  = (_overlap(qp * tt, qp * tt + tt, qk * tq, qk * tq + tq) if queries
                       else (qp * tt, qp * tt + tt))
            if lo >= hi:
                continue
            slab = (tq if queries else geom.tokens) * dh
            dst  = lo - (qk * tq if queries else 0)
            for h in range(h0, h1):
                out.append(Xfer(nm, k, 'X',
                                x2[nm].off * BYTES_PER_ELEM + (h - hk) * slab + dst * dh,
                                src0 + (lo - qp * tt) * row + s * dm + h * dh,
                                dh, dh, row, hi - lo))
    return out


def _handoff_attn(stages: List[Stage], geom: Geometry, i: int, lane: int) -> List[Xfer]:
    """Boundary 2 -> 3: A*V, transposed back to token-major and head-merged on the way.

    THIS IS ROWS 17 AND 18, and the destination stride is where they live: writing one
    d_head run per token at column h*d_head of a d_model-wide row IS the transpose back to
    token-major and the reshape that folds the head axis into the feature axis.  Every run
    is one whole DMA line, so scattering by column costs no more than a contiguous copy.

    A FAN, NOT A PAIRING: a producer owes a descriptor to each consumer its rows reach,
    once per head, and the ranges are intersected rather than assumed to line up -- 364
    does not divide 546, so each head group's middle query slab straddles both consumers
    and issues two descriptors where the outer two issue one.  Sixteen in total.

    Each source overlaps only its own destinations, so A*V moves EXACTLY ONCE.
    """
    cut  = _cut_of(stages, i)
    nxt  = stages[i + 1].n_par
    hpl  = cut.hpl
    dh   = geom.d_head * BYTES_PER_ELEM
    dm   = geom.d_model * BYTES_PER_ELEM
    tq   = geom.tokens // cut.q_split      # rows one producing lane holds
    tt   = geom.tokens // nxt              # tokens one consumer owns
    h, q = lane // cut.q_split, lane % cut.q_split
    y1   = {r.name: r for r in _y_regions(stages, geom, i, 'TensorPool')}
    y2   = {r.name: r for r in _y_regions(stages, geom, i + 1, 'TensorPool')}
    src0 = y1['ATTNH'].off * BYTES_PER_ELEM
    dst0 = y2['FFN_RESID'].off * BYTES_PER_ELEM

    out: List[Xfer] = []
    for k in range(nxt):
        lo, hi = _overlap(q * tq, (q + 1) * tq, k * tt, (k + 1) * tt)
        if lo >= hi:
            continue
        for e in range(hpl):
            out.append(Xfer('ATTN', k, 'Y',
                            dst0 + (lo - k * tt) * dm + (h * hpl + e) * dh,
                            src0 + e * tq * dh + (lo - q * tq) * dh,
                            dh, dm, dh, hi - lo))
    return out


# One builder per boundary, indexed by the PRODUCING stage.  The two carry different
# tensors in different shapes, so they are two functions rather than one with a flag.
_HANDOFF_BUILDERS = {0: _handoff_qkv, 1: _handoff_attn}


def _handoff(stages: List[Stage], geom: Geometry, i: int, lane: int) -> List[Xfer]:
    """The transfers cluster `lane` of stage i issues to stage i+1."""
    if i >= len(stages) - 1:
        return []
    if i not in _HANDOFF_BUILDERS:
        raise ValueError(
            f'no hand-off builder for the stage {i + 1} -> {i + 2} boundary; '
            f'_HANDOFF_BUILDERS covers {sorted(_HANDOFF_BUILDERS)}')
    return _HANDOFF_BUILDERS[i](stages, geom, i, lane)


def _handoff_consts(stages: List[Stage], geom: Geometry) -> List[Tuple[str, List]]:
    """The constants each boundary's emitted transfer loop is written in terms of.

    One (comment, defines) pair per boundary, in order.  Neither push helper contains a
    literal, so a change of width in STAGES is a change of one number and not an edit to
    the C.
    """
    c0, c1 = _cut_of(stages, 0), _cut_of(stages, 1)
    y0 = {r.name: r.off * BYTES_PER_ELEM
          for r in _y_regions(stages, geom, 0, 'TensorPool')}
    x1 = {r.name: r.off * BYTES_PER_ELEM for r in _x_regions(stages, geom, 1)}
    y1 = {r.name: r.off * BYTES_PER_ELEM
          for r in _y_regions(stages, geom, 1, 'TensorPool')}
    y2 = {r.name: r.off * BYTES_PER_ELEM
          for r in _y_regions(stages, geom, 2, 'TensorPool')}
    tq = geom.tokens // c1.q_split

    shared = [
        ('HEADS',    geom.heads,   'attention heads in the model'),
        ('TOKENS',   geom.tokens,  'the whole model'),
        ('DH_B',     geom.d_head * BYTES_PER_ELEM,
                     'd_head in bytes -- one DMA run, one whole 64 B line'),
        ('DM_B',     geom.d_model * BYTES_PER_ELEM, 'd_model in bytes'),
    ]
    b0 = [
        ('S2_LANES', stages[1].n_par, 'lanes on the far side of boundary 1 -> 2'),
        ('S1_QSPLIT', c0.q_split, 'token slabs each of stage 1\'s head groups is cut into'),
        ('TT1',      geom.tokens // c0.q_split, 'tokens ONE stage-1 lane holds'),
        ('S1_HPL',   c0.hpl,      'heads ONE stage-1 lane holds'),
        ('S2_QSPLIT', c1.q_split, 'query slabs each of stage 2\'s head groups is cut into'),
        ('S2_HPL',   c1.hpl,      'heads ONE stage-2 lane owns -- HEADS / its head groups'),
        ('TQ',       tq,          'queries ONE stage-2 lane scores'),
        ('QKV_Y',    y0['QKV'],   'where stage 1 leaves the projection, [TT1, 3*d_model]'),
        ('QKV_ROW_B', 3 * geom.d_model * BYTES_PER_ELEM,
                      'one row of it, and so the SOURCE STRIDE: Q|K|V x HEADS x d_head'),
        ('QKV_Q_SLAB', tq * geom.d_head * BYTES_PER_ELEM,
                       'one head of Q at the far end: only this lane\'s queries, TQ x DH_B'),
        ('QKV_KV_SLAB', geom.tokens * geom.d_head * BYTES_PER_ELEM,
                        'one head of K or V there: EVERY token, TOKENS x DH_B, because a '
                        'query scores against every key'),
        ('QKV_Q_X',  x1['QKV_Q'], 'where the three land on stage 2, each contiguous'),
        ('QKV_K_X',  x1['QKV_K'], ''),
        ('QKV_V_X',  x1['QKV_V'], ''),
    ]
    b1 = [
        ('S3_LANES', stages[2].n_par, 'lanes on the far side of boundary 2 -> 3'),
        ('TT',       geom.tokens // stages[2].n_par, 'tokens one stage-3 lane owns'),
        ('HEAD_SRC', tq * geom.d_head * BYTES_PER_ELEM,
                     'stride between two heads of A*V in the producer, when a lane owns '
                     'more than one'),
        ('ATTNH_Y',  y1['ATTNH'], 'where a stage-2 lane leaves A*V, head-split'),
        ('FFN_RESID_Y', y2['FFN_RESID'],
                     'and where the boundary lands it, token-split and head-merged'),
    ]
    return [
        ('geometry both boundaries share', shared),
        ('boundary 1 -> 2: Q by query slab, K and V whole to every lane of a head group', b0),
        ('boundary 2 -> 3: A*V, head-merged by the destination stride', b1),
    ]


# ---------------------------------------------------------------------------
# Buffer layouts
# ---------------------------------------------------------------------------
#
# The named regions each stage keeps in X_BUF (where its inbound transfers land) and in
# Y_BUF (what its outbound transfers read).  These are the DMA endpoints; the KERNELS
# address the buffer base unless BIND_X sends one of them to a named region.
#
# A region is PLACED, not merely sized, so nothing a kernel writes can land on top of one a
# transfer still has to read.  That is why the Y layouts take the pool: the RedMulE row
# padding decides how far a GEMM's output actually reaches, and the residual has to start
# past it.

@dataclass(frozen=True)
class Region:
    name:  str
    off:   int      # elements into the buffer
    elems: int


def _end(regs: List[Region]) -> int:
    return max((r.off + r.elems for r in regs), default=0)


def _region(regs: List[Region], name: str) -> Region:
    for r in regs:
        if r.name == name:
            return r
    raise KeyError(f'no region {name!r} in {[r.name for r in regs]}')


@dataclass(frozen=True)
class Loop:
    """The one loop a PART of a stage may wrap its rows in, and what moves each pass.

    Two things put a part in a loop, and they never both do at once:

      * TILES -- passes over the query axis, taken to shrink a live intermediate.  The
        inbound operand and the outbound one walk; operand B does not, since every tile of
        the score matrix contracts against the same whole K.  Unused on this partition.
      * BATCH -- passes over the heads a lane owns, forced rather than chosen: each head
        has its OWN K and V, so they cannot be one GEMM.  Here operand B walks too.  This
        is what stage 2 runs, two heads to a lane.

    `count == 1` means no loop at all, and every stride is 0.
    """
    count:  int
    x:      int          # elements the first row's A operand advances per pass
    w:      int          # 1 when operand B advances per pass (batch only); see the caller,
                         # which takes each GEMM's own N*P -- they differ inside a stage
    y:      int          # elements the last row's output advances per pass
    var:    str          # the loop variable, for the emitted C
    note:   str


def _fp(st: Stage, k: int, pool: Optional[str]) -> Tuple[int, int, int]:
    """(X, W, Y) elements item `k` of this stage needs, per pass.

    The cut comes from the item's PART -- see Cut -- which is the one place a row's lane
    count, head grouping and tile count are decided.
    """
    it = st.items[k]
    x, w, y = _row_footprint(it.row, st.cut_for(k), pool)
    if not y and it.row.excel_row in BIND_Y:
        # A PLACED LAYOUT ROW THAT LEAVES SOMETHING BEHIND.  A layout row is normally a
        # view and claims nothing; row 13 is the exception, because row 14 contracts
        # against K^T and K^T has to exist somewhere.  Its out_dims describe ONE of the
        # three tensors the template row transposes, which is exactly the one wanted here.
        y = max(1, _prod(it.row.out_dims) // max(1, st.cut_for(k).groups))
    return x, w, y


def _sh(st: Stage, k: int) -> dict:
    """The kernel shape of item `k`, on the same footing as _fp."""
    return _row_shape(st.items[k].row, st.cut_for(k))


def _part_loop(st: Stage, part: Part, pool: Optional[str] = None) -> Loop:
    """The loop the rows of one part run inside -- see Loop.

    Taken per PART rather than per stage, because the answer can differ inside one stage:
    the score matrix is the only tensor here that would ever want tiling, and wrapping
    another part in the same loop would give it a pass count over an axis it has not got.
    On this partition stage 2's attention rows run as a 2-pass BATCH loop, one head at a
    time, and nothing sets `tiles`.
    """
    lo, hi = part.lo, part.hi
    if lo >= hi:
        return Loop(1, 0, 0, 0, 't', '')
    # STRIDES ARE UNPADDED (pool=None).  They step over real data -- the slabs an earlier
    # part left and the ones the boundary will read -- so they must be the true tile or
    # head size.  The engine's row padding only says how far a kernel WALKS, which is a
    # sizing question covered in _stage_buffers; adding it here would put pass k+1 a few
    # rows past its own operand.
    batch = max(_sh(st, k).get('batch', 1) for k in range(lo, hi))
    tiles = part.cut.tiles
    if tiles > 1 and batch > 1:
        raise ValueError(
            f'stage {st.idx + 1} has a part that is both tiled ({tiles}) and batched over '
            f'{batch} heads; the emitter writes one loop, not two nested ones.  Give the '
            f'part one head per lane (head_groups = the head count) or drop the tiling')
    if batch > 1:
        return Loop(batch, _fp(st, lo, None)[0], 1, _fp(st, hi - 1, None)[2], 'h',
                    f'{batch} heads on this lane, one at a time: each has its own K and V')
    if tiles > 1:
        return Loop(tiles, _fp(st, lo, None)[0], 0, _fp(st, hi - 1, None)[2], 't',
                    f'{tiles} passes over this lane\'s queries: the same work, but only '
                    f'one tile of the score matrix is live')
    return Loop(1, 0, 0, 0, 't', '')


def _y_name(row_no: int) -> str:
    """The Y region a row writes, or '' for one that just writes scratch -- see BIND_Y."""
    return BIND_Y.get(row_no, '')


def _max_row(st: Stage, pool: str, which: int, unbound_only: bool = False) -> int:
    """The largest X (which=0) or Y (which=2) any of this stage's rows walks, per pass.

    `unbound_only` counts only the rows that write nowhere named -- the scratch a stage
    needs on top of its placed regions.  A row whose output IS a named region is not
    scratch, and counting it as well would declare the score matrix twice.

    Sizing X (which=0) skips any row whose A OPERAND LIVES IN ANOTHER BUFFER.  Stage 1 is
    almost entirely like that -- only the patch tensor arrives from outside -- and
    charging those operands to X_BUF would reserve megabytes there that nothing addresses.
    """
    # Read off BIND_X rather than Item.x_buf: this runs while the buffer layout is still
    # being built, before compute() has resolved the items' offsets.
    elsewhere = lambda it: _bind(BIND_X.get(it.row.excel_row), 'X')[0] != 'X'
    keep = [k for k, it in enumerate(st.items)
            if not (unbound_only and _y_name(it.row.excel_row))
            and not (which == 0 and elsewhere(it))]
    try:
        return max((_fp(st, k, pool)[which] for k in keep), default=0)
    except ValueError:
        return 0          # dim-less template: only --wait can get here, and it computes
                          # nothing, so the transfer endpoints are the whole layout


def _place(named: List[Tuple[str, int]], at: int = 0) -> List[Region]:
    """Lay regions out in order, each starting on a DMA line.

    Alignment is not tidiness: a descriptor whose runs start mid-line straddles two lines
    on every repetition and the leg runs at half rate, which the byte-based cost model
    cannot see.  See DMA_LINE_ELEMS.
    """
    out = []
    for name, elems in named:
        at = _line(at)
        out.append(Region(name, at, elems))
        at += elems
    return out


def _x_regions(stages: List[Stage], geom: Geometry, i: int) -> List[Region]:
    """X: what this stage takes in from OUTSIDE ITSELF, from offset 0 up.

    EVERY X REGION IS A LANDING ZONE for an inbound transfer -- an LPDDR read or the
    previous stage's hand-off -- so this table and the descriptor builders have to agree,
    and _check_handoff_sizing holds them to it.

      stage 1  PATCH        this lane's token slab of the model input, off the south edge
      stage 2  QKV_Q/K/V    this lane's head group's three slabs, from the 1 -> 2 boundary
      stage 3  RESIDUAL     this lane's token slab of row 9's output, read back off the
                            south edge two steps after stage 1 wrote it

    Q IS NOT THE SAME SHAPE AS K AND V, and that asymmetry is why splitting the queries is
    cheap: Q is [tokens / q_split, d_head] per head, while K and V stay [tokens, d_head]
    whole however the queries are divided.  Sizing Q per lane also keeps row 14's A operand
    at offset 0 of the region, so no lane-indexed offset appears in the emitted call.

    The attention output the 2 -> 3 boundary carries does NOT appear here: it lands in
    Y:FFN_RESID, because row 19's axpy computes Y += X and it is the Y side.
    """
    st  = stages[i]
    cut = st.parts[-1].cut
    if i == 0:
        return _place([('PATCH', (geom.tokens // cut.q_split) * geom.patch)])
    if i == 1:
        tq = geom.tokens // cut.q_split
        return _place([('QKV_Q', tq * geom.d_head * cut.hpl),
                       ('QKV_K', geom.tokens * geom.d_head * cut.hpl),
                       ('QKV_V', geom.tokens * geom.d_head * cut.hpl)])
    tt = geom.tokens // st.n_par
    return _place([('RESIDUAL', tt * geom.d_model)])


def _y_regions(stages: List[Stage], geom: Geometry, i: int, pool: str) -> List[Region]:
    """Y: every tensor a later row or a later phase of the same step still has to read.

    Nothing is overlaid: SCRATCH is sized from the rows that write no named region, and
    everything else is placed after it in the order BIND_Y writes it.

    EACH REGION IS SIZED BY THE ROW THAT WRITES IT AND BY WHAT LATER READS IT, not by a
    formula -- three things would otherwise be got wrong, and all three silently:

      * ROW PADDING ON THE WRITE.  A RedMulE GEMM splits M over its tiles, so the emitted
        M is rounded up and the kernel writes PAST the true row count.
      * ROW PADDING ON THE READ, which can be rounded further than the producer's: row 9's
        layernorm writes 1092 rows of RESIDUAL and row 10's projection walks 1104 of them.
      * THE BATCH.  A region only needs room for every head when the loop actually
        ADVANCES through it -- which the emitter does for a part's last row and no other.
    """
    st    = stages[i]
    named: List[Tuple[str, int]] = [('SCRATCH', _max_row(st, pool, 2, unbound_only=True))]
    # What later rows READ out of a named region, which can exceed what its producer wrote.
    reads: Dict[str, int] = {}
    for k, it in enumerate(st.items):
        xb, xr = _bind(BIND_X.get(it.row.excel_row), 'X')
        wb, wr = _bind(BIND_W.get(it.row.excel_row), 'W')
        if xb == 'Y':
            reads[xr] = max(reads.get(xr, 0), _fp(st, k, pool)[0])
        if wb == 'Y':
            reads[wr] = max(reads.get(wr, 0), _fp(st, k, pool)[1])
    for part in st.parts:
        loop = _part_loop(st, part, pool)
        for k in range(part.lo, part.hi):
            name = _y_name(st.items[k].row.excel_row)
            if not name:
                continue
            # What one pass writes, engine padding included ...
            elems = _fp(st, k, pool)[2]
            # ... plus the ground the loop covers, for the one row whose output walks.
            if loop.count > 1 and k == part.hi - 1:
                elems += (loop.count - 1) * loop.y
            named.append((name, elems))
    # SEVERAL ROWS MAY NAME ONE REGION -- row 5 writes EMB and row 6 adds the positional
    # table into it in place -- so the list is folded to one entry per name, keeping the
    # largest claim and the order of first mention.  Placing it twice would give the second
    # writer a region of its own and strand the first one's output.
    sizes: Dict[str, int] = {}
    for nm, elems in named:
        sizes[nm] = max(sizes.get(nm, 0), elems, reads.get(nm, 0))
    return _place(list(sizes.items()))


def _w_layout(stages: List[Stage], geom: Geometry, i: int, pool: str) -> List[int]:
    """Element offset of each item's operand B inside W_BUF, plus one past the end.

    The SUM of the stage's regions, not the largest of them: a stage re-reads its weights
    on every pipeline item, so they are all live at once.  A row whose B is an ACTIVATION
    is skipped -- it lives in X_BUF or Y_BUF (see BIND_W), and reserving W for it would
    declare it twice.

    Tiling does not shrink these: only the row count M divides by `tiles`, and operand B is
    (N, P).  Every tile of the score matrix contracts against the WHOLE K, which is
    exactly why tiling the queries works and tiling the keys would not.
    """
    st = stages[i]
    offs, at = [], 0
    for k, it in enumerate(st.items):
        if _bind(BIND_W.get(it.row.excel_row), 'W')[0] != 'W':
            offs.append(0)              # resolved against that buffer in compute(), not here
            continue
        at = _line(at)
        offs.append(at)
        at += _fp(st, k, pool)[1]
    offs.append(at)
    return offs


# ---------------------------------------------------------------------------
# LPDDR legs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One LPDDR transfer of one cluster.

    `node` picks which channel of the edge the cluster talks to:

      * 'column' -- the node beside the cluster, its own grid column.  The nearest one,
                    and the right choice for a tensor no other stage touches.
      * 'share'  -- the tensor is cut into `shares` PIECES, and piece j lives at node
                    j % nodes, slot j // nodes.  A cluster owns `shares / n_par`
                    consecutive pieces from lane * that.  The ONLY way one stage can read
                    back what another wrote, because the piece index is named by neither
                    stage's width.

    Distance is not charged either way; the difference is which node holds the bytes.

    `slots` is ITEM ROTATION, for a tensor written on one step and read on a LATER one:
    the leg keeps `slots` copies of every piece and uses copy (item % slots).  slots = 1
    means written and read inside one step, true of everything but the residual.  See
    _check_lpddr_slots for how deep it has to be.
    """
    name:     str
    edge:     str
    region:   str     # the X (inbound) or Y (outbound) region it lands in / reads from
    direction: str    # 'in' or 'out'
    node:     str = 'column'
    shares:   int = 1   # pieces the tensor is cut into; only meaningful for node='share'
    slots:    int = 1   # item-rotation depth

    def pieces(self, n_par: int) -> int:
        """Pieces ONE cluster of an `n_par`-wide stage moves on this leg."""
        return max(1, self.shares // n_par) if self.node == 'share' else 1


# Which stage moves what off-chip.  FOUR legs, and one of them is the price of this
# partition:
#
#   stage 1  PATCH     in   the model input, off the south edge.  Row 2 (patchify) rides it.
#   stage 1  RESIDUAL  out  row 9's output, spilled ...
#   stage 3  RESIDUAL  in   ... and read back here, two steps later, for row 19.
#   stage 3  OUTPUT    out  the channel estimate, north.  Row 33 (merge patches) rides it.
#
# THE RESIDUAL SPILLS BECAUSE IT SKIPS A STAGE: row 9 is on stage 1 and row 19 on stage 3,
# so no boundary can carry it.  The FFN residual is not like that: row 19 -> row 28 never
# leaves a lane.
#
# FOUR SLOTS, BECAUSE cwt NEEDS FOUR AND BOTH APPS READ THIS TABLE.  ctt would do with two
# -- its late write sits behind the barrier that ended the read's compute phase -- but cwt
# has ONE barrier a step, so the write CONCURRENT with the read needs a slot of its own.
# _slots_needed derives both numbers and _check_lpddr_slots holds the table to the deeper.
#
# TWO PIECES, ONE PER LANE, and 'share' is what makes the two ends NAME THE SAME BYTES:
# both stages are two lanes wide and cut the tokens the same way, so the lane with the
# matching piece index reads back what it wrote.  'column' could not do that -- it would
# send stage 3, on the NORTH row, to north nodes.  `shares` must divide both touching
# stages' widths (_check_lane_geometry enforces it) and is not a bandwidth knob.
#
# What is NOT charged is DISTANCE: stage 3 reads the south edge from the north row.
# Re-measure against RTL before treating that as free.  The other two legs take 'column'.
LPDDR_LEGS: Dict[int, List[Leg]] = {
    0: [Leg('PATCH',    LPDDR_IN_EDGE,  'PATCH',    'in'),
        Leg('RESIDUAL', LPDDR_IN_EDGE,  'RESIDUAL', 'out',
            node='share', shares=2, slots=4)],
    2: [Leg('RESIDUAL', LPDDR_IN_EDGE,  'RESIDUAL', 'in',
            node='share', shares=2, slots=4),
        Leg('OUTPUT',   LPDDR_OUT_EDGE, 'OUTPUT',   'out')],
}


def _lpddr_legs(i: int) -> List[Leg]:
    return LPDDR_LEGS.get(i, [])


def _leg_nodes(leg: Leg) -> int:
    """LPDDR channels this leg can actually spread over."""
    nodes = HBM_CHAN_PLACEMENT[leg.edge]
    return min(nodes, leg.shares) if leg.node == 'share' else nodes


def _leg_sharers(st: Stage, leg: Leg) -> int:
    """How many lanes of this stage share one LPDDR node, at worst.

    A stage with more lanes than the leg has nodes has to double up, and the lanes on one
    node divide its bandwidth -- so "per cluster" figures are quoted for the WORST-OFF
    lane, or they would not bound anything.  A 'share' leg may reach fewer nodes than the
    edge provides, because the tensor is only cut into so many pieces.
    """
    nodes = _leg_nodes(leg)
    return -(-st.n_par // nodes) if nodes else 1


def _leg_region(stages: List[Stage], geom: Geometry, i: int, leg: Leg,
                pool: str) -> Region:
    """The buffer region one leg lands in (inbound) or reads from (outbound)."""
    regs = (_x_regions(stages, geom, i) if leg.direction == 'in'
            else _y_regions(stages, geom, i, pool))
    return _region(regs, leg.region)


def _leg_piece_bytes(stages: List[Stage], geom: Geometry, i: int, leg: Leg,
                     pool: str) -> int:
    """Bytes of ONE PIECE of this leg's tensor -- ONE number for the whole leg.

    The piece must be the SAME bytes of the same node from both ends, or the reader walks
    a stride the writer never used -- and the two ends' regions do NOT agree on it:

      * A REGION MAY BE PADDED.  Stage 1's Y:RESIDUAL is sized by what READS it, and row
        10's GEMM walks 1104 of its 1092 rows, so it is 12 rows larger than the tensor.
      * A LANDING ZONE IS NOT.  Stage 3's X:RESIDUAL is exactly the slab it expects.

    So THE READER'S REGION IS AUTHORITATIVE where there is a reader, and the writer is held
    to delivering at least that much out of its own.  A leg no other stage reads back has
    one end and one answer, padding and all.
    """
    ends = [(j, lg) for j in range(len(stages))
            for lg in _lpddr_legs(j) if lg.name == leg.name]
    readers = [(j, lg) for j, lg in ends if lg.direction == 'in']
    j, lg = (readers or ends)[0]
    piece = (_leg_region(stages, geom, j, lg, pool).elems * BYTES_PER_ELEM
             // lg.pieces(stages[j].n_par))
    for k, other in ends:
        have = _leg_region(stages, geom, k, other, pool).elems * BYTES_PER_ELEM
        need = other.pieces(stages[k].n_par) * piece
        if have < need:
            raise ValueError(
                f'stage {k + 1} moves {other.pieces(stages[k].n_par)} piece(s) of the '
                f'{leg.name!r} LPDDR tensor at {piece} B each, but its {other.region} region '
                f'holds only {have} B; the two ends of a shared leg must agree on the piece')
    return piece


def _leg_bytes(stages: List[Stage], geom: Geometry, i: int, leg: Leg, pool: str) -> int:
    """Bytes ONE cluster of stage `i` moves on this leg."""
    return leg.pieces(stages[i].n_par) * _leg_piece_bytes(stages, geom, i, leg, pool)


def _lpddr_layout(stages: List[Stage], geom: Geometry, pool: str) -> Dict[str, int]:
    """Assign a per-NODE offset to every tensor that leaves or enters the chip.

    THE OFFSETS ARE PER NODE, and an edge has HBM_CHAN_PLACEMENT[edge] of them.  What a
    node has to hold is

        (pieces landing on it) x (slots) x (piece bytes)

    and the two multipliers are the two ways a tensor is replicated in LPDDR: PIECES, when
    a 'share' leg is cut into more pieces than the edge has nodes (or a 'column' leg sits
    on a stage wider than its edge), and SLOTS, item rotation for a tensor read on a later
    step -- the residual, and nothing else here.

    ONE OFFSET SPACE PER EDGE, not per direction: the south edge carries the model input
    IN and the residual OUT, so packing the two directions separately would overlay them.
    """
    align = lambda n: (n + 63) & ~63
    offs: Dict[str, int] = {}
    ends: Dict[str, int] = {}
    for i in range(len(stages)):
        for leg in _lpddr_legs(i):
            if leg.name in offs:
                continue      # the other end of a shared leg: same bytes, same offset
            nodes = _leg_nodes(leg)
            per_node = (-(-leg.shares // nodes) if leg.node == 'share'
                        else _leg_sharers(stages[i], leg))
            offs[leg.name] = ends.get(leg.edge, 0)
            ends[leg.edge] = align(
                offs[leg.name]
                + per_node * leg.slots * _leg_piece_bytes(stages, geom, i, leg, pool))
    return offs


# ---------------------------------------------------------------------------
# Per-lane shapes and footprints
# ---------------------------------------------------------------------------

def _row_shape(row: LayerRow, cut: Cut) -> dict:
    """Per-cluster, per-PASS kernel shape for one template row under one part's cut.

    A stage's lanes divide up to three axes, and they are not interchangeable:

      * `groups` cuts the LEADING axis, the attention heads.  What is left of it on one
        lane stays a BATCH: several heads, each with its own operand B, run one after
        another rather than as one tall GEMM.
      * `q_split` lanes and `tiles` passes cut the ROW axis -- the queries, or the tokens
        on a part with no head axis at all.
    """
    k = row.kernel.lower()
    ind, outd = row.in_dims, row.out_dims
    if not ind:
        raise ValueError(
            f"row {row.excel_row} ({row.kernel} '{row.op}') has no i_* dims; the template "
            f"needs the tensor-shape columns for real-kernel mode (use --wait to fall "
            f"back to timing-only mempool_wait).")

    lead  = _prod(ind[:-2]) if len(ind) > 2 else 1
    batch = max(1, lead // max(1, cut.groups))
    div   = max(1, cut.q_split * cut.tiles)
    # The row axis is the second-from-last dim: queries on the attention rows, tokens
    # everywhere else.  It is what the lanes and the tiles divide; the leading axis is the
    # batch and is divided by `groups` alone.
    rows = max(1, (ind[-2] if len(ind) > 1 else 1) // div)
    if k in GEMM_KERNELS:
        # One GEMM per BATCH ELEMENT: whatever leading axis survives `groups` is issued as
        # separate calls, because each element has its own operand B -- head h contracts
        # against K_h.  The batch does not change the MAC count, so the latency model never
        # sees it.
        return {'cls': 'gemm', 'M': rows, 'N': ind[-1], 'P': outd[-1], 'batch': batch}
    if k in ('softmax', 'layernorm'):
        # The _2x4_f16vec kernels walk two ROWS and four COLUMNS at a time through a 32-bit
        # `*(v2h *)&A[i * N + j]`, so N must be a multiple of 4: of 2 or the odd rows land
        # on a 2-byte boundary and every v2h load on them is misaligned, and of 4 or the j
        # loop runs past the row end.
        return {'cls': k, 'M': _pad(rows, 2), 'N': _pad(ind[-1], 4), 'N_true': ind[-1],
                'batch': batch}
    if k in ('gelu', 'relu'):
        return {'cls': k, 'size': _pad(max(1, _prod(ind) // (div * batch)), 8),
                'batch': batch}
    if k == 'add':
        return {'cls': 'add', 'len': _pad(max(1, _prod(ind) // (div * batch)), 8),
                'batch': batch}
    if k in LAYOUT_KERNELS:
        # A wait, not a kernel: no permutation kernel exists in software/kernels/baremetal.
        # BATCH IS FORCED TO 1 whatever the leading axis says -- the spin stands for one
        # pass over the row's whole destination, and a row like the head merge reads every
        # head at once, so it must not sit inside a per-head loop.
        return {'cls': 'wait', 'batch': 1,
                'why': 'layout op: PE index permutation'}
    raise ValueError(f"row {row.excel_row} ('{row.op}'): no kernel is mapped to "
                     f"'{row.kernel}'.")


def _gemm_rows(M: int, pool: Optional[str]) -> int:
    """The M a GEMM is actually issued with on this pool (RedMulE splits M over its tiles)."""
    tiles = POOL_REDMULE_TILES.get(pool) if pool else None
    return M if not tiles else _pad(M, tiles)


def _row_footprint(row: LayerRow, cut: Cut,
                   pool: Optional[str] = None) -> Tuple[int, int, int]:
    """(X, W, Y) element counts one cluster needs for one row, per pass.

    The BATCH multiplies W and nothing else: every batch element's operand B has to be
    resident (a lane cannot re-fetch K between heads), while only one element's A and Y
    are live at a time.  The caller adds the batch stride where an element's output has to
    survive the loop (see _part_loop).
    """
    s = _row_shape(row, cut)
    c = s['cls']
    if c == 'gemm':
        M = _gemm_rows(s['M'], pool)
        w = s['N'] * s['P'] * s.get('batch', 1)
        # A positional-encoding table is indexed by token, so it splits with the tokens;
        # every other operand B is a contraction matrix and is replicated on every lane.
        if row.excel_row in SPLIT_WEIGHT_ROWS:
            w = max(1, int(w // cut.lanes_eff))
        return M * s['N'], w, M * s['P']
    if c in ('softmax', 'layernorm'):
        return s['M'] * s['N'], 0, s['M'] * s['N']
    if c in ('gelu', 'relu'):
        return 0, 0, s['size']
    if c == 'add':
        # Y += X.  Row 6 adds the positional-encoding table, a WEIGHT indexed by token: the
        # one add whose addend is not an activation, so it is charged to W.
        if row.excel_row in SPLIT_WEIGHT_ROWS:
            return 0, s['len'], s['len']
        return s['len'], 0, s['len']
    if c == 'wait':
        # A layout row is a VIEW, not a copy: it permutes indices into a tensor that is
        # already resident and produces no second one, so it claims no buffer space.  The
        # ONE exception -- row 13, which really does write K transposed -- is handled in
        # _fp, where the row's BIND_Y entry says so.
        return 0, 0, 0
    return 0, 0, 0


def _stage_buffers(st: Stage, geom: Geometry, stages: List[Stage],
                   pool: str) -> Tuple[int, int, int]:
    """(X, W, Y) elements one cluster of this stage declares.

    X spans its inbound landing zones AND the largest A operand any of its rows walks that
    is not somewhere else -- the max of the two, not their sum: a row reads what the
    inbound transfer delivered, so the two claims are the same bytes.  Y spans everything
    _y_regions has placed.  W is the SUM of the stage's own operand-B regions.
    """
    # A loop's last pass starts (count-1) strides up the buffer, so the operands that walk
    # have to fit at that offset and not just at the base.  Taken per part, like the loops.
    nx = max(_end(_x_regions(stages, geom, st.idx)), _max_row(st, pool, 0))
    for part in st.parts:
        if part.lo >= part.hi or _bind(
                BIND_X.get(part.items[0].row.excel_row), 'X')[0] != 'X':
            continue          # its A operand walks another buffer
        loop = _part_loop(st, part, pool)
        nx = max(nx, (loop.count - 1) * loop.x + _fp(st, part.lo, pool)[0])
    # Y is exactly its placed regions: SCRATCH already covers every row that writes nowhere
    # named, and a row that writes a region is bounded by that region.
    ny = _end(_y_regions(stages, geom, st.idx, pool))
    nw = _w_layout(stages, geom, st.idx, pool)[-1]
    rnd = lambda n: ((n + 31) // 32) * 32
    return rnd(nx), rnd(nw), rnd(ny)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def _transfer_ms(nbytes: float, bw_gbs: Optional[float]) -> float:
    """Transfer time [ms]: bw GB/s == bw*1e9 B/s, so ms = bytes/(bw*1e6)."""
    return 0.0 if not bw_gbs else nbytes / (bw_gbs * 1e6)


def _lpddr_bw(cfg: dict) -> Optional[float]:
    """The bandwidth ONE cluster gets to LPDDR.

    bandwidth_lpddr_gbs is PER CHANNEL and a transfer is striped across lpddr_channels of
    them, so the DRAM side is worth their product -- CAPPED BY bandwidth_c2c_gbs, because
    the bytes still cross the on-chip network to reach the memory-side node, over the same
    link a cluster-to-cluster hand-off uses.
    """
    per = cfg.get('bandwidth_lpddr_gbs')
    if not per:
        return None
    total = per * cfg.get('lpddr_channels', 1)
    c2c   = cfg.get('bandwidth_c2c_gbs')
    return min(total, c2c) if c2c else total


def _row_latency(row: LayerRow, rate_gemm: float, rate_pe: float) -> float:
    """Latency [ms] of one row's FULL work, before the lanes divide it."""
    return row.flops / (rate_gemm if row.is_gemm else rate_pe)


def _is_layout(row: LayerRow) -> bool:
    return row.kernel.lower() in LAYOUT_KERNELS


def _layout_latency(row: LayerRow, lanes: float, rate_pe: float,
                    cycles_per_ms: float) -> float:
    """Modelled latency [ms] of ONE cluster's share of a zero-FLOP layout row.

    The same fit celere_pipeline.py uses: a permutation costs one pass over what it
    WRITES, at PE_LAYOUT_FLOPS_PER_ELEM per element, plus a fixed per-invocation entry
    cost.  The destination count comes from the template's Data_out column rather than
    from out_dims, because a row may carry several tensors at once and only the byte
    column knows how many.  `lanes` is the part's lanes_eff.  Rows folded into this one
    are NOT added: one pass over the same addresses, so one wait covers both.
    """
    elems = row.dout_bytes / BYTES_PER_ELEM / max(1e-9, lanes)
    return (PE_LAYOUT_FIXED_CYCLES / cycles_per_ms
            + elems * PE_LAYOUT_FLOPS_PER_ELEM / rate_pe)


def compute(layers: List[LayerRow], cfg: dict) -> Results:
    """Cost the model and the partition."""
    pe_util = cfg['PE_utilization']
    te_util = cfg['TE_utilization']
    pools   = list(cfg['PE_peak'].keys())
    res     = Results()

    stale = [k for k in ('pipeline_clusters', 'attention_clusters',
                         'extra_pipeline_stages') if k in cfg]
    if stale:
        raise ValueError(
            f'config key(s) {", ".join(stale)} used to size a pipeline this script no '
            f'longer builds: the partition is not optimized any more, it is the STAGES '
            f'table in cevit_pipeline.py ({len(STAGES)} stages of '
            f'{", ".join(str(s["n_par"]) for s in STAGES)} clusters).  Remove them from '
            f'the config; change STAGES instead.')

    geom = read_geometry(layers)
    res.geom = geom

    # FLOPs/ms.  Each peak is derated by the utilization of the engine that actually runs
    # it: PE_utilization on the PE path, TE_utilization on a GEMM only where a tensor
    # engine exists.  A pool without one runs its GEMMs as PE matmuls, so TE_utilization
    # does not enter its numbers at all.
    for p in pools:
        gemm_util = te_util if p in TENSOR_ENGINE_POOLS else pe_util
        res.flops_per_ms[p] = {
            'non_GEMM': cfg['PE_peak'][p]   * pe_util   * 1e9,
            'GEMM':     cfg['GEMM_peak'][p] * gemm_util * 1e9,
        }

    for p in pools:
        rate_g  = res.flops_per_ms[p]['GEMM']
        rate_ng = res.flops_per_ms[p]['non_GEMM']
        res.row_latencies[p] = {r.excel_row: _row_latency(r, rate_g, rate_ng)
                                for r in layers}

    # Per-block totals: the model's own structure, independent of the partition.
    for p in pools:
        res.block_latencies[p] = []
        total   = 0.0
        current = None
        for r in layers:
            if current is None or current['name'] != r.block:
                current = {'name': r.block, 'n_rep': r.n_rep, 'sum_rows': 0.0,
                           'subtotal': 0.0}
                res.block_latencies[p].append(current)
            current['sum_rows'] += res.row_latencies[p][r.excel_row]
            current['subtotal']  = current['sum_rows'] * current['n_rep']
        for b in res.block_latencies[p]:
            total += b['subtotal']
        res.total_latency[p] = total

    bw = Bandwidths(c2c=cfg.get('bandwidth_c2c_gbs'), lpddr=_lpddr_bw(cfg))
    # The layout fit is quoted partly in CYCLES (a fixed per-invocation entry cost), so the
    # cost model needs the clock the emitter already reads from the config.
    cycles_per_ms = cfg.get('frequency', 1_000_000_000) / 1e3

    for p in pools:
        stages  = build_stages(layers, geom)
        rate_ng = res.flops_per_ms[p]['non_GEMM']
        for st in stages:
            items = st.items
            for k, it in enumerate(items):
                # A row is divided over the lanes of ITS PART of the stage, which need not
                # be all of them and need not be a whole number: the projection is spread
                # 4.8 ways and the front end 1 way.  See Cut.lanes_eff.
                lanes = st.cut_for(k).lanes_eff
                if _is_layout(it.row):
                    # A layout row carries 0 FLOPs, so the template can say nothing about
                    # what it costs; the PE fit does.  See _layout_latency.
                    it.ms = _layout_latency(it.row, lanes, rate_ng, cycles_per_ms)
                    continue
                # The FLOPs column carries the row's whole work; a lane does its share.
                it.ms = res.row_latencies[p][it.row.excel_row] / lanes
            st.latency  = sum(it.ms for it in items)
            st.layout   = {r.name: r.off for r in _x_regions(stages, geom, st.idx)}
            st.y_layout = {r.name: r.off
                           for r in _y_regions(stages, geom, st.idx, p)}
            # Where each row finds its operands.  A is at the buffer base unless BIND_X
            # names a region; B has a region of its own in W unless BIND_W says it is an
            # activation living elsewhere -- in X, where the boundary delivered it, or in Y,
            # where an earlier row of the same stage left it.
            w_off = _w_layout(stages, geom, st.idx, p)
            where = {'X': st.layout, 'Y': st.y_layout}
            for k, it in enumerate(st.items):
                rn = it.row.excel_row
                it.w_off = w_off[k]
                wb, wr = _bind(BIND_W.get(rn), 'W')
                if wb != 'W':
                    it.w_buf, it.w_off, it.w_reg = wb, where[wb].get(wr, 0), wr
                xb, xr = _bind(BIND_X.get(rn), 'X')
                if xr:
                    it.x_buf, it.x_off, it.x_reg = xb, where[xb].get(xr, 0), xr
                it.y_off = st.y_layout.get(_y_name(rn), 0)

            # Hand-off: this cluster's own descriptors are its outbound bytes, taken at the
            # WORST-OFF lane.  Both boundaries are uniform here, so every lane comes back
            # the same, but a figure quoted "per cluster" has to bound the slowest one or
            # it bounds nothing the moment a cut stops dividing evenly.
            st.dout_c2c = max((sum(x.nbytes for x in _handoff(stages, geom, st.idx, q))
                               for q in range(st.n_par)), default=0)
            st.dout_ms  = _transfer_ms(st.dout_c2c, bw.c2c)

            st.din_ms = st.lout_ms = 0.0
            for leg in _lpddr_legs(st.idx):
                # A stage wider than its edge shares nodes, and the lanes on one node
                # divide it, so the whole leg is billed at bw / sharers.  "Per cluster" is
                # quoted for the WORST-OFF lane.  See _leg_sharers.
                share = bw.lpddr and bw.lpddr / _leg_sharers(st, leg)
                ms = _transfer_ms(_leg_bytes(stages, geom, st.idx, leg, p), share)
                if leg.direction == 'in':
                    st.din_ms += ms
                else:
                    st.lout_ms += ms
        for i, st in enumerate(stages):
            # What ONE consumer takes in: every producer's descriptors aimed at its lane 0.
            # Summed over the boundary this is exactly what the producers send -- K and V
            # included, and they are delivered twice.
            if i:
                st.din_c2c = sum(x.nbytes
                                 for lane in range(stages[i - 1].n_par)
                                 for x in _handoff(stages, geom, i - 1, lane)
                                 if x.dst_lane == 0)
            st.x_elems, st.w_elems, st.y_elems = _stage_buffers(st, geom, stages, p)
        res.stages[p]                  = stages
        res.best_throughput[p]         = _makespan(stages, False)
        res.best_throughput_overlap[p] = _makespan(stages, True)

    return res


def _overlap_wall(st: Stage) -> float:
    """cwt wall [ms] of one stage.

    THE INBOUND LPDDR READ DOES NOT OVERLAP ANYTHING, and the runtime is what decides
    that.  A cwt step reads ITEM i'S OWN input -- not a prefetch of i+1 -- so the kernels
    cannot start until it lands, and the only wait the DMA shim offers is
    flex_dma_async_wait_all(), which drains every outstanding descriptor at once.  So the
    step has to be ordered inbound -> wait -> issue the drain -> compute: the read is alone
    in flight at that first wait, and it is the OUTBOUND traffic, issued after it, that
    runs beside the cores.
    """
    return st.din_ms + max(st.latency, st.transfer_ms)


def _makespan(stages: List[Stage], overlap: bool) -> float:
    """Steady-state period of one pipeline step = what throughput is 1/of.

    Both schedules rendezvous every cluster on a global barrier, so a step costs the
    slowest thing running inside each barriered phase:

      * ctt: compute -> barrier -> transfer -> barrier, TWO SUBSTAGES, so the step is
        max_s(compute_s) + max_s(transfer_s).  That is >= the naive
        max_s(compute_s + transfer_s) whenever the slowest compute and the slowest
        transfer sit on different stages: the barrier stops them covering for each other.
      * cwt: one barrier per step, so the step is the slowest stage's own wall.
    """
    if not stages:
        return 0.0
    if overlap:
        return max(_overlap_wall(st) for st in stages)
    return (max(st.compute_ms  for st in stages)
            + max(st.transfer_ms for st in stages))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    return f'{n / 1024:,.1f} KiB' if n >= 1024 else f'{n:,.0f} B'


def _print_stage_table(pool: str, stages: List[Stage], geom: Geometry,
                       *, overlap: bool, period: float) -> None:
    """Print the per-stage table for one pool under one wall model."""
    label = ('Overlapped  (cwt: compute-while-transfer)' if overlap
             else 'Serialized  (ctt: compute-then-transfer)')
    model = ('data_in + max(compute, transfer)' if overlap else 'compute + transfer')
    per   = ('max_s(wall)   /* the slowest stage */' if overlap
             else 'max_s(compute) + max_s(transfer)   /* two barriered substages */')
    bases, occupied = _stage_bases()
    n_work = sum(st.n_par for st in stages)
    idle   = occupied - n_work
    print(f'\nPool: {pool}  --  {label}')
    print(f'  >>> Pipeline period = {period:.6g} ms')
    print(f'  {len(stages)} stages  |  {n_work} working clusters on a {GRID_X}x{GRID_Y} grid'
          + (f' ({idle} of the {occupied} cids they span sit idle)' if idle else '')
          + f'  |  stage wall = {model}')
    print(f'  {"":11}period = {per}')
    for st, base in zip(stages, bases):
        wall = _overlap_wall(st) if overlap else st.compute_ms + st.transfer_ms
        print('  ' + '=' * 78)
        where = (f'cid {base}' if st.n_par == 1
                 else f'cids {base}-{base + st.n_par - 1}')
        # compute = THE WHOLE COMPUTE HALF OF A STEP, the inbound LPDDR read included: both
        # loops read item i's own input and hold the worker cores until it lands, so it is
        # in front of the kernels and not beside them, under cwt as much as under ctt.
        rows = f' ({st.latency:.6g} kernels + {st.din_ms:.6g} in)' if st.din_ms else ''
        print(f'  Stage {st.idx + 1} [{where}]  |  '
              f'compute = {st.compute_ms:.6g} ms{rows}  |  '
              f'transfer = {st.transfer_ms:.6g} ms  |  '
              f'wall = {wall:.6g} ms   (per cluster)')
        lanes = '1 lane' if st.n_par == 1 else f'{st.n_par} lanes || '
        print(f'  {"":11}>>> {lanes}, {st.split}')
        print(f'  {"":4}{"Block":<16}{"Row":<6}{"Op":<26}{"":6}{"per lane [ms]":>16}')
        for part in st.parts:
            cut = part.cut
            # How this part divides its rows, printed per part because a stage may cut
            # different runs of its rows differently.
            tiles = f', x{cut.tiles} tiles' if cut.tiles > 1 else ''
            heads = f', {cut.groups} head groups' if cut.groups > 1 else ''
            print('  ' + '-' * 78)
            print(f'  {"":4}work / lane = 1/{cut.lanes_eff:g} of the row{heads}{tiles}')
            for it in part.items:
                tag = (' WAIT' if _is_layout(it.row) else
                       ' GEMM' if it.row.is_gemm else '     ')
                print(f'  {"":4}{it.row.block[:16]:<16}{it.row.excel_row:<6}'
                      f'{it.row.op[:26]:<26}{tag}  {it.ms:>16.6g}')
        print('  ' + '-' * 78)
        if st.din_c2c:
            print(f'  {"":4}hand-off in : {_fmt_bytes(st.din_c2c):>12}  per cluster')
        if st.dout_c2c:
            print(f'  {"":4}hand-off out: {_fmt_bytes(st.dout_c2c):>12}  per cluster'
                  f'  --> stage {st.idx + 2}   ({st.dout_ms:.6g} ms)')
        for leg in _lpddr_legs(st.idx):
            nb      = _leg_bytes(stages, geom, st.idx, leg, pool)
            arrow   = '<--' if leg.direction == 'in' else '-->'
            nodes   = HBM_CHAN_PLACEMENT[leg.edge]
            sharers = _leg_sharers(st, leg)
            share = ('' if sharers == 1 else
                     f', {st.n_par} lanes over {nodes} nodes so the worst-off sees '
                     f'1/{sharers} of one')
            # A 'column' leg takes the node beside the cluster and never reaches.
            # Otherwise the node is the piece index modulo the node count, which may be
            # sideways of the cluster's own column -- and the model bills bytes and
            # nothing for the hops, so say so.
            hops = ({0} if leg.node == 'column' else
                    {(base + k) % GRID_X - k % nodes for k in range(st.n_par)})
            reach = ('' if hops == {0} else
                     f'  [reaches {max(abs(h) for h in hops)} column(s) sideways, '
                     f'not charged]')
            print(f'  {"":4}LPDDR {leg.direction:<3}  : {_fmt_bytes(nb):>12}  per cluster'
                  f'  {arrow} {leg.edge} node {leg.name}{share}{reach}')
        print(f'  {"":4}L1 per cluster: X {_fmt_bytes(st.x_elems * BYTES_PER_ELEM)}'
              f'  + W {_fmt_bytes(st.w_elems * BYTES_PER_ELEM)}'
              f'  + Y {_fmt_bytes(st.y_elems * BYTES_PER_ELEM)}')
    print('  ' + '=' * 78)


def print_results(layers: List[LayerRow], res: Results,
                  mode: Optional[str] = None) -> None:
    pools = list(res.flops_per_ms.keys())
    geom  = res.geom

    print('#' * 78)
    print('FLOPs/ms  (achievable FLOPs per millisecond)')
    print('#' * 78)
    print(f'{"":<14}{"non-GEMM":>20}{"GEMM":>20}')
    for pool, vals in res.flops_per_ms.items():
        print(f'{pool:<14}{vals["non_GEMM"]:>20,.0f}{vals["GEMM"]:>20,.0f}')

    print()
    print('#' * 78)
    print(f'Model geometry: {geom.tokens} tokens x {geom.patch} (patch), d_model '
          f'{geom.d_model}, {geom.heads} heads x {geom.d_head}')
    print('#' * 78)

    print()
    print('#' * 78)
    print('Latency [ms] per pool  (whole model, before the lanes divide it)')
    print('#' * 78)
    header = f'{"Row":<5}{"Block":<16}{"Op":<20}{"":<5}' + ''.join(f'{p:>16}' for p in pools)
    print(header)
    print('-' * len(header))
    block = ''
    for r in layers:
        block = r.block or block
        # A layout row costs nothing HERE whatever it costs on a stage: this table is the
        # model's own structure, and the PE layout fit is per lane, so its figure only
        # exists once the partition does.  Two outcomes, and the tag says which:
        #   DMA   -- a transfer carries it (DROPPED_ROWS)
        #   WAIT  -- it runs as a mempool_wait; see the per-stage table below
        if r.excel_row in DROPPED_ROWS:
            tag = 'DMA '
        elif _is_layout(r):
            tag = 'WAIT'
        else:
            tag = 'GEMM' if r.is_gemm else '   .'
        vals = ''.join(f'{res.row_latencies[p][r.excel_row]:>16.6g}'
                       if tag not in ('DMA ', 'WAIT') else f'{"--":>16}'
                       for p in pools)
        print(f'{r.excel_row:<5}{block:<16}{r.op[:20]:<20}{tag:<5}{vals}')
    print('-' * len(header))
    totals = ''.join(f'{res.total_latency[p]:>16.6g}' for p in pools)
    print(f'{"":<5}{"":<16}{"Tot min latency":<25}{totals}')
    print(f'{"":<5}(DMA rides a transfer; WAIT is a mempool_wait sized per lane -- its')
    print(f'{"":<5} cost is in the stage table below)')

    if res.stages:
        print()
        print('#' * 78)
        print('Pipeline stages')
        print('#' * 78)
        for p in pools:
            if mode in (None, 'ctt'):
                _print_stage_table(p, res.stages[p], geom, overlap=False,
                                   period=res.best_throughput[p])
            if mode in (None, 'cwt'):
                _print_stage_table(p, res.stages[p], geom, overlap=True,
                                   period=res.best_throughput_overlap[p])


# ---------------------------------------------------------------------------
# C emission -- shared blocks
# ---------------------------------------------------------------------------

# Kernel headers pulled in only in real-kernel mode (single TU, so the non-static kernel
# definitions cause no multiple-definition trouble).  builtins_v2.h first, because
# mempool_relu_f16.h uses v2h without including it itself.
_KERNEL_INCLUDES = [
    'builtins_v2.h',
    'baremetal/mempool_redmule_f16.h',
    'baremetal/mempool_matmul_f16.h',
    'baremetal/mempool_softmax_f16.h',
    'baremetal/mempool_layernorm_f16.h',
    'baremetal/mempool_gelu_f16.h',
    'baremetal/mempool_relu_f16.h',
    'baremetal/mempool_axpy_f16.h',
]


def _c_comment(text: str) -> str:
    return text.replace('*/', '* /')


def _render_includes(wait: bool) -> List[str]:
    L = ['#include "flex_runtime.h"', '#include "flex_dma_pattern.h"']
    if not wait:
        L += [f'#include "{h}"' for h in _KERNEL_INCLUDES]
    return L


def _render_dma_helper(bench: bool, blocking: bool = True) -> str:
    """Shared C: the 2D-DMA helper every transfer in the app goes through.

    BLOCKING OR NOT IS A PROFILING DECISION, not a schedule one.  ctt is always blocking --
    its transfer phase is barriered anyway.  cwt only benefits from a non-blocking issue
    when there is a steady state to hide the transfer in, so a single-iteration run and any
    --bench run stay blocking: with the DMA waited on where it is issued, an RTL trace
    attributes cycles to that transfer.
    """
    head = [
        '/* Blocking DMA: issues the whole transfer as a single flex_dma_async_2d and waits',
        ' * for it to land -- `reps` runs of `chunk` bytes, walking the source by src_stride',
        ' * and the destination by dst_stride.',
        ' *',
        ' * BOTH BOUNDARIES STRIDE, and that is where the model\'s layout rows live: the',
        ' * 1 -> 2 hand-off strides its SOURCE to pick one head out of the projection (rows',
        ' * 11 and 12), and the 2 -> 3 hand-off strides its DESTINATION to merge the heads',
        ' * back (rows 17 and 18).  Every run is a whole 64 B DMA line either way. */',
        'static inline void dma_2d_strided(uint64_t dst, uint64_t src, uint32_t chunk,',
        '                                  uint32_t dst_stride, uint32_t src_stride,',
        '                                  uint32_t reps)',
        '{',
    ]
    if bench:
        head.append('    mempool_start_benchmark();')
    head += ['    flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);']
    if blocking:
        head.append('    flex_dma_async_wait_all();')
    if bench:
        head.append('    mempool_stop_benchmark();')
    head += [
        '}',
        '',
        '/* A contiguous transfer is the degenerate descriptor -- one repetition of `bytes`,',
        ' * with both strides set to `bytes` so a further repetition would abut it. */',
        'static inline void dma_2d(uint64_t dst, uint64_t src, uint32_t bytes)',
        '{',
        '    dma_2d_strided(dst, src, bytes, bytes, bytes, 1);',
        '}',
    ]
    if not blocking:
        head += [
            '',
            '/* The transfers above only ISSUE.  One wait drains every descriptor a cluster',
            ' * put in flight this step, and it sits just before the step barrier -- so the',
            ' * DMA has run alongside the whole compute rather than after it. */',
            'static inline void dma_wait(void) { flex_dma_async_wait_all(); }',
        ]
    return '\n'.join(head)


def _render_stage_tables(stages: List[Stage], geom: Geometry,
                         lpddr: Dict[str, int], pool: str,
                         mode: str = 'ctt') -> List[str]:
    """The cid maps, the hand-off descriptor table and the LPDDR leg table."""
    bases, n_phys = _stage_bases()
    n_st   = len(stages)
    n_work = sum(st.n_par for st in stages)

    L: List[str] = []
    A = L.append
    A(f'#define N_STAGES    {n_st}')
    A(f'#define N_CLUSTERS  {n_phys}  /* cids the pipeline OCCUPIES; {n_work} of them work */')
    A('')
    A('/* The pipeline is placed on the grid, not merely packed onto cids: stage 1 sits on')
    A(f' * the SOUTH edge row and stage {n_st} on the NORTH one, because those are the rows')
    A(' * with LPDDR nodes beside them.  A grid of another shape moves both edges, so it is')
    A(' * a compile error rather than a silent long-haul transfer. */')
    A(f'_Static_assert(ARCH_NUM_CLUSTER_X == {GRID_X},')
    A(f'               "this pipeline is laid out for a {GRID_X}-wide grid");')
    A(f'_Static_assert(ARCH_NUM_CLUSTER_Y == {GRID_Y},')
    A(f'               "this pipeline is laid out for a {GRID_Y}-tall grid");')
    A('_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER,')
    A('               "pipeline needs more clusters than this arch provides");')
    A('')
    A('/* Stages are numbered from 1, matching the console tables and the split plots.')
    A(' * Every per-stage table below is indexed by that number, so each carries an unused')
    A(' * row 0 -- cheaper to read than a -1 at every lookup. */')
    A(f'enum {{ {", ".join(f"STAGE_{i + 1}" + (" = 1" if i == 0 else "") for i in range(n_st))} }};')
    A('')
    # Stage 0 marks an IDLE cid.  A padded stage leaves cids unused so the next one starts
    # on a column boundary, and the tables are indexed BY CID, so they carry an entry for
    # those too -- main() reads CID_STAGE[cid] before it knows whether anything runs there.
    cid_stage: List[str] = []
    cid_lane:  List[str] = []
    for i, spec in enumerate(STAGES):
        cid_stage += [str(i + 1)] * spec['n_par'] + ['0'] * spec.get('pad', 0)
        cid_lane  += [str(k) for k in range(spec['n_par'])] + ['0'] * spec.get('pad', 0)
    A('/* cid -> (stage, lane), with stage 0 for a cid the pipeline leaves IDLE.  The LANE')
    A(' * is what an LPDDR leg resolves its (node, share) from -- not the cluster\'s column,')
    A(' * which is a different number as soon as a stage does not start on a column boundary')
    A(' * or is wider than the edge.  Naming the share by lane is what lets one stage read')
    A(' * back what another wrote: both resolve lane k the same way wherever they sit, and')
    A(' * the column only decides how far the bytes go. */')
    A(f'static const uint8_t CID_STAGE[N_CLUSTERS] = {{ {", ".join(cid_stage)} }};')
    A(f'static const uint8_t CID_LANE [N_CLUSTERS] = {{ {", ".join(cid_lane)} }};')
    # STAGE_BASE is the only one of these the transfer code needs: push_attn addresses its
    # consumers as STAGE_BASE[STAGE_2] + lane, and every lane COUNT it needs is already a
    # #define above (S1_LANES, S2_LANES).  A STAGE_LANES table beside them would be a second
    # statement of the same fact, and the compiler flags it as dead.
    A(f'static const uint8_t STAGE_BASE [N_STAGES + 2] = '
      f'{{ 0, {", ".join(str(b) for b in bases)}, {n_phys} }};')
    A('')

    # ---- the geometry the two hand-offs are written in terms of ----
    A('/* ---- Hand-off geometry -----------------------------------------------------')
    A(' *')
    A(' * A boundary is a LOOP over these, not a table of descriptors: nothing below is a')
    A(' * literal, so a change of width in the generator\'s STAGES table is a change of one')
    A(' * number here and no edit at all to the code that reads them.')
    A(' *')
    A(' * BETWEEN THEM THE TWO BOUNDARIES PERFORM FIVE OF THE MODEL\'S LAYOUT ROWS, and the')
    A(' * two LPDDR legs perform the other two, so none is left to run on the PE path.')
    A(' * Rows 11 and 12 are the 1 -> 2 boundary picking a head\'s column stripe out of the')
    A(' * projection; rows 17 and 18 are the 2 -> 3 boundary\'s DESTINATION STRIDE -- one')
    A(' * DH_B run per token at column h*DH_B of a DM_B-wide row IS the transpose back to')
    A(' * token-major and the reshape that folds the head axis into the feature one.  Row')
    A(' * 13 (transpose K) is ASSUMED onto the 1 -> 2 boundary and is the one optimism')
    A(' * here: the K descriptor below is a plain contiguous slab and does NOT transpose.')
    A(' * The GEMM shapes are unaffected, so the cycle counts are right; only a numerical')
    A(' * result would be wrong, and this app checks none. */')
    for title, defines in _handoff_consts(stages, geom):
        A(f'/* {title} */')
        for name, val, note in defines:
            A(f'#define {name:<14} {val}u{"  /* " + note + " */" if note else ""}')
        A('')

    # ---- the cwt parity constants ----
    if mode == 'cwt':
        # ELEMENTS, not bytes: these are added to __fp16 pointers in the compute functions
        # and to the byte bases in the transfer helpers, so both forms are emitted -- one
        # scaled table for the DMA side, the raw deltas for the kernels.
        dx = [0] + [_cwt_deltas(stages, geom, pool, i)[0] for i in range(len(stages))] + [0]
        dy = [0] + [_cwt_deltas(stages, geom, pool, i)[1] for i in range(len(stages))] + [0]
        A('/* ---- Double-buffer parity ---------------------------------------------------')
        A(' *')
        A(' * How far a stage\'s SECOND copy is, in bytes, indexed by the 1-based stage')
        A(' * number.  Only what a transfer touches is doubled -- the inbound landing zone in')
        A(' * X, and in Y whatever an LPDDR leg or the hand-off reads -- so these are a good')
        A(' * deal smaller than the buffers themselves.  A stage that doubles nothing in a')
        A(' * buffer carries 0 and its multiply folds away.')
        A(' *')
        A(' * The compute functions take the same deltas in ELEMENTS, baked into the offsets')
        A(' * they already carry; only the DMA side needs them at run time. */')
        A(f'static const uint32_t XPAR[N_STAGES + 2] = '
          f'{{ {", ".join(str(v * BYTES_PER_ELEM) + "u" for v in dx)} }};')
        A(f'static const uint32_t YPAR[N_STAGES + 2] = '
          f'{{ {", ".join(str(v * BYTES_PER_ELEM) + "u" for v in dy)} }};')
        # The two a producer needs for its CONSUMER, which it cannot read out of the tables
        # above without knowing the consumer's stage number -- so they are named directly.
        A(f'#define S2_XPAR      {_cwt_deltas(stages, geom, pool, 1)[0] * BYTES_PER_ELEM}u'
          f'  /* where stage 2 keeps its second inbound copy */')
        A(f'#define S3_YPAR      {_cwt_deltas(stages, geom, pool, 2)[1] * BYTES_PER_ELEM}u'
          f'  /* and stage 3 its second FFN_RESID */')
        A('')

    # ---- LPDDR table ----
    A('/* ---- LPDDR legs ------------------------------------------------------------')
    A(' *')
    A(' * An edge carries `nodes` LPDDR channels, one per column of the grid, and `bycol`')
    A(' * says how a cluster picks one:')
    A(' *')
    A(' *   bycol = 1  the node beside it -- its own grid COLUMN.  The nearest one, and the')
    A(' *              right choice for a tensor no other stage touches.')
    A(' *   bycol = 0  the tensor is cut into PIECES and this cluster owns `pieces` of them,')
    A(' *              starting at lane * pieces.  Piece j lives at node j % nodes, slot')
    A(' *              j / nodes.  The piece index is named by neither stage\'s width, which')
    A(' *              is the ONLY way one stage can read back what another wrote: the')
    A(' *              residual\'s pieces are resolved the same way by the stage that writes')
    A(' *              them and the stage that reads them back, wherever either one sits.')
    A(' *')
    A(' * `slots` is ITEM ROTATION.  A tensor written on one step and read on a LATER one')
    A(' * has to survive every write in between, so the leg keeps `slots` copies of each')
    A(' * piece and uses copy (item % slots).  slots = 1 means written and read inside one')
    A(' * step, which is every leg but the residual: that one is written by stage 1 in step')
    A(' * m and read by stage 3 two stages later.  The count is the DEEPER of the two')
    A(' * schedules, because one table serves both apps. */')
    A('typedef struct {')
    A('    uint8_t  edge;        /* 0 = south, 1 = north */')
    A('    uint8_t  buf;         /* 0 = X_BUF (inbound), 1 = Y_BUF (outbound) */')
    A('    uint8_t  nodes;       /* LPDDR channels this leg spreads over */')
    A('    uint8_t  bycol;       /* 1 = nearest node (own column), 0 = by piece index */')
    A('    uint8_t  pieces;      /* pieces THIS cluster moves */')
    A('    uint8_t  slots;       /* item-rotation depth; 1 = no rotation */')
    A('    uint32_t lpddr_off;   /* bytes into the node, before piece and slot */')
    A('    uint32_t l1_off;      /* bytes into that buffer */')
    A('    uint32_t bytes;       /* of ONE piece */')
    A('} lpddr_t;')
    A('')
    for key, direction in (('LPDDR_IN', 'in'), ('LPDDR_OUT', 'out')):
        entries: List[str] = []
        firsts: List[int] = [0]
        for i, st in enumerate(stages):
            legs = [lg for lg in _lpddr_legs(i) if lg.direction == direction]
            for lg in legs:
                reg     = _leg_region(stages, geom, i, lg, pool)
                sharers = _leg_sharers(st, lg)
                pieces  = lg.pieces(st.n_par)
                where = (f'nearest {lg.edge} node (own column)' if lg.node == 'column'
                         else f'{lg.edge} nodes by piece, {pieces} of {lg.shares}'
                              + (f', {sharers} lanes per node' if sharers > 1 else '')
                              + (f', {lg.slots} item slots' if lg.slots > 1 else ''))
                entries.append(
                    f'    {{ {0 if lg.edge == "south" else 1}, '
                    f'{0 if direction == "in" else 1}, '
                    f'{_leg_nodes(lg)}, '
                    f'{1 if lg.node == "column" else 0}, '
                    f'{pieces}, {lg.slots}, '
                    f'{lpddr[lg.name]:8d}, {reg.off * BYTES_PER_ELEM:8d}, '
                    f'{_leg_piece_bytes(stages, geom, i, lg, pool):8d} }},'
                    f'  /* stage {i + 1}: {lg.name}, {where} */')
            firsts.append(firsts[-1] + len(legs))
        A(f'static const lpddr_t {key}[{max(1, len(entries))}] = {{')
        L.extend(entries if entries else ['    { 0, 0, 1, 0, 1, 1, 0, 0, 0 },'])
        A('};')
        # Indexed by the 1-BASED stage number, like every other per-stage table, so entry 0
        # is the unused one and entry N_STAGES+1 is the end marker the last stage reads.
        A(f'static const uint8_t {key}_FIRST[N_STAGES + 2] = '
          f'{{ 0, {", ".join(str(v) for v in firsts)} }};')
        A('')
    return L


def _render_push_helper(cwt: bool = False) -> str:
    """Shared C: the two hand-offs, one per boundary.

    Everything here is a loop over the constants above; nothing is a descriptor table and
    nothing is a literal, so a change of width is a change of one number in STAGES.  Both
    are written as an INTERSECTION of the producer's range with the consumer's -- heads on
    the first boundary, tokens on the second -- so neither assumes the two sides are the
    same width, and a slab that straddles two consumers simply issues two descriptors.
    """
    return '\n'.join([
        '/* The boundaries cut the SAME axis on both sides at possibly different widths, so',
        ' * what one cluster owes another is where their two ranges OVERLAP.  Writing it that',
        ' * way is what makes each loop independent of which side is wider. */',
        'static inline uint32_t lo_of(uint32_t a, uint32_t b) { return a > b ? a : b; }',
        'static inline uint32_t hi_of(uint32_t a, uint32_t b) { return a < b ? a : b; }',
        '',
        '/* BOUNDARY 1 -> 2: the projection output, cut by HEAD and by TOKEN at once.',
        ' *',
        ' * THIS IS ROWS 11 AND 12 (split q,k,v, split heads), and the SOURCE ADDRESSING is',
        ' * what performs them.  Stage 1 issues row 10 as a single dense over its own token',
        ' * slab, so a lane holds [TT1, 3*d_model] row-major and head h of stripe s is',
        ' * column s*DM_B + h*DH_B of every row -- a DH_B RUN at a QKV_ROW_B STRIDE, one per',
        ' * token.  Choosing that column is the q|k|v split and the head selection at once,',
        ' * and it is free: DH_B is exactly one 64 B DMA line.  THE DESTINATION IS',
        ' * CONTIGUOUS -- a consumer wants each of its heads\' Q, K and V as plain',
        ' * [rows, d_head] tensors, which is what its X regions are.',
        ' *',
        ' * Q AND K|V DO NOT GO TO THE SAME PLACES, and that is the whole shape of this',
        ' * boundary.  A stage-2 lane owns head group k / S2_QSPLIT and query slab',
        ' * k % S2_QSPLIT.  It scores only ITS OWN queries, so Q is intersected with that',
        ' * slab and each third of it crosses once; but every one of those queries scores',
        ' * against EVERY key, so K and V are whole TOKENS-row slabs and go to all',
        ' * S2_QSPLIT lanes of the head group -- REPLICATED, deliberately, and charged for.',
        ' * A producer holds only TT1 of the tokens either way, so it writes its own rows of',
        ' * a K slab several producers fill between them.  THIRTY-TWO DESCRIPTORS per',
        ' * producer at 2 x 6 lanes.  Both ranges are intersected rather than assumed to',
        ' * line up, so nothing here knows either stage\'s width.',
        ' *',
        ' * ROW 13 (transpose K) IS ASSUMED ONTO THIS TRANSFER AND IS NOT PERFORMED.  Row 14',
        ' * contracts against K^T; what arrives is K.  A descriptor that really transposed',
        ' * it would walk 2-byte runs against a 64 B line, so it is not written.  Nothing',
        ' * about the CYCLES depends on it -- row 14\'s GEMM is issued at the same M, N and',
        ' * P -- and this app checks no result. */',
        'static inline void push_qkv(uint32_t lane, uint32_t x_base, uint64_t src'
        + (', uint32_t alt)' if cwt else ')'),
        '{',
        '    uint32_t base = STAGE_BASE[STAGE_2];',
    ] + ([] if not cwt else [
        '    /* The consumer computes this item one step from now, on the copy whose parity',
        '     * matches it -- the same `alt` the producer reads its own source from, because',
        '     * both refer to the item in flight rather than the one being computed. */',
        '    x_base += alt * S2_XPAR;',
    ]) + [
        '    /* This producer: the first head it holds, and which token slab it holds. */',
        '    uint32_t hp = (lane / S1_QSPLIT) * S1_HPL, qp = lane % S1_QSPLIT;',
        '    /* Where each stripe lands on the consumer; its SOURCE column is s * DM_B. */',
        '    static const uint32_t DST[3] = { QKV_Q_X, QKV_K_X, QKV_V_X };',
        '',
        '    for (uint32_t k = 0; k < S2_LANES; k++) {',
        '        uint32_t hk = (k / S2_QSPLIT) * S2_HPL, qk = k % S2_QSPLIT;',
        '        uint32_t h0 = lo_of(hp, hk), h1 = hi_of(hp + S1_HPL, hk + S2_HPL);',
        '        for (uint32_t s = 0; s < 3u; s++) {',
        '            uint32_t q  = (s == 0u);   /* stripe 0 is Q; 1 and 2 are K and V */',
        '            uint32_t lo = q ? lo_of(qp * TT1, qk * TQ) : qp * TT1;',
        '            uint32_t hi = q ? hi_of(qp * TT1 + TT1, qk * TQ + TQ) : qp * TT1 + TT1;',
        '            if (lo >= hi)',
        '                continue;',
        '            uint32_t slab = q ? QKV_Q_SLAB : QKV_KV_SLAB;',
        '            uint32_t dst  = lo - (q ? qk * TQ : 0u);',
        '            for (uint32_t h = h0; h < h1; h++)',
        '                dma_2d_strided((uint64_t)remote_cid(base + k,',
        '                                   x_base + DST[s] + (h - hk) * slab + dst * DH_B),',
        '                               src + (uint64_t)((lo - qp * TT1) * QKV_ROW_B',
        '                                   + QKV_Y + s * DM_B + h * DH_B),',
        '                               DH_B, DH_B, QKV_ROW_B, hi - lo);',
        '        }',
        '    }',
        '}',
        '',
        '/* BOUNDARY 2 -> 3: A*V, transposed back to token-major and head-merged on the way.',
        ' *',
        ' * THIS IS ROWS 17 AND 18, and the destination stride is where they live: writing',
        ' * one DH_B run per token at column h*DH_B of a DM_B-wide row IS the transpose back',
        ' * to token-major and the reshape that folds the head axis into the feature axis.',
        ' * Every run is DH_B -- one whole 64 B DMA line.',
        ' *',
        ' * A FAN, NOT A PAIRING: a stage-2 lane holds S2_HPL heads of a TQ-row query slab',
        ' * and a stage-3 lane owns a TT-token slab of every head, so a producer owes a',
        ' * descriptor to each consumer its rows reach -- and its rows may straddle two of',
        ' * them, which is why the ranges are intersected rather than divided.  The batch',
        ' * loop leaves a lane\'s heads back to back in ATTNH, HEAD_SRC bytes apart.',
        ' *',
        ' * Each source overlaps only its own destinations, so A*V moves EXACTLY ONCE. */',
        'static inline void push_attn(uint32_t lane, uint32_t y_base, uint64_t src'
        + (', uint32_t alt)' if cwt else ')'),
        '{',
        '    uint32_t base = STAGE_BASE[STAGE_3];',
    ] + ([] if not cwt else [
        '    y_base += alt * S3_YPAR;',
    ]) + [
        '    uint32_t h = lane / S2_QSPLIT, q = lane % S2_QSPLIT;',
        '',
        '    for (uint32_t k = 0; k < S3_LANES; k++) {',
        '        uint32_t lo = lo_of(q * TQ, k * TT);',
        '        uint32_t hi = hi_of(q * TQ + TQ, k * TT + TT);',
        '        if (lo >= hi)',
        '            continue;',
        '        for (uint32_t e = 0; e < S2_HPL; e++)',
        '            dma_2d_strided((uint64_t)remote_cid(base + k,',
        '                               y_base + FFN_RESID_Y + (lo - k * TT) * DM_B',
        '                               + (h * S2_HPL + e) * DH_B),',
        '                           src + (uint64_t)(ATTNH_Y + e * HEAD_SRC',
        '                               + (lo - q * TQ) * DH_B),',
        '                           DH_B, DM_B, DH_B, hi - lo);',
        '    }',
        '}',
    ])


def _render_lpddr_helper(cwt: bool = False) -> str:
    """Shared C: the off-chip legs of one stage, in either direction.

    A leg moves `pieces` pieces, and where each one lives is derived from the PIECE INDEX
    rather than from the cluster's lane or column, so the two ends of a shared tensor agree
    without either knowing the other's width.  hbm_addr() is never used: west node 0 holds
    nothing on this placement.
    """
    return '\n'.join([
        'static inline uint64_t lpddr_addr(uint32_t edge, uint32_t node, uint32_t off)',
        '{',
        '    return edge ? hbm_north(node, off) : hbm_south(node, off);',
        '}',
        '',
        '/* `item` is the pipeline item this stage is working on, and it is what makes a',
        ' * spilled tensor safe: a leg with slots > 1 keeps that many copies of every piece',
        ' * and uses copy (item % slots), so item m\'s residual is still there when stage 3',
        ' * reads it two steps after stage 1 wrote it.  Legs with slots == 1 ignore it. */',
        'static inline void lpddr_move(uint32_t stage, uint32_t lane, uint32_t dir,',
        '                              uint32_t item, uint64_t x_base, uint64_t y_base'
        + (', uint32_t alt)' if cwt else ')'),
        '{',
        '    const lpddr_t *tab   = dir ? LPDDR_OUT : LPDDR_IN;',
        '    const uint8_t *first = dir ? LPDDR_OUT_FIRST : LPDDR_IN_FIRST;',
        '    uint32_t col = get_pos(flex_get_cluster_id()).x;',
    ] + ([] if not cwt else [
        '    /* Both directions touch the copy of the item IN FLIGHT, not the one being',
        '     * computed: an inbound read is filling next step\'s input and an outbound write',
        '     * is draining last step\'s output.  Same parity for both -- see the main loop. */',
        '    x_base += (uint64_t)alt * XPAR[stage];',
        '    y_base += (uint64_t)alt * YPAR[stage];',
    ]) + [
        '    for (uint32_t e = first[stage]; e < first[stage + 1u]; e++) {',
        '        const lpddr_t *t = &tab[e];',
        '        uint32_t slot = t->slots > 1u ? item % t->slots : 0u;',
        '        for (uint32_t p = 0; p < t->pieces; p++) {',
        '            /* bycol: the node beside this cluster, and one piece.  Otherwise the',
        '             * PIECE INDEX decides: this cluster owns pieces lane*pieces .. +pieces,',
        '             * piece j sits at node j % nodes and share j / nodes inside it.  The',
        '             * modulo is what keeps a tensor cut into more pieces than the edge has',
        '             * channels from naming a node past the end of the map. */',
        '            uint32_t j     = lane * t->pieces + p;',
        '            uint32_t node  = t->bycol ? col : j % t->nodes;',
        '            uint32_t share = t->bycol ? 0u  : j / t->nodes;',
        '            uint64_t mem   = lpddr_addr(t->edge, node,',
        '                                        t->lpddr_off',
        '                                        + (share * t->slots + slot) * t->bytes);',
        '            uint64_t l1    = (t->buf ? y_base : x_base) + t->l1_off',
        '                             + (uint64_t)p * t->bytes;',
        '            if (dir) dma_2d(mem, l1, t->bytes);',
        '            else     dma_2d(l1, mem, t->bytes);',
        '        }',
        '    }',
        '}',
    ])


# ---------------------------------------------------------------------------
# C emission -- per-layer
# ---------------------------------------------------------------------------

def _render_layer_call(item: Item, pool: str, cut: Cut, bench: bool = False,
                       x_tile: int = 0, y_tile: int = 0, w_tile: int = 0,
                       var: str = 't', cycles_per_ms: float = 1e6,
                       loop_count: int = 1, par: str = '',
                       shift: Optional[Dict[str, int]] = None) -> List[str]:
    """C statement(s) for one row, dispatched by kernel and pool.

    GEMM runs on the RedMulE tensor engines on a pool in TENSOR_ENGINE_POOLS and on the PE
    matmul kernel elsewhere -- the same fact the cost model derates with.  Every other row
    runs on the PE path.  Each path ends aligned, so the next layer reads a completed
    output.

    x_tile / w_tile / y_tile advance an operand by one pass of the enclosing loop (see
    Loop): the row reading the part's inbound tensor takes x_tile, the row writing its
    outbound one takes y_tile, and on a BATCHED part every GEMM takes w_tile, because each
    head brings its own operand B.  Everything else is scratch the next pass overwrites.

    With bench=True the layer is wrapped in mempool_start/stop_benchmark(); the markers add
    no barrier, since the cores are already aligned on entry.
    """
    row = item.row
    s   = _row_shape(row, cut)
    c   = s['cls']
    tag = _c_comment(f'{row.op} [{row.kernel}]')
    # ONE BENCHMARK REGION PER KERNEL, not per template row: an RTL run should attribute
    # cycles to ONE kernel, so a row that lowers to several -- the projection's three GEMMs
    # -- gets one region each rather than one around all three.
    wrap = (lambda w: ['{', '    mempool_start_benchmark();']
            + ['    ' + x for x in w] + ['    mempool_stop_benchmark();', '}']) \
        if bench else (lambda w: list(w))
    # THE PARITY TERM, cwt only.  `par` is the item's low bit, and `shift` says how far the
    # second copy of a given operand is; both are 0 under ctt, where there is only one copy,
    # and the expression collapses to the plain offset.  See _cwt_deltas.
    off = lambda base, fixed, step, jump=0: (
        base + (f' + {fixed}u' if fixed else '') + (f' + {var} * {step}u' if step else '')
        + (f' + {par} * {jump}u' if par and jump else ''))
    buf = {'X': 'X_BUF', 'W': 'W_BUF', 'Y': 'Y_BUF'}
    # How far each operand's second copy is, cwt only.  Keyed by the REGION the operand is
    # bound to: an operand that just addresses the buffer base as scratch is not in the
    # dataflow, has no twin, and must not be shifted -- it can walk further than any one
    # region, so shifting it would run off the end of the buffer.
    sh      = shift or {}
    jump    = lambda reg: sh.get(reg, 0) if reg else 0
    xb  = off(buf[item.x_buf], item.x_off, x_tile, jump(item.x_reg))
    wb  = off(buf[item.w_buf], item.w_off, w_tile, jump(item.w_reg))
    yb  = off('Y_BUF', item.y_off, y_tile, jump(_y_name(row.excel_row)))
    work: List[str] = []

    def gemm(a: str, b: str, y: str, M: int, N: int, P: int, note: str) -> None:
        if pool in TENSOR_ENGINE_POOLS:
            # RedMulE partitions rows over the tensor engines.  Its internal mempool_barrier
            # syncs the cores but does NOT guarantee the HWPE write-back to Y_BUF has
            # drained before the next kernel reads it, so a trailing
            # flex_intra_cluster_sync() is appended.  GEMM (Y = X*W + Y) is the only op it
            # is driven with; a plain product is a GEMM onto a zeroed Y.
            work.append(f'redmule_synch_parallel({a}, {b}, {y}, '
                        f'{M}u, {N}u, {P}u, GEMM, false, 0);  /* {note} */')
        else:
            # No RedMulE here -> PE matmul, the path the cost model charges PE_utilization
            # for.  No internal barrier.
            work.append(f'matmul_4x2_parallel_f16vec({a}, {b}, {y}, '
                        f'{M}u, {N}u, {P}u, cid, nt);  /* {note} */')

    if c == 'gemm':
        M, N, P = _gemm_rows(s['M'], pool), s['N'], s['P']
        gemm(xb, wb, yb, M, N, P, tag)
        work.append('flex_intra_cluster_sync();')
    elif c == 'softmax':
        work.append(f'softmax_parallel_2x4_f16vec({xb}, {yb}, '
                    f'{s["M"]}u, {s["N"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'layernorm':
        work.append(f'layernorm_parallel_2x4_f16vec({xb}, {yb}, '
                    f'{s["M"]}u, {s["N"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c in ('gelu', 'relu'):
        work.append(f'{c}_f16({yb}, {s["size"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'add':
        # The real baremetal axpy kernel, called directly; it ends in mempool_log_barrier,
        # so it self-barriers.  0x3C00 == fp16 1.0, a normal multiplier (1u would be fp16
        # 0x0001, a subnormal, which forces the slow FP path on every vfmac).  Local-bank
        # vectorized unrolled axpy: no nPE arg, each core strides its own banks by
        # NUM_BANKS.
        from_w = row.excel_row in SPLIT_WEIGHT_ROWS
        src    = wb if from_w else xb
        work.append(f'axpy_f16vecp_local_unrolled4(0x3C00u, {src}, {yb}, '
                    f'{s["len"]}u);  /* {tag}: Y += {"W" if from_w else "X"} */')
    elif c == 'wait':
        # A timing stand-in: nothing in software/kernels/baremetal permutes a tensor, so
        # the spin stands for the addressing work and no bytes move.  The trailing sync
        # keeps the rendezvous the kernel it replaces would have ended on -- every core
        # spins the same count, so the barrier is nearly free.
        cyc = max(1, int(item.ms * cycles_per_ms / max(1, loop_count)))
        work.append(f'mempool_wait({cyc}UL);'
                    f'  /* {tag}: timing stand-in -- {_c_comment(s["why"])} */')
        work.append('flex_intra_cluster_sync();')
    else:
        raise ValueError(f"row {row.excel_row} ('{row.op}'): no kernel is mapped to "
                         f"'{row.kernel}'.")

    return wrap(work)


def _render_compute_stage_fns(stages: List[Stage], pool: str, wait: bool,
                              cycles_per_ms: float, bench: bool = False,
                              geom: Optional[Geometry] = None,
                              mode: str = 'ctt') -> List[str]:
    """The per-stage compute_stage_i() functions: mempool_wait (--wait) or real kernels.

    ONE FUNCTION PER STAGE, holding the stage's whole row list in order, and NO BARRIER
    INSIDE ANY OF THEM.  That is a property of the partition, not of the emitter: no stage
    exchanges anything with its own lanes, so the two global barriers a step needs are the
    wavefront's own, in main().

    A stage is emitted as its PARTS in order.  A part is not a rendezvous -- the rows run
    back to back -- it is a change of shape, and each part gets its own loop and its own
    kernel shapes.

    Every function takes `lane`, because the projection needs it: Q is the only operand on
    the stage whose address depends on which query slab this cluster owns.
    """
    L: List[str] = []
    if wait:
        L.append('/* ---- Per-cluster compute: one mempool_wait() per layer (timing-only). */')
    elif bench:
        L.append('/* ---- Per-cluster compute: real baremetal kernels, each layer bracketed')
        L.append('   by mempool_start/stop_benchmark() (the trace CSR). */')
    else:
        L.append('/* ---- Per-cluster compute: real baremetal kernels, uninstrumented. */')
    for st in stages:
        rows = ', '.join(str(it.row.excel_row) for it in st.items)
        L.append(f'/* stage {st.idx + 1}: rows {rows}, {st.n_par} lane(s) '
                 f'over {_c_comment(st.split)} */')
        L.append(f'static inline void compute_stage_{st.idx + 1}(uint32_t lane'
                 + (', uint32_t par)' if mode == 'cwt' else ')'))
        L.append('{')
        # No kernel addresses `lane`: every operand a stage reads is either at the base of
        # its buffer or at a fixed region offset, because the boundary lands each lane's own
        # slab at the start of the region it goes to.  The parameter stays in the signature
        # so main() can call every stage the same way.
        L.append('    (void)lane;')
        # THE PARITY OFFSETS, cwt only.  `shift` maps a named region to how far its second
        # copy is, so only the operands a transfer actually touches move on an odd item and
        # everything else -- the score matrix, SCRATCH, the weights -- stays where it is.
        par, shift = '', {}
        if mode == 'cwt' and geom is not None:
            dx, dy = _cwt_deltas(stages, geom, pool, st.idx)
            _, _, _, doubled = _cwt_doubled(stages, geom, pool, st.idx)
            shift = {r.name: dx for r in _x_regions(stages, geom, st.idx)}
            shift.update({nm: dy for nm in doubled})
            par = 'par'
            if wait or not any(shift.values()):
                L.append('    (void)par;')
        if not wait:
            L.append('    uint32_t cid = mempool_get_core_id();    /* in-cluster core id */')
            L.append('    uint32_t nt  = mempool_get_core_count();  /* cores per cluster */')
            L.append('    (void)cid; (void)nt;   /* unused on a RedMulE-only stage */')
        for part in st.parts:
            lo, hi = part.lo, part.hi
            if lo >= hi:
                continue
            cut  = part.cut
            loop = _part_loop(st, part, pool)
            ind  = '    '
            if len(st.parts) > 1:
                L.append(f'    /* rows '
                         f'{", ".join(str(it.row.excel_row) for it in part.items)}: '
                         f'divided over all {st.n_par} lanes. */')
            if loop.count > 1:
                L.append(f'    /* {_c_comment(loop.note)}. */')
                L.append(f'    for (uint32_t {loop.var} = 0; {loop.var} < {loop.count}u; '
                         f'{loop.var}++) {{')
                ind = '        '
            if wait:
                for k in range(lo, hi):
                    it  = st.items[k]
                    cyc = max(1, int(it.ms * cycles_per_ms / loop.count))
                    tag = ' (GEMM)' if it.row.is_gemm else ''
                    L.append(f'{ind}mempool_wait({cyc}UL);  '
                             f'/* row {it.row.excel_row}: {_c_comment(it.row.op)}{tag} */')
            else:
                for k in range(lo, hi):
                    it = st.items[k]
                    # Which pointers walk with the loop: the inbound operand on the part's
                    # first row, the outbound one on its last, and operand B on every GEMM
                    # of a batched part, because each head brings its own.  See Loop.
                    w_step = (_fp(st, k, None)[1] // loop.count
                              if loop.w and it.row.is_gemm else 0)
                    for line in _render_layer_call(it, pool, cut, bench,
                                                   loop.x if k == lo else 0,
                                                   loop.y if k == hi - 1 else 0,
                                                   w_step, loop.var,
                                                   cycles_per_ms, loop.count,
                                                   par, shift):
                        L.append(f'{ind}{line}')
            if loop.count > 1:
                L.append('    }')
        L.append('}')
        L.append('')
    return L


def _render_scratch_buffers(stages: List[Stage], geom: Geometry, pool: str,
                            mode: str = 'ctt') -> List[str]:
    """The three per-cluster L1 buffers, one per role.

    Sized by _buf_elems, which is also what _l1_footprint reports -- so the number printed on
    the console is the number the .l1 section holds, for both modes.
    """
    nx, nw, ny = _buf_elems(stages, geom, pool, mode)
    attr = '__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")))'
    attr_w = ('__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), '
              'section(".l1"), used))')
    return [
        '/* THE BUFFERS.  Per-cluster L1, placeholder data -- only cycles are measured.',
        ' *',
        ' * X_BUF is where a transfer from OUTSIDE the cluster lands -- the LPDDR read on',
        ' * stages 1 and 3, the projection stripes the boundary carries on stage 2 -- and',
        ' * W_BUF holds operand B where that is a weight.  Y_BUF is every kernel\'s output',
        ' * and the source of every outbound transfer.',
        ' *',
        ' * Each is sized to the largest claim any stage makes on it: for X the inbound',
        ' * regions plus the biggest operand, for W the SUM of the operand-B regions (a',
        ' * stage re-reads all its weights on every item), and for Y the placed regions,',
        ' * each big enough for what writes it AND for what later reads it (a RedMulE GEMM',
        ' * rounds M up, so it can walk further than its producer wrote). */',
    ] + ([] if mode != 'cwt' else [
        '/*',
        ' * UNDER cwt EACH BUFFER ALSO HOLDS A SECOND COPY of the part a transfer touches,',
        ' * and only that part: the inbound landing zone in X, and in Y whatever an LPDDR',
        ' * leg or the hand-off reads.  Item m\'s bytes are still moving while item m+1',
        ' * computes, so those two ends need one each; the working set does not, because it',
        ' * is dead the moment the step ends.  The twins are placed after everything the',
        ' * stage claims, so reaching one is a single per-stage delta times the item\'s low',
        ' * bit -- see the XPAR/YPAR constants and the `par` argument every compute function',
        ' * takes.',
        ' *',
        ' * That is why these are only a little larger than the ctt ones and not double. */',
    ]) + [
        f'#define X_ELEMS  {nx}U',
        f'#define W_ELEMS  {nw}U',
        f'#define Y_ELEMS  {ny}U',
        f'static __fp16 X_BUF[X_ELEMS] {attr};',
        f'static __fp16 W_BUF[W_ELEMS] {attr_w};',
        f'static __fp16 Y_BUF[Y_ELEMS] {attr};',
    ]


def _render_buffer_init() -> List[str]:
    """Prime the buffers with fp16 1.0 before the timed region.

    Uninitialised .l1 is garbage (NaN/Inf/subnormal), and core-side fp16 ops on such
    operands take a data-dependent, far slower path.
    """
    return [
        '    /* Prime the buffers with fp16 1.0 before timing (avoids NaN/subnormal FP',
        '     * stalls).  All cores, outside the timed region. */',
        '    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS; '
        'i += mempool_get_core_count()) X_BUF[i] = (__fp16)1.0f;',
        '    for (uint32_t i = mempool_get_core_id(); i < Y_ELEMS; '
        'i += mempool_get_core_count()) Y_BUF[i] = (__fp16)1.0f;',
        '    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; '
        'i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;',
        '    flex_intra_cluster_sync();',
    ]


def _render_ctt_loop(n_st: int) -> List[str]:
    """The compute-then-transfer wavefront: two barriered phases a step."""
    L: List[str] = []
    A = L.append
    A('    /*')
    A('     * Streaming pipeline, compute-then-transfer.  `step` counts pipeline ticks:')
    A('     * stage s handles item (step - s) when 0 <= step - s < N_ITER, so the loop')
    A('     * covers FILL, STEADY STATE and DRAIN.  Data is placeholder -- only cycles are')
    A('     * measured.')
    A('     *')
    A('     * The wavefront is keyed on the STAGE, not the cluster: a stage\'s lanes all')
    A('     * work on the SAME item in the same step, on different slices of it.  They need')
    A('     * no barrier of their own -- the TWO per-step global barriers below rendezvous')
    A('     * them.')
    A('     *')
    A('     * compute -> barrier -> transfer -> barrier, so compute and the outbound')
    A('     * transfers do NOT overlap, which is also why one buffer set suffices.')
    A('     *')
    A('     *   LPDDR(south) --> S1 --> S2 --> S3 --> LPDDR(north)')
    A('     *                      \\                 ^')
    A('     *                       `--- LPDDR(south) \'   the attention residual')
    A('     *')
    A('     * THE ATTENTION RESIDUAL SKIPS A STAGE.  Row 9 is on stage 1 and row 19 on')
    A('     * stage 3, so it is produced in step m and read in step m+2 and no boundary can')
    A('     * carry it.  It goes off-chip on a rotating set of slots -- see LPDDR_IN\'s')
    A('     * `slots` column. */')
    A('    for (uint32_t step = 0; step < N_ITER + N_STAGES - 1; step++) {')
    A('        /* stage 0 is an IDLE cid -- one the pipeline occupies but does not use.  It')
    A('         * is excluded here explicitly rather than through the phase arithmetic,')
    A('         * where `stage - 1` underflows and only happens to fail the test below. */')
    A('        uint32_t active = (cid < N_CLUSTERS) && (stage != 0u) && (phase <= step)')
    A('                          && (step - phase < N_ITER);')
    A('        /* The item this stage is working on THIS step -- the wavefront\'s whole')
    A('         * content.  Only the rotating LPDDR legs read it; it is meaningless when')
    A('         * `active` is false, and nothing uses it there. */')
    A('        uint32_t item   = step - phase;')
    A('')
    A('        /* ---- Compute phase: the inbound LPDDR read, then all cores compute ----')
    A('         *')
    A('         * ONE CALL PER STAGE, holding that stage\'s whole row list and NO BARRIER OF')
    A('         * ITS OWN.  No stage exchanges anything with its own lanes -- stages 1 and 3')
    A('         * are row-wise over tokens, and a stage-2 lane owns whole heads of its own')
    A('         * queries and reduces over nothing another lane holds -- so the two')
    A('         * global barriers in this loop are the only rendezvous a step needs, and an')
    A('         * idle cid simply falls through to them. */')
    A('        if (active) {')
    A('            if (LPDDR_IN_FIRST[stage] != LPDDR_IN_FIRST[stage + 1u]) {')
    A('                /* This stage takes bytes off the chip this step: the worker cores')
    A('                 * must not start until the dm_core\'s load has landed. */')
    A('                if (flex_is_dm_core())')
    A('                    lpddr_move(stage, lane, 0u, item,')
    A('                               (uint64_t)(uintptr_t)X_BUF,')
    A('                               (uint64_t)(uintptr_t)Y_BUF);')
    A('                flex_intra_cluster_sync();')
    A('            }')
    A('            /* Every core computes, the dm_core included: it has finished the load and')
    A('             * the intra-cluster sync above, so it is free to join. */')
    A('            switch (stage) {')
    for i in range(n_st):
        A(f'                case STAGE_{i + 1}: compute_stage_{i + 1}(lane); break;')
    A('                default: break;')
    A('            }')
    A('        }')
    A('        flex_global_barrier_xy();   /* compute done before ANY transfer */')
    A('')
    A('        /* ---- Transfer phase: every active cluster ships its own output ----')
    A('         *')
    A('         * The LPDDR legs first (stage 1 spills the residual, stage 3 drains the')
    A('         * channel estimate), then the boundary this stage feeds.  A stage feeds at')
    A('         * most one, and the last stage feeds none. */')
    A('        if (flex_is_dm_core() && active) {')
    A('            lpddr_move(stage, lane, 1u, item, (uint64_t)(uintptr_t)X_BUF,')
    A('                       (uint64_t)(uintptr_t)Y_BUF);')
    A('            if (stage == STAGE_1)')
    A('                push_qkv (lane, x_off, (uint64_t)(uintptr_t)Y_BUF);')
    A('            else if (stage == STAGE_2)')
    A('                push_attn(lane, y_off, (uint64_t)(uintptr_t)Y_BUF);')
    A('        }')
    A('        flex_global_barrier_xy();   /* all transfers done before the next compute */')
    A('    }')
    return L


def _render_cwt_loop(n_st: int, blocking: bool) -> List[str]:
    """The compute-while-transfer wavefront: one barrier a step, transfers alongside compute.

    THE WAVEFRONT IS m + 2s, NOT m + s.  Stage s handles item m at step m + 2s, so a whole
    SPARE STEP sits between a producer finishing item m and its consumer starting on it,
    and that is the step the transfer runs in.  ctt instead gives the transfer a barriered
    phase of its own, which is why its period is compute PLUS transfer.

    So at one step a cluster computes item i (on the copy whose parity matches i, after
    reading item i's own input off LPDDR) and drains item i-1 in parity i^1.

    A STAGE READS ITS OWN ITEM'S INPUT, IT DOES NOT PREFETCH THE NEXT ONE, and the runtime
    forces that: flex_dma_async_wait_all() drains every outstanding descriptor at once, so
    the read has to be ALONE in flight when it is waited on.  That pins the order to

        issue inbound(i) -> wait -> issue drain(i-1) -> compute(i) -> wait -> barrier

    and makes the wall din + max(compute, drain); see _overlap_wall.
    """
    L: List[str] = []
    A = L.append
    A('    /*')
    A('     * Streaming pipeline, COMPUTE-WHILE-TRANSFER.  `step` counts pipeline ticks and')
    A('     * stage s handles item (step - 2s), so a spare step sits between a producer and')
    A('     * its consumer and the transfers run inside it, alongside compute.  One barrier')
    A('     * per step, not two.')
    A('     *')
    A('     * A cluster touches two items in one step: it READS item i\'s input and COMPUTES')
    A('     * item i, and it DRAINS item i-1\'s output alongside the kernels.  The drain uses')
    A('     * the parity opposite i, so `par` picks compute\'s buffers and `alt` the drain\'s,')
    A('     * and no descriptor ever races a kernel.  Only the transfer endpoints are')
    A('     * doubled -- see THE BUFFERS.')
    A('     *')
    A('     * THE READ IS ITEM i\'S OWN, NOT A PREFETCH OF i+1, and the order below follows')
    A('     * from that: the kernels need it, and flex_dma_async_wait_all() drains EVERY')
    A('     * outstanding descriptor, so the read has to be alone in flight when it is')
    A('     * waited on -- before the drain is issued, never after.')
    A('     *')
    A('     *   LPDDR(south) --> S1 --> S2 --> S3 --> LPDDR(north)')
    A('     *                      \\                 ^')
    A('     *                       `--- LPDDR(south) \'   the attention residual')
    A('     *')
    A('     * THE RESIDUAL ROTATES OVER SEVERAL SLOTS HERE, more than the ctt loop needs: it')
    A('     * is written two stages before it is read, and with ONE barrier a step there is')
    A('     * nothing ordering a later item\'s write against that read.  _slots_needed derives')
    A('     * the depth and the LPDDR tables above carry it. */')
    A(f'    for (uint32_t step = 0; step < N_ITER + 2u * N_STAGES - 1u; step++) {{')
    A('        /* `it` is signed on purpose: the drain runs one step AFTER this stage\'s last')
    A('         * compute, so `it - 1` has to stay meaningful at it = 0.  Taking the low bit')
    A('         * of `it` gives compute\'s parity, and the drain\'s item differs from it in')
    A('         * exactly that bit. */')
    A('        int32_t  it    = (int32_t)step - (int32_t)phase;')
    A('        uint32_t on    = (cid < N_CLUSTERS) && (stage != 0u);')
    A('        uint32_t comp  = on && it >= 0 && it < (int32_t)N_ITER;')
    A('        uint32_t drain = on && it >= 1 && it <= (int32_t)N_ITER;')
    A('        uint32_t par   = (uint32_t)it & 1u;   /* compute\'s copy */')
    A('        uint32_t alt   = par ^ 1u;            /* the draining copy */')
    A('')
    A('        /* ---- This item\'s input, and NOTHING else in flight while it lands ----')
    A('         *')
    A('         * The kernels read what this arrives, so the cores have to wait for it -- and')
    A('         * the only wait there is drains everything outstanding.  Issuing the drain')
    A('         * first would therefore serialize the drain too and cost exactly what this')
    A('         * schedule exists to save, so the read goes first, alone. */')
    A('        if (flex_is_dm_core() && comp) {')
    A('            lpddr_move(stage, lane, 0u, (uint32_t)it,')
    A('                       (uint64_t)(uintptr_t)X_BUF,')
    A('                       (uint64_t)(uintptr_t)Y_BUF, par);')
    if not blocking:
        A('            dma_wait();')
    A('        }')
    A('')
    A('        /* ---- Item it-1\'s output, issued to run ALONGSIDE the kernels ----')
    A('         *')
    A('         * The LPDDR leg, then the boundary this stage feeds.  Both read the copy the')
    A('         * compute below is not touching, so neither races it, and neither is waited')
    A('         * on until the far end of the step. */')
    A('        if (flex_is_dm_core() && drain) {')
    A('            lpddr_move(stage, lane, 1u, (uint32_t)(it - 1),')
    A('                       (uint64_t)(uintptr_t)X_BUF,')
    A('                       (uint64_t)(uintptr_t)Y_BUF, alt);')
    A('            if (stage == STAGE_1)')
    A('                push_qkv (lane, x_off, (uint64_t)(uintptr_t)Y_BUF, alt);')
    A('            else if (stage == STAGE_2)')
    A('                push_attn(lane, y_off, (uint64_t)(uintptr_t)Y_BUF, alt);')
    A('        }')
    A('        /* Holds the worker cores until the read above has landed and the drain\'s')
    A('         * descriptors exist.  A rendezvous, not a transfer. */')
    A('        flex_intra_cluster_sync();')
    A('')
    A('        /* ---- Compute, on the copy the transfers are not touching ---- */')
    A('        if (comp) {')
    A('            switch (stage) {')
    for i in range(n_st):
        A(f'                case STAGE_{i + 1}: compute_stage_{i + 1}(lane, par); break;')
    A('                default: break;')
    A('            }')
    A('        }')
    if not blocking:
        A('')
        A('        /* The drain has to land before the barrier: the next step overwrites the')
        A('         * copy it is reading, and the consumer computes on what it delivered. */')
        A('        if (flex_is_dm_core())')
        A('            dma_wait();')
    else:
        A('')
        A('        /* The transfers above already waited where they were issued -- see')
        A('         * dma_2d_strided.  Nothing is left in flight at the barrier. */')
    A('        flex_global_barrier_xy();   /* one per step, not two */')
    A('    }')
    return L

def _render_c(pool: str, stages: List[Stage], geom: Geometry,
              cycles_per_ms: float, n_iter: int,
              bench: bool = False, wait: bool = False, mode: str = 'ctt') -> str:
    """One app's main.c, for either schedule.

    EVERYTHING BUT main() IS SHARED.  The tables, the buffers, the transfer helpers and the
    per-stage compute functions differ between the two modes only in whether they carry a
    parity argument, so they are rendered once and the two wavefronts are two loop bodies --
    see _render_ctt_loop and _render_cwt_loop.
    """
    cwt = mode == 'cwt'
    # BLOCKING TRANSFERS WHENEVER THE POINT IS TO MEASURE THEM.  ctt is barriered anyway; cwt
    # only gains from a non-blocking issue when there is a steady state to hide it in, and a
    # single-iteration run has none -- it is all fill and drain.  --bench likewise: a DMA that
    # is waited on where it is issued is a DMA an RTL trace can attribute cycles to.
    blocking = not cwt or n_iter == 1 or bench
    n_st   = len(stages)
    n_work = sum(st.n_par for st in stages)
    _, n_phys = _stage_bases()
    lpddr  = _lpddr_layout(stages, geom, pool)

    lines: List[str] = []
    A = lines.append

    kind = 'timing-only (mempool_wait)' if wait else 'real baremetal kernels'
    kind += (' + mempool_start/stop_benchmark() around every layer and DMA' if bench
             else ' (uninstrumented)')
    sched = ('compute-while-transfer' if cwt else 'compute-then-transfer')
    A(f'/* Auto-generated by cevit_pipeline.py -- {sched} pipeline for {pool} */')
    A(f'/* {n_st} stages, {n_work} working clusters over the {n_phys} cids they occupy on a '
      f'{GRID_X}x{GRID_Y} grid; {n_iter} item(s) streamed as a wavefront. */')
    A(f'/* Compute: {kind}. */')
    A('/* Dataflow: the buffers are the DMA endpoints, so a layer\'s operands are the bytes')
    A(' * the previous stage actually sent.  The core-side fp16 kernels are data-dependent')
    A(' * on NaN/subnormal operands, so per-layer cycles may differ from a run on primed')
    A(' * data; RedMulE GEMMs are fixed-latency and unaffected. */')
    A('')
    lines.extend(_render_includes(wait))
    A('')
    if wait:
        A('/* CYCLES_PER_MS: nop-loop iterations that approximate 1 ms of wall time. */')
        A(f'#define CYCLES_PER_MS  {int(cycles_per_ms)}UL')
        A('')
    A(f'#define N_ITER  {n_iter}U  /* items streamed through the pipeline */')
    A('')
    A('\n'.join(_render_stage_tables(stages, geom, lpddr, pool, mode)))
    A(_render_dma_helper(bench, blocking))
    A('')
    A(_render_push_helper(cwt))
    A('')
    A(_render_lpddr_helper(cwt))
    A('')
    lines.extend(_render_scratch_buffers(stages, geom, pool, mode))
    A('')
    lines.extend(_render_compute_stage_fns(stages, pool, wait, cycles_per_ms, bench,
                                           geom, mode))
    A('int main()')
    A('{')
    A('    uint32_t cid     = flex_get_cluster_id();')
    A('    uint32_t eoc_val = 0;')
    A('    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : 0u;')
    A('    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;')
    A('    /* Stage numbers run 1..N_STAGES; the wavefront wants the 0-based position ...')
    if cwt:
        A('     * TIMES TWO: cwt puts stage s on item m at step m + 2s, leaving a spare step')
        A('     * between a producer and its consumer for the transfer to run in. */')
        A('    uint32_t phase   = 2u * (stage - 1u);')
    else:
        A('     * as is: ctt puts stage s on item m at step m + s. */')
        A('    uint32_t phase   = stage - 1u;')
    A('    /* The RECEIVER\'s buffer offsets -- identical in every cluster (same binary), so')
    A('     * a producer can address a peer\'s L1 without any exchange.  One boundary needs')
    A('     * each: the projection stripes land in the consumer\'s X, and A*V in its Y, where')
    A('     * row 19\'s axpy wants it. */')
    A('    const uint32_t x_off = (uint32_t)(uintptr_t)X_BUF;')
    A('    const uint32_t y_off = (uint32_t)(uintptr_t)Y_BUF;')
    A('')
    A('    flex_barrier_xy_init();')
    A('    flex_global_barrier_xy();')
    if not wait:
        lines.extend(_render_buffer_init())
    A('    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();')
    A('    flex_global_barrier_xy();')
    A('')
    A('    /**************************************/')
    A('    /*  Program Execution Region -- Start */')
    A('    /**************************************/')
    A('')
    lines.extend(_render_ctt_loop(n_st) if not cwt
                 else _render_cwt_loop(n_st, blocking))
    A('')
    A('    /**************************************/')
    A('    /*  Program Execution Region -- Stop  */')
    A('    /**************************************/')
    A('    if (flex_get_core_id() == 0 && cid == 0) flex_timer_end();')
    A('    flex_global_barrier_xy();')
    A('    flex_eoc(eoc_val);')
    A('    return 0;')
    A('}')

    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# App generation
# ---------------------------------------------------------------------------

# Apps root of the mempool multi-cluster build.  Its Makefile discovers apps with
# `find $(APPS_DIR) -name "main.c"` and derives the app name from the directory, so each
# generated app is a FLAT directory holding a single main.c.
_APPS_DIR = (Path(__file__).resolve().parent.parent.parent.parent
             / 'software' / 'apps' / 'multi_cluster')


def app_name(mode: str, pool: str) -> str:
    """Directory (and binary) name of the generated app: cevit_<mode>_<pool>."""
    return f'cevit_{mode}_{pool.replace(" ", "_")}'


# Per-pool L1 geometry.  NUM_BANKS = banking_factor * num_cores: 8 x 256 for tensorpool256,
# 4 x 1024 for terapool -- and both L1s come to 4 MiB.
#   stacks  -- the sequential region the linker puts BELOW .l1 (arch.ld.c)
#   runtime -- the runtime's own .l1 objects sharing the region with ours
#   align   -- our buffers' alignment attribute, NUM_BANKS * sizeof(int32_t)
def _l1_geometry(num_cores: int, banking: int, bank_size: int,
                 stack: int = 512) -> Dict[str, int]:
    banks = banking * num_cores
    return {
        'total':   banks * bank_size,
        'stacks':  num_cores * stack,
        'runtime': 2 * banks * 4 + 2 * banks + 64,
        'align':   banks * 4,
    }


L1_GEOMETRY = {
    'TensorPool': _l1_geometry(num_cores=256,  banking=8, bank_size=2048),
    'Terapool':   _l1_geometry(num_cores=1024, banking=4, bank_size=1024),
}


def _cwt_doubled(stages: List[Stage], geom: Geometry, pool: str,
                 i: int) -> Tuple[int, int, int, List[str]]:
    """What stage `i` has to keep a SECOND copy of under cwt.

    Returns (x_span, y_lo, y_hi, names): the inbound landing zone is elements [0, x_span)
    of X_BUF, and the outbound sources are elements [y_lo, y_hi) of Y_BUF.

    Item m's bytes are still moving while item m+1's compute has started, so the two ends
    a transfer touches need two copies each: the X regions a HAND-OFF lands in, and
    whatever an LPDDR leg or the hand-off READS out of Y.

    AN LPDDR LANDING ZONE IS NOT DOUBLED -- the cwt loop reads item i's own input at the
    top of step i and computes it in the same step, so one copy is only ever touched by
    one item.  NOTHING ELSE IS EITHER: a stage's working set is dead the moment the step
    ends, and doubling the whole of X and Y would charge stage 2 an extra 2.4 MiB for
    score matrices no transfer reads.

    THE DOUBLED SET MUST BE CONTIGUOUS, and that is checked rather than hoped for: the
    emitted C reaches the second copy by adding ONE per-stage delta (see _cwt_deltas).  The
    fix if it ever stops holding is to reorder BIND_Y so everything a transfer reads sits
    at the end.  PER STAGE, because a cluster belongs to exactly one.
    """
    st    = stages[i]
    # THE INBOUND SPAN IS WHAT A HAND-OFF WRITES, not the whole of X.  Taken from the
    # descriptors the previous stage actually issues, so a boundary that stops landing in X
    # stops doubling it without an edit here.
    xregs  = _x_regions(stages, geom, i)
    landed = {r.name for r in xregs
              for lane in range(stages[i - 1].n_par) if i
              for x in _handoff(stages, geom, i - 1, lane)
              if x.dst_buf == 'X'
              and r.off * BYTES_PER_ELEM <= x.dst_off
              < (r.off + r.elems) * BYTES_PER_ELEM}
    x_span = max((r.off + r.elems for r in xregs if r.name in landed), default=0)
    x_gap  = [r.name for r in xregs if r.name not in landed and r.off < x_span]
    if x_gap:
        raise ValueError(
            f'stage {i + 1}: the cwt double-buffered span [0, {x_span}) of X_BUF holds '
            f'{", ".join(sorted(landed))}, which the hand-off writes, but also '
            f'{", ".join(x_gap)}, which it does not.  The emitted C reaches the second copy '
            f'by one per-stage delta, so the doubled regions have to be adjacent and start '
            f'at 0; reorder _x_regions so the hand-off landing zones come first.')
    regs  = {r.name: r for r in _y_regions(stages, geom, i, pool)}
    names = {lg.region for lg in _lpddr_legs(i) if lg.direction == 'out'}
    for lane in range(st.n_par):
        for x in _handoff(stages, geom, i, lane):
            for nm, r in regs.items():
                off = r.off * BYTES_PER_ELEM
                if off <= x.src_off < off + r.elems * BYTES_PER_ELEM:
                    names.add(nm)
    live = [regs[nm] for nm in names if nm in regs]
    if not live:
        return x_span, 0, 0, []
    y_lo = min(r.off for r in live)
    y_hi = max(r.off + r.elems for r in live)
    covered = sum(r.elems for r in live)
    gap = [r.name for r in _y_regions(stages, geom, i, pool)
           if r.name not in names and y_lo <= r.off < y_hi]
    if gap:
        raise ValueError(
            f'stage {i + 1}: the cwt double-buffered span [{y_lo}, {y_hi}) of Y_BUF holds '
            f'{", ".join(sorted(names))}, which a transfer reads, but also '
            f'{", ".join(gap)}, which nothing does.  The emitted C reaches the second copy '
            f'by one per-stage delta, so the doubled regions have to be adjacent; reorder '
            f'BIND_Y so everything a transfer reads sits at the end of Y_BUF.')
    if covered != y_hi - y_lo:
        raise ValueError(
            f'stage {i + 1}: the cwt doubled Y regions cover {covered} of the '
            f'{y_hi - y_lo} elements they span; _place leaves DMA-line padding between '
            f'regions, which is fine, but a hole this large means the span is wrong')
    return x_span, y_lo, y_hi, sorted(names)


def _cwt_deltas(stages: List[Stage], geom: Geometry, pool: str,
                i: int) -> Tuple[int, int]:
    """(X, Y) ELEMENTS to add to an offset to reach stage `i`'s SECOND copy.

    ONE NUMBER PER BUFFER PER STAGE, which is the whole reason the doubled regions have to
    be contiguous.  X's second landing zone is placed straight after everything the stage
    claims in X, so a region at `o` has its twin at `o + x_elems`; Y's outbound span is
    placed after everything it claims in Y, so a region at `o` has its twin at
    `o - y_lo + y_elems`.

    The emitted C multiplies these by a PARITY BIT, so an undoubled buffer costs a multiply
    by zero the compiler folds away.  BOTH ARE ZERO WHEN THE STAGE DOUBLES NOTHING in that
    buffer: a stage whose X is purely an LPDDR landing zone gets 0, and returning x_elems
    there would send every read a whole buffer past its data.
    """
    st = stages[i]
    x_span, y_lo, y_hi, _ = _cwt_doubled(stages, geom, pool, i)
    return (st.x_elems if x_span else 0), (st.y_elems - y_lo if y_hi > y_lo else 0)


def _buf_elems(stages: List[Stage], geom: Geometry, pool: str,
               mode: str) -> Tuple[int, int, int]:
    """(X, W, Y) elements the app DECLARES -- the single source both halves read.

    ONE BINARY, THREE BUFFERS, each sized to the largest claim any stage makes on it: every
    cluster gets all three whatever stage it runs.  Under cwt a stage's claim is its own
    buffers PLUS its own second copy, so the max is taken over those SUMS -- taking it over
    each term separately would add stage 2's score matrices to stage 1's projection twin
    and bill one cluster for both, though no cluster is on both stages.
    """
    dx = dy = [0] * len(stages)
    if mode == 'cwt':
        spans = [_cwt_doubled(stages, geom, pool, i) for i in range(len(stages))]
        dx = [s[0] for s in spans]
        dy = [s[2] - s[1] for s in spans]
    return (max(st.x_elems + dx[i] for i, st in enumerate(stages)),
            max(st.w_elems for st in stages),
            max(st.y_elems + dy[i] for i, st in enumerate(stages)))


def _l1_footprint(stages: List[Stage], geom: Geometry, mode: str,
                  pool: str) -> Tuple[int, str]:
    """(bytes, breakdown) of the per-cluster L1 the generated app declares.

    NOT AN ESTIMATE: _render_scratch_buffers declares exactly what _buf_elems returns, so
    this is what the .l1 section holds.  Each buffer is placed on the pool's alignment
    boundary, so the gaps count too.  W never doubles -- nothing writes it.
    """
    nx, nw, ny = (n * BYTES_PER_ELEM
                  for n in _buf_elems(stages, geom, pool, mode))
    geo = L1_GEOMETRY.get(pool, {'stacks': 0, 'align': 64})
    at = geo['stacks']
    for sz in (nx, nw, ny):
        at = ((at + geo['align'] - 1) // geo['align']) * geo['align'] + sz
    total = at - geo['stacks']
    extra = ''
    if mode == 'cwt':
        ctt = _l1_footprint(stages, geom, 'ctt', pool)[0]
        extra = f'; {(total - ctt) / 1024:.0f} KiB more than ctt, all of it second copies'
    return total, (f'X {nx}B + W {nw}B + Y {ny}B + '
                   f'{(total - nx - nw - ny) / 1024:.0f} KiB align{extra}')


def _l1_budget(pool: str, l1_bytes: int) -> int:
    """What a generated app can declare: TCDM minus the stacks and the runtime's objects."""
    geo = L1_GEOMETRY.get(pool, {})
    return (min(l1_bytes, geo.get('total', l1_bytes))
            - geo.get('stacks', 0) - geo.get('runtime', 0))


def generate_apps(res: Results, apps_dir: str, cycles_per_ms: float, mode: str,
                  n_iter: int, bench: bool = False, wait: bool = False,
                  l1_bytes: int = 4 * 1024 * 1024) -> None:
    if not res.stages:
        print('Warning: no stages built -- nothing generated.')
        return

    kind = 'mempool_wait (timing-only)' if wait else 'real baremetal kernels'
    kind += ', benchmarked' if bench else ', uninstrumented'
    if mode == 'cwt':
        # WHAT THE OVERLAP BUYS, printed beside the app: ctt's period is the slowest compute
        # PLUS the slowest transfer, taken over DIFFERENT stages, so it hides none of the
        # residual's LPDDR round trip.  The more the transfers cost, the more cwt is worth.
        for pool in res.stages:
            ctt, cwt = res.best_throughput[pool], res.best_throughput_overlap[pool]
            if ctt:
                gain = 100 * (1 - cwt / ctt)
                why = ('the residual spill is large enough that hiding it behind compute is '
                       'worth real time' if gain >= 10 else
                       'the transfers are small beside the compute, so there is little to '
                       'hide')
                print(f'  [{pool}]  overlap buys {gain:.1f}% of the period '
                      f'({ctt:.6g} -> {cwt:.6g} ms): {why}.')

    sched = 'compute-while-transfer' if mode == 'cwt' else 'compute-then-transfer'
    # BLOCKING TRANSFERS WHENEVER THE POINT IS TO MEASURE THEM -- see _render_c.  A
    # single-iteration run is all fill and drain with no steady state to hide a transfer in,
    # and --bench wants every DMA waited on where it is issued so a trace can attribute
    # cycles to it.
    if mode == 'cwt' and (n_iter == 1 or bench):
        why = 'one iteration, so there is no steady state to hide them in' if n_iter == 1 \
              else '--bench, so each one is a measurable region'
        kind += f'; transfers BLOCKING ({why})'
    print(f'\nGenerating {sched} pipeline apps '
          f'({n_iter} iterations, {kind}) -> {apps_dir}/')
    for pool, stages in res.stages.items():
        app     = app_name(mode, pool)
        app_dir = Path(apps_dir) / app
        app_dir.mkdir(parents=True, exist_ok=True)
        c_src = _render_c(pool, stages, res.geom, cycles_per_ms, n_iter, bench, wait, mode)
        (app_dir / 'main.c').write_text(c_src)

        print(f'  [{pool}]  {app_dir / "main.c"}')
        l1, parts = _l1_footprint(stages, res.geom, mode, pool)
        geo       = L1_GEOMETRY.get(pool, {})
        budget    = _l1_budget(pool, l1_bytes)
        print(f'  [{pool}]  L1 per cluster: {l1 / 1024:.1f} KiB of {budget / 1024:.0f} KiB '
              f'({parts};  budget = {geo.get("total", l1_bytes) / 1024:.0f} KiB TCDM '
              f'- {geo.get("stacks", 0) / 1024:.0f} KiB stacks '
              f'- {geo.get("runtime", 0) / 1024:.0f} KiB runtime)')
        if l1 > budget:
            # Say what fixes it as well as that it is broken: the score matrix is what a
            # tile would halve, and it is the only tensor here big enough to matter.
            tiled = [(st.idx + 1, j + 1) for st in stages
                     for j, pt in enumerate(st.parts) if pt.cut.tiles == 1
                     and any(_sh(st, k).get('cls') == 'gemm'
                             and _sh(st, k)['P'] > 512 for k in range(pt.lo, pt.hi))]
            how = (f'  Set tiles=2 on stage {tiled[0][0]} part {tiled[0][1]} in STAGES to '
                   f'halve the score matrix -- it costs no latency, only serial passes.'
                   if tiled else '')
            print(f'  [{pool}]  WARNING: over budget by {(l1 - budget) / 1024:.1f} KiB -- '
                  f'the .l1 section will not link.{how}')
        print(f'  [{pool}]  build with:  make -C {apps_dir} {app} '
              f'COMPILER=llvm config=tensorpool256')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fit_yaxis(fig, ax, top: float, label_pt: float = 10.0) -> None:
    """Scale y to the data, leaving exactly the room the bar labels need."""
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
    """Bar plot of per-cluster wall time: compute (red) vs transfer (green).

    One x position per stage, one bar group per lane -- the lanes of a stage are identical
    replicas, so every bar in a group has the same height, and drawing them all is what
    shows that the stage's wall is the height of ONE lane and not their sum.

    Serialized -> stacked bars (wall = compute + transfer).
    Overlapped -> grouped bars (wall = max(compute, transfer)), with the inbound LPDDR read
    drawn hatched as the pedestal both bars stand on: both loops read item i's own input and
    hold the cores until it lands, so it is exactly din + max(compute, dout).
    """
    bases, occupied = _stage_bases()
    tag    = 'CWT' if overlap else 'CTT'
    n_work = sum(st.n_par for st in stages)

    fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * n_work + 2.5), 4.4))
    span  = 0.82
    first = True
    first_din = True
    tick_pos, tick_lab = [], []

    for s, (st, base) in enumerate(zip(stages, bases)):
        p     = st.n_par
        slots = 2 * p if overlap else p
        bw    = span / slots
        draw  = min(bw * 0.9, 0.30 if overlap else 0.34)
        left  = s - span / 2
        wall  = _overlap_wall(st) if overlap else st.compute_ms + st.transfer_ms
        if p > 1:
            ax.axvspan(left - 0.03, left + span + 0.03, color='0.92', zorder=0)
        for lane in range(p):
            if overlap:
                xmid = left + (2 * lane + 1.0) * bw
                if st.din_ms > 0.0:
                    for x in (xmid - bw / 2, xmid + bw / 2):
                        ax.bar(x, st.din_ms, width=draw, color='green', hatch='//',
                               edgecolor='white', zorder=2,
                               label=('LPDDR in (not overlapped)'
                                      if first_din and x < xmid else None))
                    first_din = False
                ax.bar(xmid - bw / 2, st.latency, width=draw, bottom=st.din_ms,
                       color='red', zorder=2, label='compute' if first else None)
                ax.bar(xmid + bw / 2, st.transfer_ms, width=draw, bottom=st.din_ms,
                       color='green', zorder=2, label='transfer' if first else None)
            else:
                xmid = left + (lane + 0.5) * bw
                ax.bar(xmid, st.compute_ms, width=draw, color='red', zorder=2,
                       label='compute (incl. LPDDR in)' if first else None)
                ax.bar(xmid, st.transfer_ms, width=draw, bottom=st.compute_ms,
                       color='green', zorder=2, label='transfer' if first else None)
            first = False
            ax.annotate(f'c{base + lane}', xy=(xmid, wall), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, color='0.35')
        tick_pos.append(s)
        tick_lab.append(f'stage {s + 1}\n' + ('(1 lane)' if p == 1 else f'({p} lanes || )'))

    top = max((_overlap_wall(st) if overlap else st.compute_ms + st.transfer_ms)
              for st in stages)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_xlabel('pipeline stage  (the lanes of a stage run in parallel)')
    ax.set_ylabel('wall time [ms]')
    idle = occupied - n_work
    ax.set_title(f'{pool} -- {tag}  |  {len(stages)} stages over {n_work} clusters '
                 f'({GRID_X}x{GRID_Y} grid' + (f', {idle} idle)' if idle else ')'))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small',
              frameon=False)
    fig.tight_layout()
    _fit_yaxis(fig, ax, top)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def generate_plots(res: Results, mode: Optional[str], out_dir: Path) -> None:
    """Save per-pool bar plots of the stage walls to out_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless: write PNGs, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: matplotlib not installed; --plot skipped (pip install matplotlib).')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nSaving stage plots -> {out_dir}/')
    for p, stages in res.stages.items():
        psafe = p.replace(' ', '_')
        if mode in (None, 'ctt'):
            path = out_dir / f'CEViT_CTT_split_{psafe}.png'
            _plot_split(plt, p, stages, False, path)
            print(f'  [plot] {path}')
        if mode in (None, 'cwt'):
            path = out_dir / f'CEViT_CWT_split_{psafe}.png'
            _plot_split(plt, p, stages, True, path)
            print(f'  [plot] {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_summary(res: Results, path: str, csv_name: str) -> None:
    """Write the machine-readable per-(pool, mode) summary latency_throughput_bars.py reads.

    Columns: pool, mode, throughput_ms (= T_pipe, the bottleneck), n_stages, n_clusters,
    latency_ms.

    latency_ms is the end-to-end time of ONE item IN STEADY STATE, not the critical-path
    sum: every cluster rendezvous on a global barrier each step, so every stage costs a
    full period and the wavefront fixes how many steps an item is resident -- ctt puts
    stage s's work on item m at step m+s, cwt at m+2s.

    THE FACTOR OF TWO IS FORCED, not a safety margin: at m + s a stage and its consumer
    work on adjacent items in the SAME step, so the transfer between them would have to
    land in the step the consumer is already reading it.  So cwt shortens the period but
    stretches the time in flight -- it buys THROUGHPUT and pays in latency.
    """
    views = (('ctt', res.best_throughput,         1),
             ('cwt', res.best_throughput_overlap, 2))
    lines = [
        '# CEViT pipeline summary  (read by latency_throughput_bars.py)',
        f'# csv: {csv_name}',
        '# columns: pool  mode  throughput_ms  n_stages  n_clusters  latency_ms',
    ]
    for mode, best, steps_per_stage in views:
        for pool, stages in res.stages.items():
            n_stages   = len(stages)
            # The clusters that DO WORK, not the cids the pipeline spans: this column feeds
            # per-cluster efficiency comparisons, and idle padding is not a cost the
            # pipeline pays for compute -- it is a placement artefact.  The console table
            # prints both.
            n_clusters = sum(st.n_par for st in stages)
            latency    = steps_per_stage * n_stages * best[pool]
            lines.append(f'{pool}\t{mode}\t{best[pool]:.6g}\t{n_stages}\t{n_clusters}'
                         f'\t{latency:.6g}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')
    print(f'\nWrote pipeline summary -> {path}')


def run(template_path: str, config_path: str) -> Tuple[Results, List[LayerRow]]:
    """Programmatic entry point: returns (Results, parsed rows)."""
    cfg    = load_config(config_path)
    layers = load_template(template_path)
    res    = compute(layers, cfg)

    by_row = {r.excel_row: r for r in layers}
    for rn in DROPPED_ROWS:
        r = by_row.get(rn)
        if r:
            print(f"Rides a transfer (no kernel, no local copy): row {rn} '{r.op}' "
                  f'[{r.kernel}], {int(r.dout_bytes)} B')
    return res, layers


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('template', help='Path to template (.xlsx, .xlsm, .ods or .csv)')
    ap.add_argument('config',   help='Path to JSON HW config')
    ap.add_argument(
        '--mode', choices=['ctt', 'cwt'], metavar='{ctt,cwt}',
        help='Limit to one pipeline flavour: ctt = compute-then-transfer, '
             'cwt = compute-while-transfer.  Both are modelled and generated by default.')
    ap.add_argument('--iterations', type=int, default=1, metavar='N',
                    help='Items streamed through the pipeline (default: 1)')
    ap.add_argument('--out', metavar='DIR', default=str(_APPS_DIR),
                    help='Apps root under which cevit_<mode>_<pool>/main.c is created '
                         f'(default: {_APPS_DIR}).')
    ap.add_argument('--summary', metavar='PATH', default='pipeline_summary.txt',
                    help='Write the machine-readable per-(pool, mode) summary '
                         '(default: ./pipeline_summary.txt).')
    ap.add_argument('--bench', action='store_true',
                    help='Bracket every layer and every DMA of the generated app with '
                         'mempool_start_benchmark()/mempool_stop_benchmark(), so an RTL run '
                         'traces exactly the kernels and the transfers.  The markers are two '
                         '`trace` CSR writes -- they read no counter and print nothing.')
    ap.add_argument('--wait', action='store_true',
                    help='Emit the timing-only twin, where every layer is a mempool_wait() '
                         'spin sized to the modelled latency instead of a real baremetal '
                         'kernel call.  Needs no tensor-shape columns for the KERNELS, but '
                         'the hand-off descriptors still do.')
    ap.add_argument('--l1-bytes', type=int, default=4 * 1024 * 1024, metavar='N',
                    help='Cluster TCDM budget in bytes, used to warn when the generated '
                         'app\'s buffers do not fit (default: 4 MiB, the cluster_tcdm_size '
                         'of the TensorPool and Terapool arch configs).')
    ap.add_argument('--plot', action='store_true',
                    help='Save a per-pool bar plot of the stage walls as PNGs under '
                         './cevit_split_plots/.')
    args = ap.parse_args()

    if args.iterations < 1:
        ap.error('--iterations must be >= 1')

    cfg = load_config(args.config)
    res, layers = run(args.template, args.config)

    print_results(layers, res, args.mode)
    write_summary(res, args.summary, Path(args.template).name)

    cycles_per_ms = cfg.get('frequency', 1_000_000_000) / 1e3
    for m in ([args.mode] if args.mode else ['ctt', 'cwt']):
        generate_apps(res, args.out, cycles_per_ms, m, args.iterations,
                      args.bench, args.wait, args.l1_bytes)

    if args.plot:
        generate_plots(res, args.mode, Path.cwd() / 'cevit_split_plots')


if __name__ == '__main__':
    main()
