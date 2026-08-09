#!/usr/bin/env python3
"""
Compute CEViT performance tables and generate the GVSOC pipeline simulation
apps from a START/END-delimited template.

This is the CEViT-dedicated fork of pipeline.py: same parsing / costing /
cluster-splitting machinery, but its own home for CEViT-specific parsing and
optimisation, and its own artifact names so it cannot clobber the L1-transformer
ones.  Running it prints the performance tables and generates one app per
(mode, pool) under software/apps/multi_cluster/, named
`cevit_<mode>_<pool>/main.c` -- the flat layout the multi-cluster Makefile
discovers (`find $(APPS_DIR) -name "main.c"`), so each app builds with
`make -C software/apps/multi_cluster cevit_<mode>_<pool>`.  With no --mode it
prints both cluster splits (serialized + overlapped) and generates both
flavours; with --mode {ctt,cwt} it prints only that mode's split and
generates only its apps:

    * ctt -- compute-then-transfer streaming pipeline: --iterations N items are
      streamed as a wavefront (FILL -> STEADY -> DRAIN), single L1 buffer per
      cluster, compute and DMA SERIALIZED.
    * cwt -- compute-while-transfer streaming pipeline: same wavefront but with a
      DOUBLE-buffered L1 (2x footprint) and a core-role split so compute and DMA
      OVERLAP.

Each stage emits its exact ordered sequence of layers (block reps in a `for` loop).
By DEFAULT every layer is lowered to a REAL baremetal kernel call from
software/kernels/baremetal (GEMM on the RedMulE tensor engines for TensorPool / on
the PE matmul for Terapool; softmax, layernorm, gelu, add and the bookend
patchify/unpatchify on the PE path; the remaining reshape/split/transpose as local DMA
copies), operating on placeholder L1 scratch --
these kernels' cycle counts are data-independent, so the apps still need no HBM
preload and no data_<app>.h.  Pass --wait for the old timing-only behaviour, where
each layer is instead a `mempool_wait()` spin sized to the modelled latency; that
mode needs none of the tensor-shape columns.  Real-kernel mode derives each layer's
M/N/P from the spreadsheet's i_*/o_* (input / output) dim columns; a matmul's second
operand needs no columns of its own, since its shape follows from i_* and o_*.

The template (xlsx, ods or csv) is parsed without markers: the header row (the
one labelling N_repetitions / FLOPs / Weights / Data_in / Data_out) is located
automatically.  Layers are read from the row below it until the trailing
parameter block -- reading stops at the first row whose block-name cell
(column A) is filled but whose operation cell (column C) is empty, since a real
block header always carries its first layer's operation on the same row.
Anything above the header or below that stop point (parameter tables, notes or
scratch totals) is ignored, so those can live at the top or bottom of the file.

Column mapping in the data region:
    * Block name      -> column A          (no header expected)
    * Operation name  -> column C          (no header expected)
    * Kernel          -> located by header ("Kernel")
    * N_repetitions   -> located by header
    * FLOPs           -> located by header
    * Weights         -> located by header
    * Data_in         -> located by header
    * Data_out        -> located by header
A layer whose Kernel is in GEMM_KERNELS (dense, matmul, conv1d) is costed at
GEMM_peak x TE_utilization on a pool that HAS a tensor engine (TENSOR_ENGINE_POOLS)
and at GEMM_peak x PE_utilization on one that does not, where the GEMM is a PE matmul
like any other kernel -- TE_utilization never enters such a pool's numbers.  Every
other kernel runs on the PE path (PE_peak x PE_utilization).  Weights / Data_in /
Data_out are read in elements; bytes are computed as 2 x elements (FP16).

HEAD-PARALLEL ATTENTION
-----------------------
CEViT's attention core (Q*K^T, softmax, (Q*K^T)*V) materialises the
[heads, tokens, tokens] score matrix: 8.1 MB of L1 at 176 PRBs, 16.3 MB at 250 --
more than one cluster can hold.  Those layers carry no weights and no cross-head
reduction, so they split cleanly over the head dimension (i_2).  They are therefore
cut out of the Attention block into a PIPELINE STAGE OF THEIR OWN, spread over
HEAD_PARALLEL_CLUSTERS clusters (4) with one head each, which divides both the
stage's compute and its L1 footprint by that width (-> 1.9 MB / 3.9 MB per cluster).

A stage is thus no longer always one cluster.  In the generated C, `cid` is mapped
through CID_STAGE[] / CID_LANE[] to a (stage, lane) pair; the wavefront is keyed on
the stage, and the hand-off at a width change SCATTERS head-chunks (1 -> P) or
GATHERS them back (P -> 1).  The per-step global barrier already rendezvous the
lanes of a stage, so no group barrier is needed.

HOW MANY CLUSTERS
-----------------
One hard config knob, "pipeline_clusters": the total the pipeline occupies.  Exactly
one stage is head-parallel and takes HEAD_PARALLEL_CLUSTERS of them; every other stage
is one cluster.  The stage count therefore follows outright,

    n_stages = pipeline_clusters - HEAD_PARALLEL_CLUSTERS + 1

and the optimizer's only job is to PLACE those cuts (best_split, exact DP over the
atomic units).  It never picks the count, so both pools always run the same shape and
a cluster budget is never silently exceeded or left unused; a value the model admits
no legal split for is an error naming the feasible range.

Hardware peak performance and utilisation are read from a JSON config:
    {
        "PE_peak":             {"Terapool": 3.7, "TensorPool": 1.0},
        "GEMM_peak":           {"Terapool": 1.1, "TensorPool": 6.62},
        "PE_utilization":      0.5,
        "TE_utilization":      0.5,
        "pipeline_clusters":   7,
        "bandwidth_c2c_gbs":   64,
        "bandwidth_lpddr_gbs": 6.4
    }
PE_peak and GEMM_peak are in TFLOPS. PE_utilization is a fraction in [0, 1] derating
everything that runs on the PE path -- non-GEMM ops on every pool, and GEMMs too on a
pool outside TENSOR_ENGINE_POOLS. TE_utilization is a fraction in [0, 1] derating
GEMM_peak, and it applies ONLY to pools that have a tensor engine.

Layout layers (reshape / split / transpose) carry 0 FLOPs but re-lay the tensor out.
NO local L1->L1 copy is ever emitted for one: moving a tensor the neighbouring hand-off
already moves would move it twice.  Each is therefore resolved one of two ways, and a
layout row that is neither is a hard error rather than a quiet local copy:

  * FOLDED INTO A HAND-OFF.  The cluster-to-cluster DMA goes through the 2D engine, so
    it can re-lay the tensor out as its address pattern.  Both sides of the
    head-parallel attention stage fold this way and are dropped right after parsing --
    they appear in NO table, cost nothing and emit no code of their own, and the run
    log lists them:
      - on the way IN, FOLDED_LAYOUT_OPS ('split q,k,v' / 'split heads' /
        'transpose Q,K,V'), carried by the 1->P scatter;
      - on the way OUT, the layout run right after the head-parallel ops (the attention
        'transpose' + 'reshape'), carried by the P->1 gather, which interleaves each
        lane's rows into its column slot of the consumer's matrix.  Picked by position
        rather than by name, and only when ONE 2D descriptor can express the composed
        permutation -- see fold_gathered_layout_layers.
  * RUN BY THE PEs.  The BOOKENDS -- the model's leading layout run and its trailing
    one -- are the network's patchify and unpatchify, between the [Nf, Nt, C] grid and
    the [Nf/Nh*Nt/Nw*C, Nh*Nw] token matrix.  No hand-off borders them (they are the
    first and last layers), and they are a real index permutation, so the cores run
    them: real-kernel mode emits patchify_f16 / unpatchify_f16 INTO the app itself, and
    they are costed on the PE path at PE_LAYOUT_FLOPS_PER_ELEM per element moved
    (pool-dependent, like all compute).  The patch split is not in the template, so it
    comes from the PATCH_F / PATCH_T constants, which are reconciled with each tagged
    row's dims.  Being the FIRST and LAST layers they always fall in stage 0 and the
    last stage -- so they too are picked by position, and the choice cannot depend on a
    split it would otherwise perturb.  The run log lists them.

Every transfer the generated app issues -- cluster-to-cluster hand-off, HBM load and
HBM writeback alike -- goes through flex_dma_async_2d, funnelled through the
dma_2d_strided/dma_2d(_issue) helpers.  A contiguous transfer is the degenerate 2D
descriptor (one repetition, both strides = the chunk size), so nothing changes about
the bytes moved; using one engine path throughout is what lets a hand-off carry a
re-layout in its strides rather than needing a separate copy.

Transfer time is bytes/(bandwidth_gbs*1e6) ms, and each leg is costed at the
bandwidth of the link it crosses (either is optional; omit or 0 to make that link
free):
    * bandwidth_c2c_gbs   -- on-chip cluster-to-cluster data_out hand-off.
    * bandwidth_lpddr_gbs -- cluster <-> LPDDR: cluster 0's data_in read and the
                             LAST cluster's data_out write, which leave the chip.
So each cluster's transfer is its last layer's data_out (to its successor, or to
LPDDR if it is the last cluster), plus the LPDDR data_in load on cluster 0.

The script reproduces the three reference tables:
    * K1:M3   -> FLOPs/ms per pool, split non-GEMM / GEMM
    * T52:W54 -> L1 footprint (compute-then-transfer vs compute-while-transfer)
    * Y5:Z53  -> Per-row, per-block and total latency [ms] per pool

For xlsx/ods, the script reads the *cached* calculated values of formula cells.
If a spreadsheet was edited but never opened in a calculator (Excel /
LibreOffice), reopen and save it once so the cached values are refreshed.

Usage:
    python cevit_pipeline.py CEViT_172PRB.csv config.json                 # both splits, both modes
    python cevit_pipeline.py CEViT_172PRB.csv config.json --mode ctt --iterations 8
    python cevit_pipeline.py CEViT_172PRB.csv config.json --mode cwt --iterations 8
    python cevit_pipeline.py CEViT_246PRB.csv config.json --out /tmp/apps # other apps root
    python cevit_pipeline.py CEViT_246PRB.csv config.json --plot          # also save split bar plots
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
DEFAULT_COLS: Dict[str, int] = {
    'block': 0,   # column A
    'op':    2,   # column C
}

# Tensor-shape columns, located by their header label.  These are OPTIONAL: they
# are only needed to emit real kernel calls (the default), which derive each
# layer's M/N/P from them.  The --wait mode never touches them, so a template
# without these columns still works there.  `in_*` = input operand dims, `out_*` =
# output dims.  There are no second-operand (j_*) columns: a matmul's/dense's second
# operand is (N, P) = (in_dims[-1], out_dims[-1]), i.e. fully implied by the two.
# Trailing empty cells are dropped, so [1, 688, 84, ''] -> [1, 688, 84].
DIM_HEADERS: Dict[str, str] = {
    'in_1': 'i_1', 'in_2': 'i_2', 'in_3': 'i_3', 'in_4': 'i_4',
    'out_1': 'o_1', 'out_2': 'o_2', 'out_3': 'o_3', 'out_4': 'o_4',
}

BYTES_PER_ELEM = 2  # spreadsheet uses *2 (FP16)

# Kernels dispatched to the GEMM path (costed at GEMM_peak); every other kernel runs
# on the PE path (PE_peak x PE_utilization).  Read from the Kernel column.
# conv1d is here because it is lowered to a GEMM (im2col) on both pools.
GEMM_KERNELS = {'dense', 'matmul', 'conv1d'}

# Pools with a RedMulE tensor engine.  A GEMM only lands on dedicated silicon here, so
# only here is GEMM_peak derated by TE_utilization; on a pool without one a GEMM is a
# PE matmul like everything else and takes PE_utilization -- TE_utilization must not
# touch it.  This also picks the kernel the generated C calls (redmule_synch_parallel
# vs matmul_4x2_parallel_f16vec), so both the model and the code stay on one fact.
TENSOR_ENGINE_POOLS = {'TensorPool'}

# Head-parallel attention.  These ops materialise the [heads, tokens, tokens] score
# matrix, whose L1 footprint (8.1 MB @176 PRBs, 16.3 MB @250 PRBs) does not fit in a
# single cluster.  They carry no weights and no cross-head reduction, so they split
# cleanly over i_2 (the head count): each cluster owns one head, holding 1/P of the
# footprint and doing 1/P of the FLOPs.  They must be CONTIGUOUS inside the block, and
# the run becomes its own pipeline stage spanning HEAD_PARALLEL_CLUSTERS clusters.
HEAD_PARALLEL_BLOCK = 'Attention'
HEAD_PARALLEL_OPS   = ['Q*K^T', 'softmax', '(Q*K^T)*V']

# Width of that stage, i.e. how many heads run side by side.  A property of the model
# (the score matrix has to be cut this many ways to fit L1), not a tuning knob, so it
# lives here and not in the config: config.json only says how many clusters the WHOLE
# pipeline gets ("pipeline_clusters"), of which this stage takes these.
HEAD_PARALLEL_CLUSTERS = 4

# Patch geometry: one patch is PATCH_F subcarriers x PATCH_T symbols.  The model's first
# and last layout rows are the patchify / unpatchify between the [Nf, Nt, C] grid and the
# [Nf/Nh*Nt/Nw*C, Nh*Nw] token matrix, and the PE kernels that run them need this split --
# which the template does NOT determine: it gives Nh*Nw (the output row length, 84) but not
# how it factors, and both (6, 14) and (12, 7) divide the dims consistently.  It is a
# property of the model like HEAD_PARALLEL_CLUSTERS, so it lives here rather than in the
# config; mark_pe_layout_rows validates it against every row it tags.
PATCH_F, PATCH_T = 12, 7

# Cost of a PE-run layout kernel (patchify / unpatchify): a per-element term charged on the
# PE path as a FLOP-equivalent, plus a per-invocation entry cost (function entry + the
# all-core flex_intra_cluster_sync the caller rendezvous on).  BOTH are needed -- the tensor
# scales with the model's PRB count and the entry cost does not, so a purely per-element
# charge cannot fit two models at once.  Fitted through measured TensorPool runs (256 cores,
# 1 GHz) of cycles = FIXED + elems * K / (PE_peak * PE_utilization * 1e3):
#
#            elements   patchify   unpatchify
#   174PRB     58,464       2,892        2,991
#   246PRB     82,656       3,663        3,751
#
# K lands at 8.9 -- close to the 2 instructions x 4 FLOPs/instruction the scalar kernel
# suggests, once TCDM contention is folded in -- and comes out the same in either direction
# (8.16 vs 8.04 cycles/element/core), one strided side either way, so one K covers both.
# Re-fit from a fresh pair if the kernel changes.  NOTE the fixed term is a 256-core
# rendezvous: on Terapool's 1024 cores it is understated.
PE_LAYOUT_FLOPS_PER_ELEM = 8.9
PE_LAYOUT_FIXED_CYCLES = {'patchify': 1029, 'unpatchify': 1154}

# Layout ops absorbed into the pipeline hand-off.  Each of these is the tail of the stage
# that feeds the head-parallel attention core, and the cluster-to-cluster DMA that hands
# that stage's data_out to the next stage can perform the re-layout as its address
# pattern -- doing it again locally would just copy the same tensor twice.  They are
# therefore dropped from the model entirely: no latency, no L1 footprint, no emitted copy.
# Matched on the op name (column C), which is identical in every CEViT template.
FOLDED_LAYOUT_OPS = {'split q,k,v', 'split heads', 'transpose Q,K,V'}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bandwidths:
    """Per-link bandwidths [GB/s].  An unset (None) or 0 link costs no time.

    The pipeline crosses two physically different links, so they are costed apart:
    an on-chip cluster-to-cluster hand-off is not the same as going off-chip to LPDDR.
    """
    c2c:   Optional[float] = None   # cluster -> cluster data_out hand-off (on-chip)
    lpddr: Optional[float] = None   # cluster <-> LPDDR: cluster 0's data_in read and
                                    # the last cluster's data_out write


@dataclass
class LayerRow:
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
    # Tensor shapes (trailing empties dropped).  Empty when the template lacks the
    # i_*/o_* columns; real-kernel emission requires them, --wait does not.
    in_dims:       List[int] = field(default_factory=list)   # input operand
    out_dims:      List[int] = field(default_factory=list)   # output
    # Layout rows the PEs run instead of the local DMA: '' (a DMA copy, the default),
    # 'patchify' (the model's leading layout run) or 'unpatchify' (its trailing one).
    # Set by mark_pe_layout_rows before costing, since it changes the row's latency.
    pe_layout:     str = ''


@dataclass
class Block:
    name:  str
    n_rep: int
    rows:  List[LayerRow] = field(default_factory=list)
    # Clusters this block is spread over in parallel (1 = ordinary sequential block).
    # >1 marks a head-parallel run: the block becomes a stage of n_par clusters, each
    # owning one head, so its compute and its L1 footprint are divided by n_par.
    n_par: int = 1


@dataclass
class ClusterSegment:
    """A slice of one block assigned to a cluster.

    A block is split ONE of two orthogonal ways, never both:
      * rep-range  (rep_from..rep_to) -- for n_rep>1 blocks split across clusters by whole
        passes; the row-range then spans every row (the default).
      * row-range  (row_from..row_to) -- for n_rep=1 blocks split mid-block at layer
        granularity so a long block (e.g. FFN) can spread over serial pipeline stages;
        rep_from==rep_to==0.
    """
    block:    Block
    rep_from: int   # 0-based, inclusive
    rep_to:   int   # 0-based, inclusive
    row_from: int = 0                 # 0-based, inclusive
    row_to:   Optional[int] = None    # 0-based, inclusive; None => through the last row

    @property
    def n_reps(self) -> int:
        return self.rep_to - self.rep_from + 1

    @property
    def rows(self) -> List[LayerRow]:
        """The block rows this segment actually owns (its row sub-range)."""
        end = len(self.block.rows) if self.row_to is None else self.row_to + 1
        return self.block.rows[self.row_from:end]

    @property
    def is_partial(self) -> bool:
        """True when this segment holds only part of its block's rows."""
        return self.row_from > 0 or (self.row_to is not None
                                     and self.row_to < len(self.block.rows) - 1)


@dataclass
class ClusterSplit:
    """One PIPELINE STAGE.  A stage owns n_par clusters running in parallel (n_par == 1
    for every ordinary stage); every figure below is therefore PER CLUSTER, already
    divided by n_par where the stage is head-parallel."""
    idx:                       int     # stage index (NOT a physical cluster id)
    segments:                  List[ClusterSegment]
    latency:                   float   # compute latency [ms] of ONE cluster of this stage
    transfer_ms:               float   # hand-off transfer time [ms] (dout/BW, +din/BW for stage 0)
    din_ms:                    float   # stage 0 only: the LPDDR data_in read [ms], already
                                       # INSIDE transfer_ms.  Held apart because both generated
                                       # loops serialize it before the compute (dma_2d then
                                       # flex_intra_cluster_sync), so it can never hide behind
                                       # the compute the way the outbound hand-off does.
    first_weights:             float   # weights_bytes of the stage's first layer
    first_din:                 float   # din_bytes of the stage's first layer
    last_weights:              float   # weights_bytes of the stage's last layer
    last_dout:                 float   # dout_bytes of the stage's last layer (pipeline output)
    l1_compute_then_transfer:  float   # per-cluster (max(w)+max(din)+max(dout))/n_par [bytes]
    l1_compute_while_transfer: float   # 2 × compute_then_transfer [bytes]
    n_par:                     int = 1  # clusters running this stage in parallel


@dataclass
class Results:
    flops_per_ms: Dict[str, Dict[str, float]]         = field(default_factory=dict)
    l1_compute_then_transfer:  float                  = 0.0
    l1_compute_while_transfer: float                  = 0.0
    row_latencies:   Dict[str, Dict[int, float]]      = field(default_factory=dict)
    block_latencies: Dict[str, List[Dict]]            = field(default_factory=dict)
    total_latency:   Dict[str, float]                 = field(default_factory=dict)
    blocks:          List[Block]                      = field(default_factory=list)
    # Serialized split (ctt: wall = compute + transfer).
    clusters:        Dict[str, List[ClusterSplit]]    = field(default_factory=dict)
    # Overlapped split (cwt: wall = max(compute, transfer)).
    clusters_overlap: Dict[str, List[ClusterSplit]]   = field(default_factory=dict)
    # Best (minimised) per-cluster wall = pipeline throughput, after rebalancing
    # the work over the fixed cluster count.  Keyed by pool, one per wall model.
    best_throughput:         Dict[str, float]         = field(default_factory=dict)  # ctt
    best_throughput_overlap: Dict[str, float]         = field(default_factory=dict)  # cwt
    # Descriptor with which the gather carries the attention output re-layout, when the
    # model has such a run and one 2D descriptor can express it (see GatherFold).
    gather_fold:     Optional['GatherFold']           = None


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

    The header row is the first row in which *every* named field of
    HEADER_ALIASES has a matching label cell (same test build_column_map uses).
    Secondary sub-header rows (e.g. "Data_in (Bytes)") do not match because the
    strings differ, so they are skipped automatically.
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

    Layers run from `first_row` until the trailing parameter block: reading
    stops at the first row whose block-name cell (column A) is filled but whose
    operation cell (column C) is empty.  A real block header always carries its
    first layer's operation on the same row, so this only fires on the bottom
    parameter table (or a stray END-like row).  Falls back to end-of-grid.
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

    The i_*/o_* tensor-shape columns are OPTIONAL: they are located when present
    (needed for real-kernel emission) and simply omitted otherwise, so --wait-only
    templates still parse.
    """
    cols = dict(DEFAULT_COLS)
    for field_name, aliases in HEADER_ALIASES.items():
        match = next(
            (c for c, v in enumerate(header_row) if _to_str(v) in aliases),
            None,
        )
        if match is None:
            raise ValueError(
                f"Header for {field_name!r} not found (expected one of {aliases}). "
                f"Got header row: {[_to_str(x) for x in header_row]}"
            )
        cols[field_name] = match
    for field_name, label in DIM_HEADERS.items():
        match = next(
            (c for c, v in enumerate(header_row) if _to_str(v) == label),
            None,
        )
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
            + "\n\nOpen the file in Excel/LibreOffice, recalculate (Ctrl+Shift+F9), save, and try again."
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
            vals: List[int] = []
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


def drop_folded_layout_layers(rows: List[LayerRow]) -> Tuple[List[LayerRow], List[LayerRow]]:
    """Split off the layout rows the pipeline hand-off performs (FOLDED_LAYOUT_OPS).

    Returns (kept, folded).  The folded rows are gone from the model for good: they cost
    no latency, claim no L1 and emit no code, because the stage hand-off that follows them
    already moves the same tensor and can re-lay it out on the fly.

    That argument only holds for an op that neither computes nor resizes anything, so the
    two properties are checked rather than assumed -- a template edit that breaks either
    must fail loudly instead of quietly dropping real work.
    """
    kept, folded = [], []
    for r in rows:
        if r.op not in FOLDED_LAYOUT_OPS:
            kept.append(r)
            continue
        if r.flops:
            raise ValueError(
                f"Row {r.excel_row} ('{r.op}') is in FOLDED_LAYOUT_OPS but carries "
                f"{r.flops:g} FLOPs; only compute-free ops can ride the hand-off.")
        if r.din_bytes != r.dout_bytes:
            raise ValueError(
                f"Row {r.excel_row} ('{r.op}') is in FOLDED_LAYOUT_OPS but resizes its "
                f"tensor ({r.din_bytes:g} -> {r.dout_bytes:g} bytes); only a "
                f"size-preserving op can ride the hand-off.")
        if r.block:
            raise ValueError(
                f"Row {r.excel_row} ('{r.op}') is in FOLDED_LAYOUT_OPS but opens block "
                f"'{r.block}'; dropping it would orphan the rest of the block.")
        folded.append(r)
    return kept, folded


@dataclass(frozen=True)
class GatherFold:
    """The 2D descriptor with which the P->1 gather carries the attention output re-layout.

    row_bytes is the contiguous run one lane contributes to each destination row; the rest
    of the descriptor follows from it at run time (reps = own/row_bytes, dst_stride =
    row_bytes*lanes, src_stride = row_bytes), so the generated C needs this one number.
    """
    row_bytes: int
    reps:      int
    rows:      Tuple['LayerRow', ...]


def fold_gathered_layout_layers(rows: List[LayerRow],
                                n_par: int) -> Tuple[List[LayerRow], Optional[GatherFold]]:
    """Split off the layout run the GATHER performs, and return its 2D descriptor.

    The mirror of drop_folded_layout_layers, on the other side of the head-parallel stage:
    there the scatter re-lays the tensor out on its way IN, here the gather does it on the
    way OUT.  Both leave no local copy behind -- the only data movement the model keeps is
    a hand-off between stages or a PE kernel in the first and last stage.

    The rows are picked by POSITION, not by name: the layout run immediately following the
    head-parallel ops.  Those ops are forced to be a stage of their own
    (split_head_parallel_blocks), so that run is necessarily the head of the stage the
    gather feeds -- whereas the op names here are the bare 'transpose' / 'reshape', which
    would also match the patchify bookends.

    ONE descriptor can only do this when the run composes to

        [n_par, R, D]  ->  [R, n_par*D]

    i.e. each lane owns a dense [R, D] block and lands in column slot `lane` of the result:
    chunk D*BYTES_PER_ELEM, reps R, src_stride D*BYTES_PER_ELEM, dst_stride n_par*D*BYTES_PER_ELEM.
    Both offset sequences are then uniform, which is exactly what the engine's 2D form
    (`for r in reps: copy chunk from src + r*src_stride to dst + r*dst_stride`) can express.
    A run that composes to anything else RAISES rather than folding on a claim the hand-off
    could not honour: it needs a PE kernel, which is a decision for whoever changes the
    template, not something to paper over with a local copy.
    """
    if n_par <= 1:
        return rows, None

    def is_layout(r: LayerRow) -> bool:
        return r.kernel.lower() in LAYOUT_KERNELS

    # Locate the head-parallel run, tracking the block each row belongs to (only the
    # block's first row carries the name).
    block, last_par = '', -1
    for i, r in enumerate(rows):
        if r.block:
            block = r.block
        if block == HEAD_PARALLEL_BLOCK and r.op in HEAD_PARALLEL_OPS:
            last_par = i
    if last_par < 0:
        return rows, None

    run: List[LayerRow] = []
    for r in rows[last_par + 1:]:
        if r.block or not is_layout(r):
            break
        run.append(r)
    if not run:
        return rows, None

    for r in run:
        if r.flops:
            raise ValueError(
                f"Row {r.excel_row} ('{r.op}') follows the head-parallel run but carries "
                f"{r.flops:g} FLOPs; only compute-free ops can ride the gather.")
        if r.din_bytes != r.dout_bytes:
            raise ValueError(
                f"Row {r.excel_row} ('{r.op}') follows the head-parallel run but resizes "
                f"its tensor ({r.din_bytes:g} -> {r.dout_bytes:g} bytes); only a "
                f"size-preserving op can ride the gather.")

    # Leading 1s are the template's tensor-count column, not a real dimension.
    def squeeze(dims: List[int]) -> List[int]:
        i = 0
        while i < len(dims) - 1 and dims[i] == 1:
            i += 1
        return dims[i:]

    src_dims, dst_dims = squeeze(run[0].in_dims), squeeze(run[-1].out_dims)
    if len(src_dims) != 3 or len(dst_dims) != 2 or src_dims[0] != n_par \
            or dst_dims[0] != src_dims[1] or dst_dims[1] != n_par * src_dims[2]:
        raise ValueError(
            f"Rows {run[0].excel_row}-{run[-1].excel_row} sit between the head-parallel "
            f"stage and its consumer, but compose to {src_dims} -> {dst_dims} instead of "
            f"[{n_par}, R, D] -> [R, {n_par}*D].  One 2D descriptor cannot express that, "
            f"so the gather cannot carry it; give those rows a PE kernel instead.")

    reps, D = src_dims[1], src_dims[2]
    kept = [r for r in rows if all(r is not f for f in run)]
    return kept, GatherFold(row_bytes=D * BYTES_PER_ELEM, reps=reps, rows=tuple(run))


def _patch_geometry(row: 'LayerRow') -> Dict[str, int]:
    """(Nf, Nt, C, Nh, Nw) of a patchify / unpatchify row, checked against PATCH_F/PATCH_T.

    One side of the row is the grid [Nf, Nt, C] and the other the token matrix
    [(Nf/Nh)*(Nt/Nw)*C, Nh*Nw]; which is which follows from the direction.  The patch split
    is not in the template (only Nh*Nw is), so the constant is reconciled with the dims here
    rather than trusted: a template whose geometry it cannot explain must fail loudly.
    """
    grid = row.in_dims  if row.pe_layout == 'patchify' else row.out_dims
    tok  = row.out_dims if row.pe_layout == 'patchify' else row.in_dims
    if len(grid) < 3 or len(tok) < 2:
        raise ValueError(
            f"Row {row.excel_row} ('{row.op}') is the model's {row.pe_layout} but its dims "
            f"are {row.in_dims} -> {row.out_dims}; the grid side needs [.., Nf, Nt, C] and "
            f"the token side [.., rows, Nh*Nw].")
    Nf, Nt, C = grid[-3:]
    R,  P     = tok[-2:]
    Nh, Nw    = PATCH_F, PATCH_T
    if Nh * Nw != P or Nf % Nh or Nt % Nw or (Nf // Nh) * (Nt // Nw) * C != R:
        raise ValueError(
            f"Row {row.excel_row} ('{row.op}'): a {Nh}x{Nw} patch (PATCH_F, PATCH_T) does "
            f"not reconcile the grid [{Nf}, {Nt}, {C}] with the tokens [{R}, {P}] -- expected "
            f"Nh*Nw == {P}, Nh | {Nf}, Nw | {Nt} and (Nf/Nh)*(Nt/Nw)*C == {R}.  Fix PATCH_F / "
            f"PATCH_T for this model.")
    return {'Nf': Nf, 'Nt': Nt, 'C': C, 'Nh': Nh, 'Nw': Nw}


def mark_pe_layout_rows(rows: List[LayerRow]) -> List[LayerRow]:
    """Tag the model's leading and trailing layout runs to run on the PEs; return them.

    These two are the network's patchify and unpatchify -- an index permutation the cores can
    do themselves, in parallel, inside the stage they already occupy -- so handing them to the
    single DMA channel (what every other layout row still does) would serialise the whole
    cluster behind one core for no reason.

    They are identified by POSITION, not by name: the leading run is necessarily in stage 0
    and the trailing one in the last stage, whatever the splitter decides.  That order matters,
    because tagging changes their latency and the splitter is driven by those latencies -- so
    this must not itself depend on the split.  Call before compute().
    """
    def is_layout(r: LayerRow) -> bool:
        return r.kernel.lower() in LAYOUT_KERNELS

    head: List[LayerRow] = []
    for r in rows:
        if not is_layout(r):
            break
        head.append(r)
    tail: List[LayerRow] = []
    for r in reversed(rows):
        if not is_layout(r):
            break
        tail.append(r)
    tail.reverse()

    tagged: List[LayerRow] = []
    seen: set = set()
    for r, kind in ([(r, 'patchify') for r in head] +
                    [(r, 'unpatchify') for r in tail]):
        if id(r) in seen:      # a model of nothing but layout rows: count each row once
            continue
        seen.add(id(r))
        r.pe_layout = kind
        _patch_geometry(r)     # validate now, before anything is costed on it
        tagged.append(r)
    return tagged


def group_into_blocks(rows: List[LayerRow]) -> List[Block]:
    """Group consecutive rows under their preceding block-header row."""
    blocks: List[Block] = []
    current: Optional[Block] = None
    for r in rows:
        if r.block:
            current = Block(name=r.block, n_rep=r.n_rep)
            blocks.append(current)
        if current is None:
            raise ValueError(f"Row {r.excel_row} has no preceding block header")
        current.rows.append(r)
    return blocks


def split_head_parallel_blocks(blocks: List[Block], n_par: int) -> List[Block]:
    """Carve the head-parallel run out of HEAD_PARALLEL_BLOCK into its own Block.

    The attention core (HEAD_PARALLEL_OPS) is split over heads across n_par clusters,
    which the splitter can only express at BLOCK granularity -- so the block is cut into
    up to three: "<name> (pre)", "<name> (heads)" (n_par > 1) and "<name> (post)".  The
    packer then naturally puts a stage boundary on either side of the parallel run,
    because a block with n_par > 1 is forced to be a stage of its own.

    Cutting a block MID-ROW is only sound when it runs once.  With n_rep > 1, rep k+1's
    pre-rows would have to run on a cluster that already handed its output downstream --
    a cycle in a linear pipeline -- so that is rejected rather than silently mismodelled.
    """
    if n_par <= 1:
        return blocks

    out: List[Block] = []
    for b in blocks:
        if b.name != HEAD_PARALLEL_BLOCK:
            out.append(b)
            continue

        marked = [i for i, r in enumerate(b.rows) if r.op in HEAD_PARALLEL_OPS]
        if not marked:
            out.append(b)
            continue

        first, last = marked[0], marked[-1]
        if marked != list(range(first, last + 1)):
            got = ', '.join(b.rows[i].op for i in marked)
            raise ValueError(
                f"Head-parallel ops must be contiguous inside block {b.name!r}, "
                f"but they are interleaved with other layers (found: {got}). "
                f"Reorder the template or adjust HEAD_PARALLEL_OPS."
            )
        if b.n_rep != 1:
            raise ValueError(
                f"Block {b.name!r} has N_repetitions={b.n_rep}, but it must be 1 to be "
                f"cut mid-block for head-parallel attention: a repeated block split "
                f"across clusters would have to flow backwards through the pipeline. "
                f"Unroll the block in the template, or drop the parallel split."
            )

        for tag, rows, par in (('pre',   b.rows[:first],       1),
                               ('heads', b.rows[first:last+1], n_par),
                               ('post',  b.rows[last+1:],      1)):
            if rows:
                out.append(Block(name=f'{b.name} ({tag})', n_rep=b.n_rep,
                                 rows=list(rows), n_par=par))
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _transfer_ms(num_bytes: float, bandwidth_gbs: Optional[float]) -> float:
    """Transfer time [ms] to move num_bytes at bandwidth_gbs GB/s.

    bandwidth_gbs GB/s == bandwidth_gbs*1e9 B/s, so ms = bytes/(bw*1e9)*1e3
    = bytes/(bw*1e6).  Returns 0.0 when bandwidth is unset/zero (compute-only).
    """
    if not bandwidth_gbs:
        return 0.0
    return num_bytes / (bandwidth_gbs * 1e6)


def split_into_clusters(
    blocks: List[Block],
    row_latencies: Dict[int, float],
    throughput_ms: float,
    bw: Bandwidths = Bandwidths(),
    overlap: bool = False,
) -> List[List[ClusterSegment]]:
    """Greedy left-to-right partition at single-repetition granularity.

    Iterates rep-by-rep through each block (one full pass through the block's
    rows is the atomic unit).  A new cluster is opened when one more pass
    would push the cluster's wall-clock past throughput_ms.  A single oversized
    pass is always emitted as a solo entry (cannot split mid-pass).

    The per-cluster wall-clock budget depends on ``overlap``:

    * ``overlap=False`` (ctt, compute-then-transfer): compute and DMA
      are serialized, so wall = compute + transfer.
    * ``overlap=True`` (compute-while-transfer, double-buffered): compute and
      DMA overlap, so the steady-state wall = max(compute, transfer) -- a
      less conservative budget that generally yields fewer, larger clusters.

    Transfer is the hand-off cost: the prospective last block's data_out, plus (for
    cluster 0 only) the LPDDR data_in load of the very first layer.  Each leg is
    costed at the bandwidth of the link it actually crosses -- see ``bw`` below.
    With both bandwidths unset the transfer term is 0, so both budgets reduce to
    the same compute-only behaviour.

    Returns a list of STAGES.  A stage is normally one cluster, but a head-parallel
    block (n_par > 1) is forced to be a stage of its own spanning n_par clusters.
    """
    # Cluster 0 reads the model's data_in from LPDDR.
    din0_ms = _transfer_ms(blocks[0].rows[0].din_bytes, bw.lpddr) if blocks else 0.0
    clusters: List[List[ClusterSegment]] = []
    current:  List[ClusterSegment] = []
    running = 0.0
    is_first_cluster = True
    for b in blocks:
        # Per-cluster cost of one pass: a head-parallel block spreads each pass over
        # n_par clusters, so each of them carries only 1/n_par of the work.
        block_sum  = sum(row_latencies[r.excel_row] for r in b.rows) / b.n_par

        if b.n_par > 1:
            # Head-parallel: its own stage, regardless of the budget.  Close whatever
            # is open, emit it alone, and start a fresh cluster after it -- this is
            # what puts the hard boundary before and after the parallel run.
            if current:
                clusters.append(current)
                current = []
                running = 0.0
            clusters.append([ClusterSegment(block=b, rep_from=0, rep_to=b.n_rep - 1)])
            is_first_cluster = False
            continue

        # Hand-off to the next cluster = the LAST layer's data_out (e.g. the
        # skip-connection / final output). Intermediate per-row douts (Q.K^T, softmax,
        # ...) are tensors that live in the cluster's L1 and never cross the NoC, so
        # they must NOT be counted. This matches the generator's CLUSTER_DOUT_BYTES.
        is_last_block = b is blocks[-1]

        if b.n_rep == 1:
            # Row-granular: cut BETWEEN layers so a long block (e.g. FFN) can spread over
            # serial pipeline stages instead of landing whole on one cluster.  Each row is
            # an atomic unit; the hand-off if a cut falls after row i is row i's data_out.
            row_start: Optional[int] = None
            for i, row in enumerate(b.rows):
                row_lat = row_latencies[row.excel_row]
                # The model's very last layer writes to LPDDR; every other hand-off crosses
                # the on-chip cluster-to-cluster link.
                is_final_row = is_last_block and i == len(b.rows) - 1
                xfer_ms = _transfer_ms(row.dout_bytes,
                                       bw.lpddr if is_final_row else bw.c2c)
                # stage 0's data_in read is serialized in front of the compute, so it sits
                # OUTSIDE the overlap max (see _overlap_wall).
                din_term      = din0_ms if is_first_cluster else 0.0
                compute_term  = running + row_lat
                transfer_term = xfer_ms + din_term
                prospective   = (din_term + max(compute_term, xfer_ms) if overlap
                                 else compute_term + transfer_term)
                if running > 0 and prospective > throughput_ms:
                    if row_start is not None:
                        current.append(ClusterSegment(block=b, rep_from=0, rep_to=0,
                                                      row_from=row_start, row_to=i - 1))
                        row_start = None
                    clusters.append(current)
                    current = []
                    running = 0.0
                    is_first_cluster = False
                if row_start is None:
                    row_start = i
                running += row_lat
            if row_start is not None:
                current.append(ClusterSegment(block=b, rep_from=0, rep_to=0,
                                              row_from=row_start, row_to=len(b.rows) - 1))
            continue

        # n_rep > 1: split by whole passes (rep granularity); a block is never cut mid-row
        # when it repeats, since a later rep's rows can't flow back to an earlier cluster.
        dout       = b.rows[-1].dout_bytes
        xfer_c2c   = _transfer_ms(dout, bw.c2c)
        xfer_lpddr = _transfer_ms(dout, bw.lpddr)
        rep_start: Optional[int] = None
        for rep in range(b.n_rep):
            is_final_pass = is_last_block and rep == b.n_rep - 1
            xfer_ms       = xfer_lpddr if is_final_pass else xfer_c2c
            # prospective wall-clock if this pass joins the current cluster and
            # the cluster ends here: compute so far + this pass, against its
            # hand-off transfer (overlap=max, otherwise serialized=sum).
            din_term      = din0_ms if is_first_cluster else 0.0
            compute_term  = running + block_sum
            transfer_term = xfer_ms + din_term
            prospective   = (din_term + max(compute_term, xfer_ms) if overlap
                             else compute_term + transfer_term)
            if running > 0 and prospective > throughput_ms:
                # close partial segment for this block before cutting
                if rep_start is not None:
                    current.append(ClusterSegment(block=b, rep_from=rep_start, rep_to=rep - 1))
                    rep_start = None
                clusters.append(current)
                current = []
                running = 0.0
                is_first_cluster = False
            if rep_start is None:
                rep_start = rep
            running += block_sum
        if rep_start is not None:
            current.append(ClusterSegment(block=b, rep_from=rep_start, rep_to=b.n_rep - 1))
    if current:
        clusters.append(current)
    return clusters


# ---------------------------------------------------------------------------
# Exact stage splitting (dynamic programming)
# ---------------------------------------------------------------------------
#
# The greedy above answers "how many stages does budget T need?"; binary-searching T
# over it only minimises the bottleneck when the per-stage cost is monotone in the
# layers assigned and equals the objective.  Neither holds here: a stage's transfer is
# its LAST layer's data_out, so appending a layer can make a stage CHEAPER, and the ctt
# period is max_s(compute) + max_s(transfer) -- two barriered substages -- not the
# per-stage max the greedy packs against.  The DP below is exact under both objectives,
# needs no monotonicity, and is cheap at these sizes (tens of units, <= ~10 stages).


@dataclass
class _Unit:
    """An indivisible piece of the layer chain -- a cut may only fall between units.

    Same granularity the greedy uses: one unit per ROW for an n_rep==1 block (so a long
    block can spread over serial stages), one per REP for a repeated block, and the whole
    head-parallel block as a single unit that must occupy a stage of its own.
    """
    block: Block
    kind:  str          # 'par' | 'row' | 'rep'
    lo:    int          # rep index, or row index for kind == 'row'
    hi:    int

    @property
    def n_reps(self) -> int:
        return self.hi - self.lo + 1 if self.kind != 'row' else 1

    @property
    def rows(self) -> List[LayerRow]:
        if self.kind == 'row':
            return self.block.rows[self.lo:self.hi + 1]
        return self.block.rows


def _atomic_units(blocks: List[Block]) -> List[_Unit]:
    """The layer chain as a flat list of indivisible units, in execution order."""
    units: List[_Unit] = []
    for b in blocks:
        if b.n_par > 1:
            units.append(_Unit(b, 'par', 0, b.n_rep - 1))
        elif b.n_rep == 1:
            units.extend(_Unit(b, 'row', i, i) for i in range(len(b.rows)))
        else:
            units.extend(_Unit(b, 'rep', r, r) for r in range(b.n_rep))
    return units


def _units_to_segments(units: List[_Unit]) -> List[ClusterSegment]:
    """Coalesce a stage's units back into ClusterSegments, merging runs of one block."""
    segs: List[ClusterSegment] = []
    run: Optional[_Unit] = None
    for u in units:
        if run is not None and u.block is run.block and u.kind == run.kind:
            run = _Unit(run.block, run.kind, run.lo, u.hi)
            continue
        if run is not None:
            segs.append(_segment_of(run))
        run = u
    if run is not None:
        segs.append(_segment_of(run))
    return segs


def _segment_of(u: _Unit) -> ClusterSegment:
    if u.kind == 'row':
        return ClusterSegment(block=u.block, rep_from=0, rep_to=0,
                              row_from=u.lo, row_to=u.hi)
    return ClusterSegment(block=u.block, rep_from=u.lo, rep_to=u.hi)


def _unit_stage_cost(units: List[_Unit], is_first: bool, is_last: bool,
                     row_lat: Dict[int, float], bw: Bandwidths
                     ) -> Tuple[float, float, float]:
    """(compute_ms, transfer_ms, din_ms) of one stage -- the same formulas
    _build_cluster_splits applies, so the DP optimises exactly what _makespan later
    reports.  din_ms is the stage-0 LPDDR data_in read, already inside transfer_ms and
    returned apart because cwt cannot overlap it (see _overlap_wall)."""
    n_par   = max(u.block.n_par for u in units)
    compute = sum(u.n_reps * sum(row_lat[r.excel_row] for r in u.rows)
                  for u in units) / n_par
    last_r  = units[-1].rows[-1]
    xfer    = _transfer_ms(last_r.dout_bytes / n_par,
                           bw.lpddr if is_last else bw.c2c)
    din     = _transfer_ms(units[0].rows[0].din_bytes, bw.lpddr) if is_first else 0.0
    return compute, xfer + din, din


def _dp_partition(units: List[_Unit], k_stages: int,
                  value: List[List[float]], legal: List[List[bool]],
                  ) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """Minimise the MAX per-stage `value` over contiguous partitions into k_stages.

    value[j][i] is the stage spanning units[j:i]; legal[j][i] gates it (the
    head-parallel unit must sit alone).  Returns (minimax, [(lo, hi) per stage]).
    """
    n   = len(units)
    INF = float('inf')
    # f[k][i] = best achievable max-value covering units[:i] with k stages
    f    = [[INF] * (n + 1) for _ in range(k_stages + 1)]
    prev = [[-1]  * (n + 1) for _ in range(k_stages + 1)]
    f[0][0] = 0.0
    for k in range(1, k_stages + 1):
        for i in range(1, n + 1):
            best, arg = INF, -1
            for j in range(k - 1, i):
                if f[k - 1][j] == INF or not legal[j][i]:
                    continue
                cand = max(f[k - 1][j], value[j][i])
                if cand < best:
                    best, arg = cand, j
            f[k][i], prev[k][i] = best, arg
    if f[k_stages][n] == INF:
        return INF, None
    bounds, i = [], n
    for k in range(k_stages, 0, -1):
        j = prev[k][i]
        bounds.append((j, i))
        i = j
    return f[k_stages][n], list(reversed(bounds))


def _opt_overlap(units: List[_Unit], k: int, comp, xfer, din, legal
                 ) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """cwt optimum: the period is max_s(wall_s) -- a plain minimax, so one DP pass over
    the per-stage wall solves it exactly.  wall_s puts stage 0's data_in read outside the
    overlap max (see _overlap_wall), which is why `din` is carried alongside."""
    n    = len(units)
    wall = [[_overlap_wall_terms(comp[j][i], xfer[j][i], din[j][i])
             for i in range(n + 1)] for j in range(n + 1)]
    return _dp_partition(units, k, wall, legal)


def _opt_serial(units: List[_Unit], k: int, comp, xfer, din, legal
                ) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
    """ctt optimum: the period is max_s(compute_s) + max_s(transfer_s), which no single
    minimax captures.  Every stage's compute is one of O(n^2) achievable sums, so sweep
    those as a cap C in increasing order, minimise the max TRANSFER subject to it, and
    score the resulting partition honestly (its own max compute may sit below C).  C is
    a lower bound on any period reachable under it, so the sweep stops once C exceeds
    the best found.  `din` is unused here: ctt serializes the data_in read into the
    transfer substage, so it is already inside xfer where this model wants it."""
    n = len(units)
    best_period, best_bounds = float('inf'), None
    caps = sorted({comp[j][i] for j in range(n) for i in range(j + 1, n + 1)
                   if legal[j][i]})
    for cap in caps:
        if cap >= best_period:
            break
        capped = [[xfer[j][i] if comp[j][i] <= cap + 1e-15 else float('inf')
                   for i in range(n + 1)] for j in range(n + 1)]
        _, bounds = _dp_partition(units, k, capped, legal)
        if not bounds:
            continue
        period = (max(comp[j][i] for j, i in bounds)
                  + max(xfer[j][i] for j, i in bounds))
        if period < best_period:
            best_period, best_bounds = period, bounds
    return best_period, best_bounds


def best_split(blocks: List[Block], row_lat: Dict[int, float], bw: Bandwidths,
               overlap: bool, k_stages: int) -> Tuple[float, List[ClusterSplit]]:
    """Exactly minimise the pipeline period over all splits into k_stages stages.

    Optimises for the requested wall model, then breaks TIES by the other model: several
    partitions usually reach the same optimal period, and since both modes have to run
    this one split, the tie-break decides how good the other mode's schedule is.  It is
    done by restricting the stages to those the optimum already tolerates and optimising
    the other objective inside that set -- so the primary period can never get worse.

    Returns (period, splits) or (inf, []) when k_stages admits no legal partition.
    """
    units = _atomic_units(blocks)
    n     = len(units)
    if k_stages < 1 or k_stages > n:
        return float('inf'), []

    par_idx = {i for i, u in enumerate(units) if u.kind == 'par'}
    legal   = [[False] * (n + 1) for _ in range(n + 1)]
    comp    = [[0.0] * (n + 1) for _ in range(n + 1)]
    xfer    = [[0.0] * (n + 1) for _ in range(n + 1)]
    din     = [[0.0] * (n + 1) for _ in range(n + 1)]
    for j in range(n):
        for i in range(j + 1, n + 1):
            hit = par_idx & set(range(j, i))
            # a head-parallel unit owns its stage: alone, or not in it at all
            legal[j][i] = (not hit) or (i - j == 1)
            if not legal[j][i]:
                continue
            comp[j][i], xfer[j][i], din[j][i] = _unit_stage_cost(
                units[j:i], j == 0, i == n, row_lat, bw)

    primary = _opt_overlap if overlap else _opt_serial
    period, bounds = primary(units, k_stages, comp, xfer, din, legal)
    if not bounds:
        return float('inf'), []

    # Tie-break.  Keep only the stages the winning split already lives with -- for cwt
    # every wall it allows, for ctt its own compute and transfer maxima -- then optimise
    # the OTHER model within them.  Any partition built from those stages has a primary
    # period no worse than `period`, so this is free.
    eps = 1e-12
    if overlap:
        keep = [[legal[j][i]
                 and _overlap_wall_terms(comp[j][i], xfer[j][i], din[j][i]) <= period + eps
                 for i in range(n + 1)] for j in range(n + 1)]
        _, alt = _opt_serial(units, k_stages, comp, xfer, din, keep)
    else:
        cap_c = max(comp[j][i] for j, i in bounds)
        cap_t = max(xfer[j][i] for j, i in bounds)
        keep  = [[legal[j][i] and comp[j][i] <= cap_c + eps
                  and xfer[j][i] <= cap_t + eps
                  for i in range(n + 1)] for j in range(n + 1)]
        _, alt = _opt_overlap(units, k_stages, comp, xfer, din, keep)
    if alt:
        bounds = alt

    splits = _build_cluster_splits(
        [_units_to_segments(units[j:i]) for j, i in bounds], row_lat, bw)
    return _makespan(splits, overlap), splits


def _build_cluster_splits(
    cluster_groups: List[List[ClusterSegment]],
    row_lat: Dict[int, float],
    bw: Bandwidths,
) -> List[ClusterSplit]:
    """Turn a greedy cluster grouping into ClusterSplit metadata records.

    Shared by both the serialized and overlapped splits: derives per-cluster
    compute latency, hand-off transfer time, first/last layer byte sizes and
    the L1 footprints.  The grouping itself decides where the cuts fall; this
    only annotates it.

    Unlike the greedy, this sees the whole grouping, so it knows which cluster is
    last and charges every transfer to the link it really crosses.
    """
    n_clusters = len(cluster_groups)
    splits: List[ClusterSplit] = []
    for idx, segs in enumerate(cluster_groups):
        all_rows    = [r for seg in segs for r in seg.rows]
        # Clusters running this stage in parallel.  Only a head-parallel stage has
        # n_par > 1, and the splitter guarantees it holds exactly that one block.
        n_par       = max(seg.block.n_par for seg in segs)
        grp_latency = sum(
            seg.n_reps * sum(row_lat[r.excel_row] for r in seg.rows)
            for seg in segs
        ) / n_par
        max_w    = max(r.weights_bytes for r in all_rows)
        max_din  = max(r.din_bytes     for r in all_rows)
        max_dout = max(r.dout_bytes    for r in all_rows)
        # Each cluster of a head-parallel stage holds one head: 1/n_par of the tensors.
        l1_ctt   = (max_w + max_din + max_dout) / n_par
        first_r  = segs[0].rows[0]
        last_r   = segs[-1].rows[-1]
        # hand-off bytes: the stage's pipeline output is its last layer's data_out
        # (== cs.last_dout, == the generator's STAGE_DOUT_BYTES).  The LAST stage
        # writes it out to LPDDR; every other stage hands it to its successor over
        # the on-chip link.  Plus, on stage 0 only, the LPDDR data_in read of the
        # first layer.  A head-parallel stage's n_par clusters each ship their own
        # 1/n_par slice concurrently, so the stage's wall is the per-cluster leg.
        out_bw   = bw.lpddr if idx == n_clusters - 1 else bw.c2c
        xfer_ms  = _transfer_ms(last_r.dout_bytes / n_par, out_bw)
        din_ms   = _transfer_ms(first_r.din_bytes, bw.lpddr) if idx == 0 else 0.0
        xfer_ms += din_ms
        splits.append(ClusterSplit(
            idx                       = idx,
            segments                  = segs,
            latency                   = grp_latency,
            transfer_ms               = xfer_ms,
            din_ms                    = din_ms,
            first_weights             = first_r.weights_bytes,
            first_din                 = first_r.din_bytes,
            last_weights              = last_r.weights_bytes,
            last_dout                 = last_r.dout_bytes,
            l1_compute_then_transfer  = l1_ctt,
            l1_compute_while_transfer = 2 * l1_ctt,
            n_par                     = n_par,
        ))
    return splits


def _makespan(splits: List[ClusterSplit], overlap: bool) -> float:
    """Steady-state period of one pipeline step = what throughput is 1/of.

    Both generated loops rendezvous every cluster on a global barrier, so a step costs
    the slowest thing running inside each barriered phase -- and the two modes differ in
    how many phases a step has:

    * ``overlap=False`` (ctt): the loop is compute -> barrier -> transfer -> barrier, i.e.
      TWO SUBSTAGES.  No cluster transfers while another computes, so the step is the
      slowest COMPUTE substage plus the slowest TRANSFER substage, each taken over the
      whole split:  max_s(compute_s) + max_s(transfer_s).
    * ``overlap=True`` (cwt): double buffering lets a cluster push item m-1 while it
      computes item m, and there is one barrier per step, so the step is the SLOWEST
      STAGE's own wall:  max_s(wall_s).

      wall_s = max(compute_s, dout_s) for every stage but the first.  Stage 0 also reads
      the model's data_in from LPDDR, and that read does NOT overlap anything: the
      generated loop issues it with the blocking dma_2d and then holds the worker cores
      on a flex_intra_cluster_sync() until it lands, so it is serialized IN FRONT of the
      compute.  Only the outbound hand-off overlaps.  Hence
      wall_0 = din_0 + max(compute_0, dout_0), NOT max(compute_0, din_0 + dout_0) --
      the latter hides an exposed LPDDR read behind the compute and under-reports the
      period whenever stage 0 is the bottleneck.

    The ctt period is >= the naive per-stage max_s(compute_s + transfer_s) whenever the
    slowest compute and the slowest transfer sit on DIFFERENT stages: the barrier between
    the phases stops them covering for each other.
    """
    if not splits:
        return 0.0
    if overlap:
        return max(_overlap_wall(cs) for cs in splits)
    return (max(cs.latency       for cs in splits)
            + max(cs.transfer_ms for cs in splits))


def _overlap_wall_terms(compute: float, transfer: float, din: float) -> float:
    """cwt wall [ms] of one stage from its raw terms (`transfer` INCLUDES `din`).

    din is 0 on every stage but the first, where this collapses to the plain
    max(compute, transfer).  See _makespan for why stage 0 differs.
    """
    return din + max(compute, transfer - din)


def _overlap_wall(cs: ClusterSplit) -> float:
    """cwt wall [ms] of one built stage -- _overlap_wall_terms on a ClusterSplit."""
    return _overlap_wall_terms(cs.latency, cs.transfer_ms, cs.din_ms)


def _pipeline_clusters(cfg: dict) -> int:
    """Read (and sanity-check) the single hard cluster budget from the config."""
    # These two used to set the cluster count between them.  A config still carrying
    # either would now be silently ignored, so refuse it instead.
    stale = [k for k in ('attention_clusters', 'extra_pipeline_stages') if k in cfg]
    if stale:
        raise ValueError(
            f'Config key(s) {", ".join(stale)} were replaced by the single hard '
            f'"pipeline_clusters" (total clusters the pipeline occupies; the '
            f'head-parallel attention stage takes {HEAD_PARALLEL_CLUSTERS} of them). '
            f'Remove them from the config.')
    if 'pipeline_clusters' not in cfg:
        raise ValueError(
            'Config is missing "pipeline_clusters": the total number of clusters the '
            'pipeline occupies, of which the head-parallel attention stage takes '
            f'{HEAD_PARALLEL_CLUSTERS}.')
    n = int(cfg['pipeline_clusters'])
    if n < HEAD_PARALLEL_CLUSTERS:
        raise ValueError(
            f'"pipeline_clusters": {n} is below the {HEAD_PARALLEL_CLUSTERS} clusters '
            f'the head-parallel attention stage alone needs.')
    return n


def compute(layers: List[LayerRow], cfg: dict,
            split_model: str = 'ctt') -> Results:
    """Model the app and split it into pipeline stages.

    split_model picks which wall model the stage boundaries are optimized for -- 'ctt'
    (compute + transfer) or 'cwt' (max(compute, transfer)).  Both modes then RUN that
    same split: the layer-to-stage assignment is shared, only the schedule differs.
    """
    pe_util = cfg['PE_utilization']
    te_util = cfg['TE_utilization']
    pools   = list(cfg['PE_peak'].keys())
    res     = Results()
    if split_model not in ('ctt', 'cwt'):
        raise ValueError(f"split_model must be 'ctt' or 'cwt', got {split_model!r}")
    ref_overlap = split_model == 'cwt'

    # K1:M3 -- FLOPs/ms.  Each peak is derated by the utilization of the engine that
    # actually runs it: PE_utilization on the PE path (non-GEMM), and TE_utilization on
    # a GEMM only where a tensor engine exists.  A pool without one runs its GEMMs as PE
    # matmuls, so TE_utilization does not enter its numbers at all.
    for p in pools:
        gemm_util = te_util if p in TENSOR_ENGINE_POOLS else pe_util
        res.flops_per_ms[p] = {
            'non_GEMM': cfg['PE_peak'][p]   * pe_util   * 1e9,
            'GEMM':     cfg['GEMM_peak'][p] * gemm_util * 1e9,
        }

    # Y5:Z53 -- per-row, per-block and total latency [ms].  Compute layers cost
    # flops/rate.  The only layout rows still in the model are the PE-run bookends
    # (patchify / unpatchify): they carry 0 FLOPs but occupy the cores, so they are charged
    # on the PE path -- PE_LAYOUT_FLOPS_PER_ELEM per element at the pool's non-GEMM rate,
    # plus the PE_LAYOUT_FIXED_CYCLES entry cost, which is a latency and so does NOT scale
    # with the pool's throughput.  Every other layout row rides a stage hand-off and was
    # dropped before costing -- it moves no bytes of its own, so there is nothing to charge.
    fixed_ms = lambda cyc: cyc / (cfg.get('frequency', 1_000_000_000) / 1e3)

    def _row_lat(r: LayerRow, rate_g: float, rate_ng: float) -> float:
        if r.kernel.lower() in LAYOUT_KERNELS:
            if not r.pe_layout:
                raise ValueError(
                    f"Row {r.excel_row} ('{r.op}') is a layout row that is neither a PE "
                    f"bookend nor folded into a stage hand-off.  Costing it would mean "
                    f"charging for a local L1->L1 copy the generated app does not emit -- "
                    f"fold it into the neighbouring hand-off or give it a PE kernel.")
            elems = r.dout_bytes / BYTES_PER_ELEM
            return (fixed_ms(PE_LAYOUT_FIXED_CYCLES[r.pe_layout])
                    + elems * PE_LAYOUT_FLOPS_PER_ELEM / rate_ng)
        return r.flops / (rate_g if r.is_gemm else rate_ng)

    for p in pools:
        rate_g  = res.flops_per_ms[p]['GEMM']
        rate_ng = res.flops_per_ms[p]['non_GEMM']
        res.row_latencies[p] = {
            r.excel_row: _row_lat(r, rate_g, rate_ng)
            for r in layers
        }

    blocks = group_into_blocks(layers)
    res.blocks = blocks
    for p in pools:
        res.block_latencies[p] = []
        total = 0.0
        for b in blocks:
            block_sum = sum(res.row_latencies[p][r.excel_row] for r in b.rows)
            contrib   = block_sum * b.n_rep
            res.block_latencies[p].append({
                'name':     b.name,
                'n_rep':    b.n_rep,
                'sum_rows': block_sum,
                'subtotal': contrib,
            })
            total += contrib
        res.total_latency[p] = total

    # Pipeline cluster splitting.  ONE split per pool, shared by both modes: ctt and
    # cwt must run the same layers on the same stages, so that comparing them shows the
    # schedule's effect and not a different partitioning.  The cuts are optimized under
    # the reference wall model (split_model) and the split is then evaluated under both.
    bw = Bandwidths(c2c   = cfg.get('bandwidth_c2c_gbs'),
                    lpddr = cfg.get('bandwidth_lpddr_gbs'))
    # Carve the head-parallel attention core into its own block so the splitter
    # can give it a stage of n_par clusters.  Only the PIPELINE view is reshaped;
    # the latency tables above still report the model's real blocks.
    n_par = HEAD_PARALLEL_CLUSTERS
    pipe_blocks = split_head_parallel_blocks(blocks, n_par)
    # The cluster budget is HARD: config.json names the total, and since exactly one
    # stage is head-parallel (n_par clusters) and every other is one cluster, that
    # total fixes the stage count outright.  The optimizer only places the cuts.
    n_clusters = _pipeline_clusters(cfg)
    n_stages   = n_clusters - n_par + 1
    for p in pools:
        row_lat = res.row_latencies[p]
        _, balanced = best_split(
            pipe_blocks, row_lat, bw, ref_overlap, n_stages)
        if not balanced:
            # A bound large enough that the greedy cuts ONLY where the head-parallel
            # stage forces it gives the minimum legal stage count for this model.
            budget = res.total_latency[p] * 2.0 + 1.0
            n_base = len(split_into_clusters(
                pipe_blocks, row_lat, budget, bw, ref_overlap))
            n_max  = len(_atomic_units(pipe_blocks))
            raise ValueError(
                f'"pipeline_clusters": {n_clusters} admits no legal split of this model '
                f'on {p}: {n_stages} stages is outside the feasible range '
                f'{n_base}..{n_max}.  Set pipeline_clusters between '
                f'{n_base + n_par - 1} and {n_max + n_par - 1}.')
        # Same stages for both modes; only the wall model differs, so each mode's
        # period is that model's makespan of this one split.  The mode the split was
        # NOT optimized for therefore reports its true period on these cuts, which is
        # the point -- it is no longer free to move the layer boundaries.
        res.clusters[p]                = balanced
        res.clusters_overlap[p]        = balanced
        res.best_throughput[p]         = _makespan(balanced, False)
        res.best_throughput_overlap[p] = _makespan(balanced, True)

    # T52:W54 -- L1 footprint (bytes)
    max_w    = max(r.weights_bytes for r in layers)
    max_din  = max(r.din_bytes     for r in layers)
    max_dout = max(r.dout_bytes    for r in layers)
    res.l1_compute_then_transfer  = max_w + max_din + max_dout
    res.l1_compute_while_transfer = 2 * res.l1_compute_then_transfer

    return res


# ---------------------------------------------------------------------------
# Stage <-> physical cluster mapping
# ---------------------------------------------------------------------------

def _stage_bases(stages: List[ClusterSplit]) -> List[int]:
    """First physical cluster id of each stage.

    Stages are packed back-to-back over the cluster ids: an ordinary stage takes one
    cluster, a head-parallel stage takes n_par.  So a 1/4/1 pipeline occupies clusters
    0 | 1,2,3,4 | 5 -- six in total, with bases [0, 1, 5].
    """
    bases: List[int] = []
    cid = 0
    for cs in stages:
        bases.append(cid)
        cid += cs.n_par
    return bases


def seg_label(seg: ClusterSegment) -> str:
    """Block name, annotated with its row range when the segment is a partial block.

    A block split mid-rows across serial stages (e.g. FFN over two clusters) shows as
    `FFN[rows 1-4]` / `FFN[rows 5-7]` so the split is legible in tables and C comments.
    """
    if not seg.is_partial:
        return seg.block.name
    return f'{seg.block.name}[rows {seg.row_from + 1}-'\
           f'{(seg.row_to if seg.row_to is not None else len(seg.block.rows) - 1) + 1}]'


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_cluster_table(pool: str, clusters: List[ClusterSplit],
                         row_lat: Dict[int, float], *, overlap: bool,
                         best_throughput: float) -> None:
    """Print the per-stage split table for one pool under one wall model.

    The two modes share one split (same layers, same stages), so printing both tables
    shows the SAME stage contents twice: what differs is the wall charged to each stage
    and hence the period.  overlap selects that -- max(compute, transfer) for the
    compute-while-transfer (cwt) schedule, else compute + transfer (ctt).
    best_throughput is the pipeline PERIOD of this split under that model (see
    ``_makespan``: one max over stages for cwt, the sum of the two substage maxima for
    ctt); it is printed prominently at the head of the table.

    The cwt wall of stage 0 carries its LPDDR data_in read OUTSIDE the max (the generated
    loop serializes that read before the compute), so its printed wall exceeds
    max(compute, transfer) -- see ``_overlap_wall``.
    """
    label = ('Overlapped  (cwt: compute-while-transfer)' if overlap
             else 'Serialized  (ctt: compute-then-transfer)')
    model  = ('max(compute, transfer)   /* stage 0: data_in + max(compute, dout) */'
              if overlap else 'compute + transfer')
    period = ('max_s(wall)   /* the slowest stage */' if overlap
              else 'max_s(compute) + max_s(transfer)   /* two barriered substages */')
    n_stages = len(clusters)
    bases    = _stage_bases(clusters)
    n_phys   = sum(cs.n_par for cs in clusters)
    print(f'\nPool: {pool}  --  {label}')
    print(f'  >>> Pipeline period = {best_throughput:.6g} ms  '
          f'(bottleneck of the shared {n_stages}-stage split)')
    print(f'  {n_stages} stage(s)  |  {n_phys} physical cluster(s)  |  '
          f'stage wall = {model}  |  period = {period}')
    for cs, base in zip(clusters, bases):
        print('  ' + '=' * 78)
        seg_desc = ', '.join(
            f'{seg_label(seg)} rep {seg.rep_from+1}-{seg.rep_to+1}/{seg.block.n_rep}'
            for seg in cs.segments
        )
        wall = _overlap_wall(cs) if overlap else cs.latency + cs.transfer_ms
        where = (f'clusters {base}-{base + cs.n_par - 1}' if cs.n_par > 1
                 else f'cluster {base}')
        print(f'  Stage {cs.idx} [{where}]  |  compute = {cs.latency:.6g} ms  |  '
              f'transfer = {cs.transfer_ms:.6g} ms  |  wall = {wall:.6g} ms'
              + ('   (per cluster)' if cs.n_par > 1 else ''))
        if cs.n_par > 1:
            print(f'  {"":11}>>> HEAD-PARALLEL: {cs.n_par} clusters in parallel, '
                  f'1 head each -- compute and L1 divided by {cs.n_par}')
        print(f'  {"":11}{seg_desc}')
        print(f'  {"":4}{"Block":<16}{"Reps":<8}{"Row":<6}{"Op":<26}{"":6}{"lat/rep [ms]":>14}{"×N":>4}{"total [ms]":>12}')
        print('  ' + '-' * 78)
        for seg in cs.segments:
            rep_tag = f'{seg.rep_from+1}-{seg.rep_to+1}' if seg.n_reps > 1 else f'{seg.rep_from+1}'
            for r in seg.rows:
                tag     = ' GEMM' if r.is_gemm else '     '
                lat     = row_lat[r.excel_row] / cs.n_par   # per-cluster share
                tot_lat = lat * seg.n_reps
                print(f'  {"":4}{seg.block.name[:16]:<16}{rep_tag:<8}{r.excel_row:<6}'
                      f'{r.op[:26]:<26}{tag}  {lat:>14.6g}{seg.n_reps:>4}{tot_lat:>12.6g}')
        print('  ' + '-' * 78)
        # Every byte figure below is PER CLUSTER: a head-parallel stage's clusters each
        # take in, and hand off, only their own 1/n_par slice.
        din_each  = cs.first_din / cs.n_par
        dout_each = cs.last_dout / cs.n_par
        each      = '  (per cluster)' if cs.n_par > 1 else ''
        print(f'  {"":4}First layer : weights={cs.first_weights:>12,.0f} B'
              f'  data_in ={din_each:>12,.0f} B{each}')
        if cs.idx < n_stages - 1:
            print(f'  {"":4}Last  layer : weights={cs.last_weights:>12,.0f} B'
                  f'  data_out={dout_each:>11,.0f} B{each}  --> pipeline to stage {cs.idx+1}')
        else:
            print(f'  {"":4}Last  layer : weights={cs.last_weights:>12,.0f} B'
                  f'  data_out={dout_each:>11,.0f} B{each}')
        print(f'  {"":4}Max L1 compute-then-transfer  = {cs.l1_compute_then_transfer:>12,.0f} B{each}')
        print(f'  {"":4}Max L1 compute-while-transfer = {cs.l1_compute_while_transfer:>12,.0f} B{each}')
    print('  ' + '=' * 74)


def print_results(layers: List[LayerRow], res: Results,
                  mode: Optional[str] = None) -> None:
    pools = list(res.flops_per_ms.keys())

    print('#' * 78)
    print('FLOPs/ms  (achievable FLOPs per millisecond)')
    print('#' * 78)
    print(f'{"":<14}{"non-GEMM":>20}{"GEMM":>20}')
    for pool, vals in res.flops_per_ms.items():
        print(f'{pool:<14}{vals["non_GEMM"]:>20,.0f}{vals["GEMM"]:>20,.0f}')

    print()
    print('#' * 78)
    print('L1 footprint [bytes]')
    print('#' * 78)
    print(f'Compute then transfer :  {res.l1_compute_then_transfer:>15,.0f}')
    print(f'Compute while transfer:  {res.l1_compute_while_transfer:>15,.0f}')

    print()
    print('#' * 78)
    print('Latency [ms] per pool')
    print('#' * 78)
    header = f'{"Row":<5}{"Block":<16}{"Op":<20}{"":<5}' + ''.join(f'{p:>16}' for p in pools)
    print(header)
    print('-' * len(header))

    blocks = res.blocks
    for b in blocks:
        for r in b.rows:
            tag  = 'GEMM' if r.is_gemm else '   .'
            vals = ''.join(f'{res.row_latencies[p][r.excel_row]:>16.6g}' for p in pools)
            print(f'{r.excel_row:<5}{b.name:<16}{r.op[:20]:<20}{tag:<5}{vals}')
        subtotal = ''.join(
            f'{next(x for x in res.block_latencies[p] if x["name"] == b.name)["subtotal"]:>16.6g}'
            for p in pools
        )
        print(f'{"":<5}{"":<16}{f"-> x{b.n_rep} subtotal":<25}{subtotal}')
        print()

    print('-' * len(header))
    totals = ''.join(f'{res.total_latency[p]:>16.6g}' for p in pools)
    print(f'{"":<5}{"":<16}{"Tot min latency":<25}{totals}')

    if res.clusters:
        print()
        print('#' * 78)
        print('Pipeline cluster split')
        print('#' * 78)
        # Show only the split relevant to the chosen --mode: serialized
        # (compute + transfer) for ctt, overlapped (max(compute, transfer))
        # for cwt.  With no --mode show both for inspection.
        show_serial  = mode in (None, 'ctt')
        show_overlap = mode in (None, 'cwt')
        for p in pools:
            if show_serial and p in res.clusters:
                _print_cluster_table(p, res.clusters[p],
                                     res.row_latencies[p], overlap=False,
                                     best_throughput=res.best_throughput[p])
            if show_overlap and p in res.clusters_overlap:
                _print_cluster_table(p, res.clusters_overlap[p],
                                     res.row_latencies[p], overlap=True,
                                     best_throughput=res.best_throughput_overlap[p])


# ---------------------------------------------------------------------------
# Shared code renderers
# ---------------------------------------------------------------------------

def _render_dma_helper(bench: bool) -> str:
    """Shared C: the blocking 1D-DMA helper.

    Reused verbatim by the ctt and cwt pipeline generators.

    bench=True brackets the transfer with mempool_start_benchmark()/mempool_stop_benchmark()
    (the `trace` CSR), so an RTL run traces the DMA and nothing around it.  The markers are
    two CSR writes -- no counter read, no printf, nothing that perturbs the region the way
    the old mcycle log did.  bench=False (default) emits the bare transfer.
    """
    head = [
        '/* Blocking DMA: issues the whole transfer as a single flex_dma_async_2d and',
        ' * waits for it to land.  Every transfer in this app goes through the 2D engine,',
        ' * which is what lets a hand-off carry a re-layout in its strides instead of',
        ' * needing a local copy: `reps` runs of `chunk` bytes, walking the source by',
        ' * src_stride and the destination by dst_stride. */',
        'static inline void dma_2d_strided(uint64_t dst, uint64_t src, uint32_t chunk,',
        '                                  uint32_t dst_stride, uint32_t src_stride,',
        '                                  uint32_t reps)',
        '{',
    ]
    if bench:
        head.append('    mempool_start_benchmark();')
    head += [
        '    flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);',
        '    flex_dma_async_wait_all();',
    ]
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
    return '\n'.join(head)


# ---------------------------------------------------------------------------
# Pipeline (ctt/cwt) per-layer compute helpers
# ---------------------------------------------------------------------------

def _layer_cycles(row: 'LayerRow', row_lat: Dict[int, float],
                  cycles_per_ms: int) -> int:
    """Per-layer wait-loop count (>=1) from the row's latency [ms]."""
    return max(1, int(row_lat.get(row.excel_row, 0.0) * cycles_per_ms))


def _c_comment(text: str) -> str:
    return text.replace('*/', '* /')


def _render_compute_body(cluster: 'ClusterSplit', row_lat: Dict[int, float],
                         cycles_per_ms: int) -> List[str]:
    """One `mempool_wait` per layer, block reps wrapped in a `for` loop.

    On a head-parallel stage each cluster owns one head, so every layer of the stage
    costs 1/n_par of its full latency.
    """
    lines: List[str] = []
    if cluster.n_par > 1:
        lines.append(f'    /* head-parallel: this cluster owns 1 of {cluster.n_par} heads, '
                     f'so every wait below is 1/{cluster.n_par} of the layer\'s full cost */')
    for seg in cluster.segments:
        b   = seg.block
        rng = (f'{seg.rep_from+1}-{seg.rep_to+1}/{b.n_rep}'
               if seg.n_reps > 1 else f'{seg.rep_from+1}/{b.n_rep}')
        label = _c_comment(seg_label(seg))
        lines.append(f'    /* {label} x{seg.n_reps}  (reps {rng}) */')
        lines.append(f'    for (uint32_t r = 0; r < {seg.n_reps}u; r++) {{')
        for row in seg.rows:
            cyc = max(1, _layer_cycles(row, row_lat, cycles_per_ms) // cluster.n_par)
            tag = ' (GEMM)' if row.is_gemm else ''
            lines.append(f'        mempool_wait({cyc}UL);  '
                         f'/* {label}: {_c_comment(row.op)}{tag} */')
        lines.append('    }')
    return lines


# ---------------------------------------------------------------------------
# Real-kernel dispatch (default; --wait selects _render_compute_body above)
# ---------------------------------------------------------------------------
#
# Every layer is lowered to a real baremetal kernel call operating on three L1
# scratch buffers X_BUF / W_BUF / Y_BUF (placeholder data -- these kernels' cycle
# counts are data-independent).  The M/N/P dims come from the spreadsheet's
# i_*/o_* columns.  On a head-parallel stage each cluster owns one head, so the
# outermost multiplicity (matmul batch / softmax-layernorm rows / elementwise size /
# layout bytes) is divided by n_par -- the same 1/n_par split _render_compute_body
# applies to its wait counts.

# Kernel headers pulled in only in real-kernel mode (single TU, so the non-static
# kernel definitions cause no multiple-definition trouble).  builtins_v2.h first,
# because mempool_relu_f16.h uses v2h without including it itself.
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

# The patch permutation kernels the generated app carries ITSELF.  They live here rather
# than in software/kernels/baremetal/ because they are specific to this model's patch
# geometry and nothing else in the tree calls them: emitting them into main.c is what
# makes a generated app self-contained, needing only the shared baremetal headers above.
# Both are pure index permutations, and the strided side uses the Xpulp post-increment
# form (p.lh / p.sh, `xpulppostmod`) -- note the `!`, without which the assembler picks
# the encoding that does NOT write the address register back.
_RESHAPE_KERNELS = r"""


/* Patch extraction / re-assembly for a vision-transformer front end, on the PE path.
 *
 * These are the two ends of the network's layout: patchify turns the [Nf, Nt, C] input
 * grid into the [(Nf/Nh)*(Nt/Nw)*C, Nh*Nw] token matrix the transformer consumes, and
 * unpatchify puts the tokens back on the grid.  Both are pure index permutations, so
 * their cost is address arithmetic and scalar halfword traffic -- the strides on the
 * grid side (C between neighbouring t, Nt*C between neighbouring f) are not unit, so
 * the accesses cannot be v2h and there is nothing to vectorise.
 *
 * One patch is Nh subcarriers x Nw symbols; the channel index C (2, for re/im) is a
 * dimension of the OUTPUT ROW index, not of the row contents:
 *
 *     row r = (fp * TP + tp) * C + c        TP = Nt/Nw,  fp < Nf/Nh,  tp < TP,  c < C
 *     dst[r * (Nh*Nw) + f * Nw + t] = src[((fp*Nh + f) * Nt + (tp*Nw + t)) * C + c]
 *
 * Since there is no arithmetic to hide the addressing behind, the inner loop is issued
 * as inline assembly and the strided side uses the Xpulp post-increment form (p.lh /
 * p.sh, `xpulppostmod`): the address register is bumped by the element stride as part
 * of the access, so a patch row costs one memory instruction per element with no
 * separate address update.  The contiguous side keeps immediate-offset lh / sh, and the
 * inner loop is unrolled by 4 so the loads of a block issue before its stores wait on
 * them.  Halfwords travel in integer registers as raw bits -- these kernels move fp16,
 * they never interpret it.
 *
 * Both kernels parallelise over r, so each core owns whole contiguous destination rows
 * (patchify) or a disjoint patch of the grid (unpatchify) -- no element is written
 * twice and no core needs to know what another is doing.  Neither barriers internally;
 * the caller rendezvous afterwards.
 */

static void patchify_f16(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                  uint32_t Nf, uint32_t Nt, uint32_t C, uint32_t Nh,
                  uint32_t Nw, uint32_t core_id, uint32_t numThreads) {
  uint32_t FP = Nf / Nh;      // patches along frequency
  uint32_t TP = Nt / Nw;      // patches along time
  uint32_t P = Nh * Nw;       // elements in one output row (one patch, one channel)
  uint32_t R = FP * TP * C;   // output rows

  // Byte strides on the grid side: between neighbouring t, and between the starts of
  // two neighbouring f-lines of the same patch.
  const uint32_t s_t = C * (uint32_t)sizeof(__fp16);
  const uint32_t s_f = Nt * C * (uint32_t)sizeof(__fp16);

  for (uint32_t r = core_id; r < R; r += numThreads) {
    uint32_t c = r % C;
    uint32_t q = r / C;
    uint32_t tp = q % TP;
    uint32_t fp = q / TP;

    __fp16 *__restrict__ out = &dst[r * P];
    // Grid origin of this patch, already offset to its channel: element (f, t) of the
    // patch sits at line[t * C], with line advancing by Nt * C per f.
    const __fp16 *__restrict__ line = &src[((fp * Nh) * Nt + tp * Nw) * C + c];

    for (uint32_t f = 0; f < Nh; f++) {
      const __fp16 *in = line;   // walks the f-line, bumped by p.lh
      __fp16 *o = out;           // walks the (contiguous) output row
      uint32_t t = 0;
      uint32_t x0, x1, x2, x3;

      for (; t + 4 <= Nw; t += 4) {
        asm volatile("p.lh %[x0], %[s_t](%[in]!);"
                     "p.lh %[x1], %[s_t](%[in]!);"
                     "p.lh %[x2], %[s_t](%[in]!);"
                     "p.lh %[x3], %[s_t](%[in]!);"
                     "sh   %[x0], 0(%[o]);"
                     "sh   %[x1], 2(%[o]);"
                     "sh   %[x2], 4(%[o]);"
                     "sh   %[x3], 6(%[o]);"
                     : [x0] "=&r"(x0), [x1] "=&r"(x1), [x2] "=&r"(x2),
                       [x3] "=&r"(x3), [in] "+&r"(in)
                     : [s_t] "r"(s_t), [o] "r"(o)
                     : "memory");
        o += 4;
      }
      for (; t < Nw; t++) {
        asm volatile("p.lh %[x0], %[s_t](%[in]!);"
                     "sh   %[x0], 0(%[o]);"
                     : [x0] "=&r"(x0), [in] "+&r"(in)
                     : [s_t] "r"(s_t), [o] "r"(o)
                     : "memory");
        o++;
      }

      out += Nw;
      line = (const __fp16 *)((const char *)line + s_f);
    }
  }

  return;
}

static void unpatchify_f16(const __fp16 *__restrict__ src, __fp16 *__restrict__ dst,
                    uint32_t Nf, uint32_t Nt, uint32_t C, uint32_t Nh,
                    uint32_t Nw, uint32_t core_id, uint32_t numThreads) {
  uint32_t FP = Nf / Nh;
  uint32_t TP = Nt / Nw;
  uint32_t P = Nh * Nw;
  uint32_t R = FP * TP * C;

  const uint32_t s_t = C * (uint32_t)sizeof(__fp16);
  const uint32_t s_f = Nt * C * (uint32_t)sizeof(__fp16);

  for (uint32_t r = core_id; r < R; r += numThreads) {
    uint32_t c = r % C;
    uint32_t q = r / C;
    uint32_t tp = q % TP;
    uint32_t fp = q / TP;

    const __fp16 *__restrict__ in = &src[r * P];
    __fp16 *__restrict__ line = &dst[((fp * Nh) * Nt + tp * Nw) * C + c];

    for (uint32_t f = 0; f < Nh; f++) {
      const __fp16 *i = in;   // walks the (contiguous) input row
      __fp16 *out = line;     // walks the f-line, bumped by p.sh
      uint32_t t = 0;
      uint32_t x0, x1, x2, x3;

      for (; t + 4 <= Nw; t += 4) {
        asm volatile("lh   %[x0], 0(%[i]);"
                     "lh   %[x1], 2(%[i]);"
                     "lh   %[x2], 4(%[i]);"
                     "lh   %[x3], 6(%[i]);"
                     "p.sh %[x0], %[s_t](%[out]!);"
                     "p.sh %[x1], %[s_t](%[out]!);"
                     "p.sh %[x2], %[s_t](%[out]!);"
                     "p.sh %[x3], %[s_t](%[out]!);"
                     : [x0] "=&r"(x0), [x1] "=&r"(x1), [x2] "=&r"(x2),
                       [x3] "=&r"(x3), [out] "+&r"(out)
                     : [s_t] "r"(s_t), [i] "r"(i)
                     : "memory");
        i += 4;
      }
      for (; t < Nw; t++) {
        asm volatile("lh   %[x0], 0(%[i]);"
                     "p.sh %[x0], %[s_t](%[out]!);"
                     : [x0] "=&r"(x0), [out] "+&r"(out)
                     : [s_t] "r"(s_t), [i] "r"(i)
                     : "memory");
        i++;
      }

      in += Nw;
      line = (__fp16 *)((char *)line + s_f);
    }
  }

  return;
}
"""


def _uses_reshape_kernels(clusters: List[ClusterSplit]) -> bool:
    """True when some stage runs a PE-run layout row, i.e. the app needs the kernels."""
    return any(row.pe_layout
               for cs in clusters for seg in cs.segments for row in seg.rows)


def _render_reshape_kernels() -> List[str]:
    """The patchify / unpatchify kernel source, emitted into the generated app."""
    return (['/* Patch permutation kernels, carried by this app itself so it is '
             'self-contained. */'] + _RESHAPE_KERNELS.splitlines())


# Layout-only ops.  The bookends -- the leading and trailing layout run, which
# mark_pe_layout_rows tags as the model's patchify / unpatchify -- are lowered to the PE
# kernels the app carries itself (_RESHAPE_KERNELS), which do the real permutation.  Every
# other layout row is folded into the neighbouring stage hand-off, whose DMA re-lays the
# tensor out as its address pattern; none is left to be copied locally.
LAYOUT_KERNELS = {'reshape', 'split', 'transpose'}


def _prod(xs: List[int]) -> int:
    p = 1
    for x in xs:
        p *= x
    return p


def _layer_shape(row: 'LayerRow', n_par: int) -> dict:
    """Per-cluster kernel shape for one layer, derived from its i_*/o_* dims.

    The outermost multiplicity is divided by n_par (>=1) so a head-parallel cluster
    only does its one head's share.  Raises if the template lacked the dim columns.
    """
    k = row.kernel.lower()
    if k not in LAYOUT_KERNELS and not row.in_dims:
        raise ValueError(
            f"Row {row.excel_row} ({row.kernel} '{row.op}') has no i_* dims; the "
            f"template needs the tensor-shape columns for real-kernel mode "
            f"(use --wait to fall back to timing-only mempool_wait)."
        )
    ind, outd = row.in_dims, row.out_dims

    def last(lst: List[int], i: int) -> int:   # lst[-i], or 1 if too short
        return lst[-i] if len(lst) >= i else 1

    if k in ('dense', 'matmul', 'conv1d'):
        M = last(ind, 2)
        N = last(ind, 1)
        P = last(outd, 1)
        batch = _prod(ind[:-2]) if len(ind) > 2 else 1
        return {'cls': 'gemm', 'M': M, 'N': N, 'P': P,
                'batch': max(1, batch // n_par)}
    if k in ('softmax', 'layernorm'):
        N = last(ind, 1)
        M = _prod(ind[:-1]) if len(ind) > 1 else 1
        return {'cls': k, 'M': max(1, M // n_par), 'N': N}
    if k in ('gelu', 'relu'):
        return {'cls': k, 'size': max(1, _prod(ind) // n_par)}
    if k == 'add':
        return {'cls': 'add', 'len': max(1, _prod(ind) // n_par)}
    # layout op -- sized by its data_out (== data_in) footprint
    nbytes = max(1, int(row.dout_bytes) // n_par)
    if row.pe_layout:
        # Real permutation on the cores.  The patch nest has no head dimension to split, so
        # a head-parallel stage could not run it -- which never happens, since these rows are
        # the pipeline's bookends and stage 0 / the last stage are never head-parallel.
        if n_par > 1:
            raise ValueError(
                f"Row {row.excel_row} ('{row.op}') is the model's {row.pe_layout} but landed "
                f"on a head-parallel stage ({n_par} lanes); it has no head dimension to split.")
        return {'cls': row.pe_layout, 'elems': max(1, nbytes // BYTES_PER_ELEM),
                **_patch_geometry(row)}
    return {'cls': 'layout', 'bytes': nbytes,
            'elems': max(1, nbytes // BYTES_PER_ELEM)}


def _layer_footprint(row: 'LayerRow', n_par: int) -> Tuple[int, int, int]:
    """(X_BUF, W_BUF, Y_BUF) element counts one layer needs, per cluster."""
    s = _layer_shape(row, n_par)
    c = s['cls']
    if c == 'gemm':
        # Operand A from X, operand B from W -- including matmul, whose B is an
        # activation: parking it in W instead of a second region of X keeps X at one
        # tensor, which is what makes the attention stage (an M x M score matrix in
        # both X and Y) fit L1 at all.
        return s['M'] * s['N'], s['N'] * s['P'], s['M'] * s['P']
    if c in ('softmax', 'layernorm'):
        return s['M'] * s['N'], 0, s['M'] * s['N']
    if c in ('gelu', 'relu'):
        return 0, 0, s['size']
    if c == 'add':
        # Y += X: the addend is the residual, i.e. this stage's input, so it comes
        # from X like every other operand A and W_BUF stays weights-only.
        return s['len'], 0, s['len']
    # layout, whether it runs on the DMA or on the PEs: X holds the tensor, Y the permuted
    # copy, and both sides have the same element count.
    return s['elems'], 0, s['elems']


def _scratch_elems(clusters: List['ClusterSplit'],
                   wait: bool = False) -> Tuple[int, int, int]:
    """Max (X_BUF, W_BUF, Y_BUF) element counts over every layer of the app.

    X and Y are also the pipeline's DMA endpoints -- X receives the stage's inbound
    hand-off (HBM on stage 0, the producer's push otherwise) and Y feeds the outbound
    one -- so each is sized to the LARGER of its compute footprint and the biggest
    hand-off it must hold.

    --wait is sized IDENTICALLY, even though its mempool_wait bodies touch nothing: it is
    the timing twin of the real app, so it has to declare the same L1 or it would occupy a
    different machine than the one it stands in for -- and could link where the real app
    does not.  The compute terms are dropped only for a template that lacks the i_*/o_*
    dim columns altogether, which --wait alone tolerates; there is nothing to size them
    from, so the hand-off terms are all that remain.

    Each is rounded up to a 32-element (64-byte) boundary.  Buffers are reused across
    layers and batch iterations (placeholder data), so sizing to the max is enough, and
    the round-up keeps every row of a double-buffered [2][N] array 64-byte aligned for
    the DMA.
    """
    nx = nw = ny = 1
    try:
        for cs in clusters:
            for seg in cs.segments:
                for row in seg.rows:
                    x, w, y = _layer_footprint(row, cs.n_par)
                    nx, nw, ny = max(nx, x), max(nw, w), max(ny, y)
    except ValueError:
        if not wait:
            raise          # real-kernel mode cannot emit those calls either
        nx = nw = ny = 1   # dim-less template: hand-off terms only
    # hand-off terms: _cluster_din already gives the FULL gathered tensor on a P->1
    # stage and the 1/P slice on a scatter, i.e. exactly what X_BUF receives.
    elems = lambda nbytes: (int(nbytes) + BYTES_PER_ELEM - 1) // BYTES_PER_ELEM
    for i in range(len(clusters)):
        nx = max(nx, elems(_cluster_din(clusters, i)))
        ny = max(ny, elems(_cluster_dout(clusters, i)))
    rnd = lambda n: ((n + 31) // 32) * 32
    return rnd(nx), rnd(nw), rnd(ny)


def _render_layer_call(row: 'LayerRow', pool: str, n_par: int,
                       bench: bool = False, dbl: bool = False) -> List[str]:
    """C statement(s) for one layer, dispatched by kernel and pool.

    GEMM (dense/matmul/conv1d) runs on the RedMulE tensor engines on a pool in
    TENSOR_ENGINE_POOLS and on the PE matmul kernel elsewhere -- the same fact the cost
    model derates with.  Every other op runs on the PE path.  Each
    kernel path ends aligned (the kernel barriers internally, or a trailing
    flex_intra_cluster_sync() is appended), so the next layer reads a completed output.

    With bench=True the layer is wrapped in mempool_start_benchmark()/mempool_stop_benchmark()
    so an RTL run traces exactly the kernel (its trailing sync included, i.e. the layer is
    traced until every core is done).  The markers add no barrier of their own: the cores are
    already aligned on entry (the previous layer ended in one, the first follows a global
    barrier), so the traced region is the layer as it runs unbenchmarked.

    A DMA-run LAYOUT layer is exempt: its body is a dma_2d(), which brackets itself when bench
    is on -- wrapping it again would nest two `trace` toggles and the inner stop would end the
    region early.  The PE-run ones (patchify / unpatchify) are plain kernels and are wrapped
    like the rest.
    """
    s   = _layer_shape(row, n_par)
    c   = s['cls']
    tag = _c_comment(f'{row.op} [{row.kernel}]')
    work: List[str] = []
    # With double buffering the kernels work on the parity the caller passes in.
    xbuf = 'X_BUF[buf]' if dbl else 'X_BUF'
    ybuf = 'Y_BUF[buf]' if dbl else 'Y_BUF'

    if c == 'gemm':
        M, N, P, batch = s['M'], s['N'], s['P'], s['batch']
        # Operand A from X, operand B from W.  A matmul's B is an activation rather
        # than a weight, but nothing is validated end to end and keeping it out of X
        # halves the attention stage's X footprint (see _layer_footprint).
        b_buf = 'W_BUF'
        if pool in TENSOR_ENGINE_POOLS:
            # RedMulE: partitions rows over the tensor engines.  Its internal
            # mempool_barrier syncs the cores, but does NOT guarantee the HWPE
            # write-back to Y_BUF has drained before the next kernel reads it, so a
            # trailing flex_intra_cluster_sync() is appended.
            # GEMM (Y = X*W + Y) is the only op RedMulE is driven with; a plain
            # product is a GEMM onto a zeroed Y.
            call = (f'redmule_synch_parallel({xbuf}, {b_buf}, {ybuf}, '
                    f'{M}u, {N}u, {P}u, GEMM, false, 0);')
            needs_sync = True
        else:
            # No RedMulE on this pool -> PE matmul; no internal barrier.  This is the
            # path the cost model charges PE_utilization for, not TE_utilization.
            call = (f'matmul_4x2_parallel_f16vec({xbuf}, {b_buf}, {ybuf}, '
                    f'{M}u, {N}u, {P}u, cid, nt);')
            needs_sync = True
        if batch > 1:
            work.append(f'for (uint32_t b = 0; b < {batch}u; b++) {{  '
                        f'/* {tag}: {batch} batched matmuls (per cluster) */')
            work.append(f'    {call}')
            if needs_sync:
                work.append('    flex_intra_cluster_sync();')
            work.append('}')
        else:
            work.append(f'{call}  /* {tag} */')
            if needs_sync:
                work.append('flex_intra_cluster_sync();')
    elif c == 'softmax':
        work.append(f'softmax_parallel_2x4_f16vec({xbuf}, {ybuf}, '
                    f'{s["M"]}u, {s["N"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'layernorm':
        work.append(f'layernorm_parallel_2x4_f16vec({xbuf}, {ybuf}, '
                    f'{s["M"]}u, {s["N"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'gelu':
        work.append(f'gelu_f16({ybuf}, {s["size"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'relu':
        work.append(f'relu_f16({ybuf}, {s["size"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'add':
        # The real baremetal/mempool_axpy_f16.h kernel, called directly.  It ends in
        # mempool_log_barrier, so it self-barriers -- no trailing flex_intra_cluster_sync().
        # 0x3C00 == fp16 1.0: a normal multiplier (1u would be fp16 0x0001, a subnormal, which
        # forces the slow FP path on every vfmac).  Local-bank vectorized unrolled axpy: its
        # signature has no nPE arg (each core strides its own local banks by NUM_BANKS).
        # The addend is the residual -- this stage's input -- so it reads X, not W.
        work.append(f'axpy_f16vecp_local_unrolled4(0x3C00u, {xbuf}, {ybuf}, {s["len"]}u);  /* {tag}: Y += X */')
    elif c in ('patchify', 'unpatchify'):
        # The model's bookend reshapes, run by the cores as the real index permutation
        # rather than handed to the single DMA channel.  An ordinary PE kernel in every
        # respect: all cores enter it, and it takes the normal benchmark wrapping below.
        work.append(f'{c}_f16({xbuf}, {ybuf}, {s["Nf"]}u, {s["Nt"]}u, {s["C"]}u, '
                    f'{s["Nh"]}u, {s["Nw"]}u, cid, nt);  /* {tag} */')
        work.append('flex_intra_cluster_sync();')
    elif c == 'layout':
        # No layout row may reach here: every one of them is either a bookend the PEs run
        # (handled above) or is folded into the hand-off that already moves the tensor.  A
        # local L1->L1 copy would be that same tensor moved twice, so a row that reaches
        # this branch is a modelling gap, not something to quietly emit.
        raise ValueError(
            f"Row {row.excel_row} ('{row.op}') is a layout row that is neither a PE bookend "
            f"nor folded into a stage hand-off, so it would need a local L1->L1 copy -- "
            f"moving the same tensor twice.  Fold it into the neighbouring hand-off (see "
            f"drop_folded_layout_layers / fold_gathered_layout_layers) or give it a PE kernel.")
    else:
        raise ValueError(
            f"Row {row.excel_row} ('{row.op}'): no kernel is mapped to '{row.kernel}'.")

    if not bench:
        return work
    # Trace exactly this layer: two CSR writes around the work, no barrier added.
    out = ['{', '    mempool_start_benchmark();']
    out += ['    ' + w for w in work]
    out += ['    mempool_stop_benchmark();', '}']
    return out


def _render_compute_body_real(cluster: 'ClusterSplit', pool: str,
                              bench: bool = False, dbl: bool = False) -> List[str]:
    """Real kernel calls for a stage, block reps wrapped in a `for` loop."""
    lines: List[str] = []
    if cluster.n_par > 1:
        lines.append(f'    /* head-parallel: this cluster owns 1 of {cluster.n_par} heads, '
                     f'so each kernel below runs its 1/{cluster.n_par} share */')
    for seg in cluster.segments:
        b   = seg.block
        rng = (f'{seg.rep_from+1}-{seg.rep_to+1}/{b.n_rep}'
               if seg.n_reps > 1 else f'{seg.rep_from+1}/{b.n_rep}')
        label = _c_comment(seg_label(seg))
        lines.append(f'    /* {label} x{seg.n_reps}  (reps {rng}) */')
        lines.append(f'    for (uint32_t r = 0; r < {seg.n_reps}u; r++) {{')
        for row in seg.rows:
            for cline in _render_layer_call(row, pool, cluster.n_par, bench, dbl):
                lines.append(f'        {cline}')
        lines.append('    }')
    return lines


# ---------------------------------------------------------------------------
# Shared stage/lane scaffolding for the generated C
# ---------------------------------------------------------------------------
#
# A stage is no longer necessarily one cluster: a head-parallel stage spans n_par
# clusters.  So `cid` -- which the old generator used at once as stage selector,
# wavefront phase and DMA target -- is decoupled into (stage, lane) via lookup
# tables.  Clusters are packed back-to-back, e.g. a 1/4/1 pipeline occupies
#
#     cid    0 | 1  2  3  4 | 5
#     stage  0 |     1      | 2
#     lane   0 | 0  1  2  3 | 0

def _cluster_din(clusters: List[ClusterSplit], i: int) -> float:
    """Bytes ONE cluster of stage i takes in (its slice of the hand-off it receives)."""
    incoming = clusters[0].first_din if i == 0 else clusters[i - 1].last_dout
    return incoming / clusters[i].n_par


def _cluster_dout(clusters: List[ClusterSplit], i: int) -> float:
    """Bytes ONE cluster of stage i hands off (its slice of the stage's data_out)."""
    return clusters[i].last_dout / clusters[i].n_par


def _gather_stage(clusters: List[ClusterSplit]) -> int:
    """Index of the stage whose hand-off is a gather (P lanes -> 1), or -1 if there is none."""
    for i in range(len(clusters) - 1):
        if clusters[i].n_par > 1 and clusters[i + 1].n_par == 1:
            return i
    return -1


def _render_stage_tables(clusters: List[ClusterSplit],
                         fold: Optional['GatherFold'] = None) -> List[str]:
    """Emit the stage/lane maps and the per-cluster hand-off sizes."""
    bases  = _stage_bases(clusters)
    n_st   = len(clusters)
    n_phys = sum(cs.n_par for cs in clusters)

    cid_stage = [str(i) for i, cs in enumerate(clusters) for _ in range(cs.n_par)]
    cid_lane  = [str(k) for cs in clusters for k in range(cs.n_par)]

    L: List[str] = []
    A = L.append
    A(f'#define N_STAGES    {n_st}')
    A(f'#define N_CLUSTERS  {n_phys}  /* sum of the stages\' widths */')
    A('')
    A('/* A head-parallel stage needs more clusters than an ordinary one; make a machine')
    A(' * too small to host the pipeline a compile error rather than a silent mis-run. */')
    A('_Static_assert(N_CLUSTERS <= ARCH_NUM_CLUSTER,')
    A('               "pipeline needs more clusters than this arch provides");')
    A('')
    A('/* cid -> (stage, lane) */')
    A(f'static const uint8_t CID_STAGE[N_CLUSTERS] = {{ {", ".join(cid_stage)} }};')
    A(f'static const uint8_t CID_LANE [N_CLUSTERS] = {{ {", ".join(cid_lane)} }};')
    A('')
    A('/* stage -> first cid, and how many clusters run it in parallel */')
    A(f'static const uint8_t STAGE_BASE [N_STAGES + 1] = '
      f'{{ {", ".join(str(b) for b in bases)}, {n_phys} }};')
    A(f'static const uint8_t STAGE_LANES[N_STAGES] = '
      f'{{ {", ".join(str(cs.n_par) for cs in clusters)} }};')
    A('')
    A('/* Bytes ONE cluster of a stage hands off (its slice of the stage data_out) */')
    A(f'static const uint32_t STAGE_DOUT_BYTES[N_STAGES] = {{')
    for i, cs in enumerate(clusters):
        seg_info = ', '.join(
            f'{seg_label(seg)}({seg.rep_from+1}-{seg.rep_to+1}/{seg.block.n_rep})'
            for seg in cs.segments
        )
        par = f' x{cs.n_par} lanes' if cs.n_par > 1 else ''
        A(f'    {int(_cluster_dout(clusters, i))}U,  '
          f'/* stage {i}{par}: [{_c_comment(seg_info)}] */')
    A('};')

    # Non-zero only on a stage whose gather re-lays the tensor out: the contiguous run one
    # lane contributes to each destination row.  0 means "gather the slices side by side".
    gs  = _gather_stage(clusters) if fold else -1
    row = [str(fold.row_bytes) + 'U' if i == gs else '0U' for i in range(n_st)]
    A('')
    A('/* Per-lane row bytes of a re-laying gather: the lane interleaves its rows into')
    A(' * column slot `lane` of the consumer\'s matrix, which is the model\'s transpose +')
    A(' * reshape done as the hand-off\'s address pattern.  0 = plain contiguous gather. */')
    A(f'static const uint32_t STAGE_GATHER_ROW_BYTES[N_STAGES] = {{ {", ".join(row)} }};')
    if gs >= 0:
        A(f'/* stage {gs}: {fold.row_bytes} B x {fold.reps} reps per lane, dst_stride '
          f'{fold.row_bytes * clusters[gs].n_par} B */')
    return L


def _render_push_helper() -> str:
    """Shared C: hand a stage's output to the next stage, whatever the fan-out.

    Stage widths differ, so the hand-off is one of three shapes.  Let `own` be the bytes
    this cluster holds (STAGE_DOUT_BYTES[stage]):

      * same width  -- lane k feeds lane k of the next stage: one push of `own`.
      * 1 -> P      -- SCATTER: the single producer slices `own` into P head-chunks and
                       pushes chunk k to lane k of the next stage.
      * P -> 1      -- GATHER: each of the P producers pushes its `own` into the single
                       consumer's buffer.  With STAGE_GATHER_ROW_BYTES set, the push also
                       carries the model's transpose+reshape: the lane interleaves its
                       rows into column slot `lane` of the consumer's matrix, so the
                       tensor arrives already laid out and no local copy follows it.
                       Otherwise the P slices simply land side-by-side.
    """
    return '\n'.join([
        'static inline void push_to_next(uint32_t stage, uint32_t lane,',
        '                                uint32_t dst_off, uint64_t src)',
        '{',
        '    uint32_t next     = stage + 1u;',
        '    uint32_t lanes    = STAGE_LANES[stage];',
        '    uint32_t lanes_nx = STAGE_LANES[next];',
        '    uint32_t own      = STAGE_DOUT_BYTES[stage];',
        '    uint32_t base_nx  = STAGE_BASE[next];',
        '',
        '    if (lanes == lanes_nx) {                 /* straight hand-off */',
        '        dma_2d((uint64_t)remote_cid(base_nx + lane, dst_off), src, own);',
        '    } else if (lanes == 1u) {                /* scatter: 1 -> lanes_nx heads */',
        '        uint32_t chunk = own / lanes_nx;',
        '        for (uint32_t k = 0; k < lanes_nx; k++)',
        '            dma_2d((uint64_t)remote_cid(base_nx + k, dst_off),',
        '                   src + (uint64_t)(k * chunk), chunk);',
        '    } else {                                 /* gather: lanes heads -> 1 */',
        '        uint32_t rb = STAGE_GATHER_ROW_BYTES[stage];',
        '        if (rb)                              /* re-laying out on the way */',
        '            dma_2d_strided((uint64_t)remote_cid(base_nx, dst_off + lane * rb),',
        '                           src, rb, rb * lanes, rb, own / rb);',
        '        else',
        '            dma_2d((uint64_t)remote_cid(base_nx, dst_off + lane * own), src, own);',
        '    }',
        '}',
    ])


def _render_dma_issue_helper() -> str:
    """Shared C (real-kernel mode): fire-and-forget DMA -- program + return.

    Unlike dma_2d it does NOT wait, so the issuing core can move on and join the
    compute while the copy runs.  Every issue must be drained by a later
    flex_dma_async_wait_all() before the source/destination buffer is reused.

    It carries no benchmark markers of its own: issue and drain sit on opposite sides
    of the compute, so main() brackets the two regions separately when --bench is on
    (bracketing them as one would nest the per-layer markers in between).
    """
    return '\n'.join([
        'static inline void dma_2d_strided_issue(uint64_t dst, uint64_t src, uint32_t chunk,',
        '                                        uint32_t dst_stride, uint32_t src_stride,',
        '                                        uint32_t reps)',
        '{',
        '    flex_dma_async_2d(dst, src, chunk, dst_stride, src_stride, reps);',
        '}',
        '',
        'static inline void dma_2d_issue(uint64_t dst, uint64_t src, uint32_t bytes)',
        '{',
        '    dma_2d_strided_issue(dst, src, bytes, bytes, bytes, 1);',
        '}',
    ])


def _render_push_issue_helper() -> str:
    """Shared C (real-kernel mode): async twin of push_to_next.

    Same three hand-off shapes, but every sub-DMA is issued without waiting.  The
    caller drains them with one flex_dma_async_wait_all() after the compute it
    overlaps, so the outbound transfer of the previous item runs concurrently with
    this step's compute (true compute-while-transfer).  Like dma_2d_issue it carries
    no benchmark markers -- main() brackets the issue and the drain.
    """
    return '\n'.join([
        'static inline void push_to_next_issue(uint32_t stage, uint32_t lane,',
        '                                      uint32_t dst_off, uint64_t src)',
        '{',
        '    uint32_t next     = stage + 1u;',
        '    uint32_t lanes    = STAGE_LANES[stage];',
        '    uint32_t lanes_nx = STAGE_LANES[next];',
        '    uint32_t own      = STAGE_DOUT_BYTES[stage];',
        '    uint32_t base_nx  = STAGE_BASE[next];',
        '',
        '    if (lanes == lanes_nx) {                 /* straight hand-off */',
        '        dma_2d_issue((uint64_t)remote_cid(base_nx + lane, dst_off), src, own);',
        '    } else if (lanes == 1u) {                /* scatter: 1 -> lanes_nx heads */',
        '        uint32_t chunk = own / lanes_nx;',
        '        for (uint32_t k = 0; k < lanes_nx; k++)',
        '            dma_2d_issue((uint64_t)remote_cid(base_nx + k, dst_off),',
        '                         src + (uint64_t)(k * chunk), chunk);',
        '    } else {                                 /* gather: lanes heads -> 1 */',
        '        uint32_t rb = STAGE_GATHER_ROW_BYTES[stage];',
        '        if (rb)                              /* re-laying out on the way */',
        '            dma_2d_strided_issue((uint64_t)remote_cid(base_nx, dst_off + lane * rb),',
        '                                 src, rb, rb * lanes, rb, own / rb);',
        '        else',
        '            dma_2d_issue((uint64_t)remote_cid(base_nx, dst_off + lane * own), src, own);',
        '    }',
        '}',
    ])


# ---------------------------------------------------------------------------
# Shared renderer blocks (used by both ctt and cwt)
# ---------------------------------------------------------------------------

def _render_includes(wait: bool) -> List[str]:
    """Header includes: always the flex runtime; the baremetal kernels in real mode."""
    L = ['#include "flex_runtime.h"',
         '#include "flex_dma_pattern.h"']
    if not wait:
        L += [f'#include "{h}"' for h in _KERNEL_INCLUDES]
    return L


def _render_dataflow_note() -> List[str]:
    """Header note: the buffers are the DMA endpoints, so layer inputs are real traffic.

    Worth stating in the generated file because it bears on the measured cycles: X_BUF is
    overwritten every item by the producer's Y_BUF (and on stage 0 by whatever HBM holds,
    which nothing initialises), so the core-side fp16 kernels -- softmax, layernorm, gelu,
    axpy, and matmul on Terapool -- see computed values instead of the primed 1.0s and may
    take data-dependent FP paths.  RedMulE GEMMs are fixed-latency and unaffected.
    """
    return [
        '/* Dataflow: X_BUF is both the kernel input and the inbound DMA target, Y_BUF both',
        ' * the kernel output and the outbound DMA source.  So a layer\'s operands are the',
        ' * bytes the previous stage actually sent (and, on stage 0, uninitialised HBM):',
        ' * the core-side fp16 kernels are data-dependent on NaN/subnormal operands, so',
        ' * per-layer cycles may differ from a run on primed data.  RedMulE GEMMs are',
        ' * fixed-latency and unaffected. */',
    ]


def _render_scratch_buffers(clusters: List[ClusterSplit], dbl: bool = False,
                            wait: bool = False) -> List[str]:
    """The three per-cluster L1 buffers, one per role.

    X = kernel input AND the inbound DMA target (stage 0 reads HBM into it, every other
    stage receives its producer's push there).  W = weights / operand B, resident and
    never a DMA endpoint.  Y = kernel output AND the outbound DMA source (the push to the
    next stage, or the HBM write on the last stage).

    With dbl (cwt) X and Y are doubled and indexed by item parity, so the transfer of one
    item runs against the compute of the next; W needs no parity.
    """
    nx, nw, ny = _scratch_elems(clusters, wait)
    d   = '[2]' if dbl else ''
    par = ('  Index [2] is the ITEM PARITY (compute uses one, the DMA the other).'
           if dbl else '')
    attr = '__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), section(".l1")))'
    # W is the one buffer no DMA touches, so --wait never references it -- yet it still has
    # to occupy its L1, which is why it is declared: `used` keeps it and silences the warning.
    attr_w = ('__attribute__((aligned(NUM_BANKS * sizeof(int32_t)), '
              'section(".l1"), used))')
    return [
        '/* Per-cluster L1 buffers, one per role (placeholder data -- only cycles are',
        ' * measured).  X = kernel input, and where the inbound DMA lands (HBM on stage 0,',
        ' * the previous cluster\'s push otherwise).  W = weights / operand B, resident.',
        ' * Y = kernel output, and what the outbound DMA sends on to the next stage (or to',
        ' * HBM on the last stage).  Every kernel of a stage reads X at its own offset and',
        ' * writes Y -- nothing is validated end to end, so no ping-pong between layers is',
        f' * needed.  Sized to the app-wide per-cluster maximum of compute and hand-off.{par} */',
        f'#define X_ELEMS  {nx}U',
        f'#define W_ELEMS  {nw}U',
        f'#define Y_ELEMS  {ny}U',
        f'static __fp16 X_BUF{d}[X_ELEMS] {attr};',
        f'static __fp16 W_BUF[W_ELEMS] {attr_w};',
        f'static __fp16 Y_BUF{d}[Y_ELEMS] {attr};',
    ]


def _render_buffer_init(dbl: bool = False) -> List[str]:
    """Prime the buffers with a normal fp16 value (1.0) -- emitted in main() BEFORE
    the timed region.  Uninitialised .l1 is garbage (NaN/Inf/subnormal), and core-side fp16
    ops on such operands take a data-dependent, far slower path that stalls the model; normal
    inputs give the normal-path, data-independent cycle counts we intend to measure.  Each core
    fills its stride of the per-cluster buffers, then a rendezvous.  Both parities are primed
    when X/Y are double-buffered."""
    xb = 'X_BUF[b][i]' if dbl else 'X_BUF[i]'
    yb = 'Y_BUF[b][i]' if dbl else 'Y_BUF[i]'
    ind = '    ' if dbl else ''
    L = [
        '    /* Prime the buffers with fp16 1.0 before timing (avoids NaN/subnormal FP',
        '     * stalls; gives the normal-path cycle counts).  All cores, outside the timed region. */',
    ]
    if dbl:
        L.append('    for (uint32_t b = 0; b < 2u; b++) {')
    L += [
        f'{ind}    for (uint32_t i = mempool_get_core_id(); i < X_ELEMS; i += mempool_get_core_count()) {xb} = (__fp16)1.0f;',
        f'{ind}    for (uint32_t i = mempool_get_core_id(); i < Y_ELEMS; i += mempool_get_core_count()) {yb} = (__fp16)1.0f;',
    ]
    if dbl:
        L.append('    }')
    L += [
        '    for (uint32_t i = mempool_get_core_id(); i < W_ELEMS; i += mempool_get_core_count()) W_BUF[i] = (__fp16)1.0f;',
        '    flex_intra_cluster_sync();',
    ]
    return L


def _render_compute_stage_fns(clusters: List[ClusterSplit], pool: str, wait: bool,
                              row_lat: Dict[int, float], cycles_per_ms: int,
                              bench: bool = False, dbl: bool = False) -> List[str]:
    """The per-stage compute_stage_i() functions: mempool_wait (--wait) or real kernels.

    With dbl the functions take the item parity and read X_BUF[buf] / write Y_BUF[buf].
    """
    L: List[str] = []
    if wait:
        L.append('/* ---- Per-cluster compute: one mempool_wait() per layer (timing-only). */')
    elif bench:
        L.append('/* ---- Per-cluster compute: real baremetal kernel calls, X -> Y,')
        L.append('   each layer bracketed by mempool_start/stop_benchmark() (trace CSR). */')
    else:
        L.append('/* ---- Per-cluster compute: real baremetal kernel calls, X -> Y.')
        L.append('   No instrumentation -- the kernels run raw. */')
    # --wait takes the parity argument too: the two builds share one main loop, so the
    # signatures have to agree, even though a mempool_wait() body never reads it.
    arg = 'uint32_t buf' if dbl else 'void'
    for i, cs in enumerate(clusters):
        seg_str = ', '.join(f'{seg_label(seg)} x{seg.n_reps}' for seg in cs.segments)
        L.append(f'/* stage {i} : [{_c_comment(seg_str)}] */')
        L.append(f'static inline void compute_stage_{i}({arg})')
        L.append('{')
        if wait:
            if dbl:
                L.append('    (void)buf;   /* no buffer is touched: this is the timing twin */')
            L.extend(_render_compute_body(cs, row_lat, cycles_per_ms))
        else:
            L.append('    uint32_t cid = mempool_get_core_id();    /* in-cluster core id */')
            L.append('    uint32_t nt  = mempool_get_core_count();  /* cores per cluster */')
            L.append('    (void)cid; (void)nt;   /* unused on a RedMulE-only stage */')
            L.extend(_render_compute_body_real(cs, pool, bench, dbl))
        L.append('}')
        L.append('')
    return L


# ---------------------------------------------------------------------------
# C renderers (one per mode)
# ---------------------------------------------------------------------------

def _render_ctt_c(pool: str, clusters: List[ClusterSplit],
                  row_lat: Dict[int, float], cycles_per_ms: int, n_iter: int,
                  bench: bool = False, wait: bool = False,
                  fold: Optional['GatherFold'] = None) -> str:
    n            = len(clusters)
    n_phys       = sum(cs.n_par for cs in clusters)
    first_din    = int(clusters[0].first_din)
    hbm_out_off  = (first_din + 63) & ~63

    stage_fns = _render_compute_stage_fns(clusters, pool, wait, row_lat,
                                          cycles_per_ms, bench)

    lines: List[str] = []
    A = lines.append

    kind = 'timing-only (mempool_wait)' if wait else 'real baremetal kernels'
    kind += (' + mempool_start/stop_benchmark() around every layer and DMA' if bench
             else ' (uninstrumented)')
    A(f'/* Auto-generated by cevit_pipeline.py -- compute-then-transfer pipeline for {pool} */')
    A(f'/* {n} stage(s) over {n_phys} cluster(s); {n_iter} item(s) streamed as a wavefront. */')
    A(f'/* Compute: {kind}. */')
    if not wait:
        lines.extend(_render_dataflow_note())
    A('')
    lines.extend(_render_includes(wait))
    A('')
    if wait:
        A('/* CYCLES_PER_MS: nop-loop iterations that approximate 1 ms of wall time. */')
        A(f'#define CYCLES_PER_MS  {cycles_per_ms}UL')
        A('')
    A(f'#define N_ITER               {n_iter}U  /* items streamed through the pipeline */')
    A(f'#define CLUSTER_0_DIN_BYTES  {first_din}U  /* first layer data_in  */')
    A(f'#define HBM_INPUT_OFFSET     0UL')
    A(f'#define HBM_OUTPUT_OFFSET    {hbm_out_off}UL  /* after input data in HBM */')
    A('')
    A('\n'.join(_render_stage_tables(clusters, fold)))
    A('')
    A(_render_dma_helper(bench))
    A('')
    A(_render_push_helper())
    A('')
    if not wait and _uses_reshape_kernels(clusters):
        lines.extend(_render_reshape_kernels())
        A('')
    # buffers must precede the compute_stage fns that reference them
    lines.extend(_render_scratch_buffers(clusters, dbl=False, wait=wait))
    A('')
    lines.extend(stage_fns)
    A('int main()')
    A('{')
    A('    uint32_t cid     = flex_get_cluster_id();')
    A('    uint32_t eoc_val = 0;')
    A('    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : (uint32_t)N_STAGES;')
    A('    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;')
    A('    /* buf_off: TCDM offset of the RECEIVER\'s X_BUF -- identical in every cluster')
    A('     * (same binary), so a producer can address it without any exchange. */')
    A('    const uint32_t buf_off = (uint32_t)(uintptr_t)X_BUF;')
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
    A('    /*')
    A('     * Real streaming pipeline (compute-then-transfer).')
    A('     * step counts pipeline ticks: STAGE s handles item (step - s) when')
    A('     * 0 <= step - s < N_ITER.  The loop covers FILL (stages switch on as step')
    A('     * grows), STEADY STATE (all N busy on different items) and DRAIN (stages')
    A('     * switch off as items leave).  Data is placeholder -- only cycles are measured.')
    A('     *')
    A('     * The wavefront is keyed on the STAGE, not the cluster: a head-parallel stage')
    A('     * has STAGE_LANES[s] clusters that all work on the SAME item in the same step,')
    A('     * one head each.  They need no barrier of their own -- the per-step global')
    A('     * barriers below already rendezvous them.')
    A('     *')
    A('     * Within each active cluster the worker cores run compute_stage_s() while the')
    A('     * dm_core handles DMA.  compute -> barrier -> transfer -> barrier, so compute and')
    A('     * the outbound transfer do NOT overlap (compute-then-transfer) -- which is also')
    A('     * why one X_BUF and one Y_BUF suffice: nothing reads Y_BUF while compute writes it.')
    A('     *')
    A('     *   HBM --[DMA]--> S0 --[DMA]--> S1 --> ... --> S(N-1) --[DMA]--> HBM')
    A('     */')
    A('    for (uint32_t step = 0; step < N_ITER + N_STAGES - 1; step++) {')
    A('        uint32_t active = (cid < N_CLUSTERS) && (stage <= step)')
    A('                          && (step - stage < N_ITER);')
    A('')
    A('        /* ---- Compute phase: ALL cores compute; dm_core also loads stage-0 input ---- */')
    A('        if (active) {')
    A('            if (stage == 0u) {')
    A('                /* Stage 0 input arrives from HBM this step: its worker cores must')
    A('                 * not compute until the dm_core load has completed. */')
    A('                if (flex_is_dm_core()) {')
    A('                    dma_2d((uint64_t)(uintptr_t)X_BUF,')
    A('                           hbm_addr(HBM_INPUT_OFFSET),')
    A('                           CLUSTER_0_DIN_BYTES);')
    A('                }')
    A('                flex_intra_cluster_sync();   /* worker cores wait for the HBM load */')
    A('            }')
    A('            /* Every core computes (incl. dm_core): on stage 0 it has finished the HBM')
    A('             * load + intra-cluster sync above, so it is free to join the compute. */')
    A('            switch (stage) {')
    for i in range(n):
        A(f'                case {i}: compute_stage_{i}(); break;')
    A('                default: break;')
    A('            }')
    A('        }')
    A('        flex_global_barrier_xy();   /* compute done before ANY transfer */')
    A('')
    A('        /* ---- Transfer phase (all active clusters in parallel): send Y_BUF ---- */')
    A('        if (flex_is_dm_core() && active) {')
    A('            if (stage == N_STAGES - 1u) {')
    A('                dma_2d(hbm_addr(HBM_OUTPUT_OFFSET + lane * STAGE_DOUT_BYTES[stage]),')
    A('                       (uint64_t)(uintptr_t)Y_BUF,')
    A('                       STAGE_DOUT_BYTES[stage]);')
    A('            } else {')
    A('                push_to_next(stage, lane, buf_off, (uint64_t)(uintptr_t)Y_BUF);')
    A('            }')
    A('        }')
    A('        flex_global_barrier_xy();   /* all transfers done before next compute */')
    A('    }')
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


def _render_cwt_c(pool: str, clusters: List[ClusterSplit],
                  row_lat: Dict[int, float], cycles_per_ms: int, n_iter: int,
                  bench: bool = False, wait: bool = False,
                  fold: Optional['GatherFold'] = None) -> str:
    n            = len(clusters)
    n_phys       = sum(cs.n_par for cs in clusters)
    first_din    = int(clusters[0].first_din)
    hbm_out_off  = (first_din + 63) & ~63
    # X and Y are double-buffered by item parity so a transfer can run against the next
    # item's compute; in --wait mode no kernel touches them, but the DMAs still do.
    dbl          = True
    # Real overlap only pays off with more than one item in flight: at N_ITER == 1 a
    # cluster computes at step 2*stage and sends at 2*stage+1, so nothing is ever
    # concurrent and the BLOCKING dma_2d/push_to_next makes the transfer a self-contained
    # region the trace can time.  --wait picks the same branch as the real build: it is
    # that app's timing twin, so it runs the same schedule and issues the same transfers.
    overlap      = n_iter > 1

    stage_fns = _render_compute_stage_fns(clusters, pool, wait, row_lat,
                                          cycles_per_ms, bench, dbl)

    lines: List[str] = []
    A = lines.append

    kind = 'timing-only (mempool_wait)' if wait else 'real baremetal kernels'
    kind += (' + mempool_start/stop_benchmark() around every layer and DMA' if bench
             else ' (uninstrumented)')
    A(f'/* Auto-generated by cevit_pipeline.py -- compute-while-transfer pipeline for {pool} */')
    A(f'/* {n} stage(s) over {n_phys} cluster(s); {n_iter} item(s) streamed as a wavefront. */')
    A(f'/* Compute: {kind}. */')
    if not wait:
        lines.extend(_render_dataflow_note())
    A('')
    lines.extend(_render_includes(wait))
    A('')
    if wait:
        A('/* CYCLES_PER_MS: nop-loop iterations that approximate 1 ms of wall time. */')
        A(f'#define CYCLES_PER_MS  {cycles_per_ms}UL')
        A('')
    A(f'#define N_ITER               {n_iter}U  /* items streamed through the pipeline */')
    A(f'#define CLUSTER_0_DIN_BYTES  {first_din}U  /* first layer data_in  */')
    A(f'#define HBM_INPUT_OFFSET     0UL')
    A(f'#define HBM_OUTPUT_OFFSET    {hbm_out_off}UL  /* after input data in HBM */')
    A('')
    A('\n'.join(_render_stage_tables(clusters, fold)))
    A('')
    A(_render_dma_helper(bench))
    A('')
    if overlap:
        # Async fire-and-forget transfer so the dm_core can issue the outbound DMA of
        # the previous item and then JOIN the compute -- real overlap, and every core
        # reaches the kernels' internal all-core barriers.
        A(_render_dma_issue_helper())
        A('')
    A(_render_push_helper())
    A('')
    if overlap:
        A(_render_push_issue_helper())
        A('')
    if not wait and _uses_reshape_kernels(clusters):
        lines.extend(_render_reshape_kernels())
        A('')
    # buffers must precede the compute_stage fns that reference them
    lines.extend(_render_scratch_buffers(clusters, dbl=dbl, wait=wait))
    A('')
    lines.extend(stage_fns)
    A('int main()')
    A('{')
    A('    uint32_t cid     = flex_get_cluster_id();')
    A('    uint32_t eoc_val = 0;')
    A('    uint32_t stage   = (cid < N_CLUSTERS) ? CID_STAGE[cid] : (uint32_t)N_STAGES;')
    A('    uint32_t lane    = (cid < N_CLUSTERS) ? CID_LANE [cid] : 0u;')
    A('')
    A('    flex_barrier_xy_init();')
    A('    flex_global_barrier_xy();')
    if not wait:
        lines.extend(_render_buffer_init(dbl))
    A('    if (flex_get_core_id() == 0 && cid == 0) flex_timer_start();')
    A('    flex_global_barrier_xy();')
    A('')
    A('    /**************************************/')
    A('    /*  Program Execution Region -- Start */')
    A('    /**************************************/')
    A('')
    A('    /*')
    A('     * Real streaming pipeline (compute-while-transfer), hazard-free double buffering.')
    A('     *')
    A('     * Two constraints set the schedule:')
    A('     *   (1) intra-cluster overlap: a cluster computes item m into one buffer while')
    A('     *       its dm_core pushes item (m-1) from the other => push lags own compute by 1.')
    A('     *   (2) inter-stage: a consumer may compute item m only the step AFTER its')
    A('     *       producer pushed item m (the per-step barrier publishes the push).')
    A('     * Chaining (1)+(2): compute(s+1,m) = compute(s,m) + 2  =>  inter-stage offset 2.')
    A('     *')
    A('     * The wavefront is keyed on the STAGE, not the cluster: the STAGE_LANES[s] clusters')
    A('     * of a head-parallel stage share one phase and work the same item, one head each.')
    A('     *')
    A('     * X_BUF/Y_BUF are indexed by ITEM PARITY.  Stage s computes item m = step - 2*stage;')
    A('     * since 2*stage is EVEN, cbuf = step%2 and the outbound buffer obuf = cbuf^1.  The')
    A('     * producer and consumer of item m both resolve to m%2, so the producer writes')
    A('     * exactly the buffer the consumer reads next -- no free-running index, no hazard.')
    A('     * (Keeping the offset even is what makes this hold -- do not change it.)')
    A('     *')
    if overlap:
        A('     * Overlap: the dm_core ISSUES the outbound DMA of the previous item (async, no')
        A('     * wait) and then JOINS the compute like every other core, draining the transfer')
        A('     * only after compute.  So the outbound copy runs concurrently with this step\'s')
        A('     * compute (compute-while-transfer), and every core reaches the kernels\' internal')
        A('     * all-core barriers.  The parity split is what keeps that safe: the outbound')
        A('     * reads Y_BUF[obuf] while compute writes Y_BUF[cbuf], and the producer writes')
        A('     * this cluster\'s X_BUF[obuf] while compute reads X_BUF[cbuf].')
    else:
        A('     * N_ITER == 1: a cluster computes at step 2*stage and sends at step 2*stage+1, so')
        A('     * no compute ever shares a step with this cluster\'s outbound -- there is nothing')
        A('     * to overlap.  The dm_core therefore uses the BLOCKING dma_2d/push_to_next and')
        A('     * every core (incl. dm_core) joins the compute, so the kernels\' internal all-core')
        A('     * barriers are still reached.  Same schedule as the overlapped build, but the')
        A('     * transfer is a self-contained region -- which is what makes it measurable.')
    A('     *')
    A('     *   HBM --[DMA]--> S0 --[DMA]--> S1 --> ... --> S(N-1) --[DMA]--> HBM')
    A('     */')
    A('    for (uint32_t step = 0; step < 2u * (N_STAGES - 1) + N_ITER + 1u; step++) {')
    A('        uint32_t base = 2u * stage;')
    A('        /* compute item (step-2*stage) X_BUF[cbuf] -> Y_BUF[cbuf]; push item')
    A('         * (step-2*stage-1) from Y_BUF[obuf] into the next stage\'s X_BUF[obuf] */')
    A('        uint32_t compute_active  = (cid < N_CLUSTERS) && (step >= base)')
    A('                                   && (step - base < N_ITER);')
    A('        uint32_t outbound_active = (cid < N_CLUSTERS) && (step >= base + 1u)')
    A('                                   && (step - base - 1u < N_ITER);')
    A('        uint32_t cbuf = step & 1u;        /* compute buffer  = item parity */')
    A('        uint32_t obuf = cbuf ^ 1u;        /* outbound buffer = previous item parity */')
    A('')
    A('        if (compute_active || outbound_active) {')
    A('            if (stage == 0u && compute_active) {')
    A('                /* Stage 0 input arrives from HBM this step: its worker cores must')
    if overlap:
        A('                 * not compute until the dm_core load has completed.  (The outbound')
        A('                 * transfer below still overlaps the compute.) */')
    else:
        A('                 * not compute until the dm_core load has completed. */')
    A('                if (flex_is_dm_core()) {')
    A('                    dma_2d((uint64_t)(uintptr_t)X_BUF[cbuf],')
    A('                           hbm_addr(HBM_INPUT_OFFSET),')
    A('                           CLUSTER_0_DIN_BYTES);')
    A('                }')
    A('                flex_intra_cluster_sync();   /* worker cores wait for the HBM load */')
    A('            }')
    if overlap:
        A('            /* (1) dm_core ISSUES the outbound of the previous item -- no wait. */')
        A('            if (flex_is_dm_core() && outbound_active) {')
        A('                if (stage == N_STAGES - 1u) {')
        A('                    dma_2d_issue(hbm_addr(HBM_OUTPUT_OFFSET + lane * STAGE_DOUT_BYTES[stage]),')
        A('                                 (uint64_t)(uintptr_t)Y_BUF[obuf],')
        A('                                 STAGE_DOUT_BYTES[stage]);')
        A('                } else {')
        A('                    push_to_next_issue(stage, lane,')
        A('                                       (uint32_t)(uintptr_t)X_BUF[obuf],')
        A('                                       (uint64_t)(uintptr_t)Y_BUF[obuf]);')
        A('                }')
        A('            }')
        A('            /* (2) ALL cores (incl. dm_core) compute this stage: X_BUF[cbuf] -> Y_BUF[cbuf]. */')
        A('            if (compute_active) {')
        A('                switch (stage) {')
        for i in range(n):
            A(f'                    case {i}: compute_stage_{i}(cbuf); break;')
        A('                    default: break;')
        A('                }')
        A('            }')
        A('            /* (3) dm_core drains its outbound before Y_BUF[obuf] is reused next parity. */')
        A('            if (flex_is_dm_core() && outbound_active) {')
        if bench:
            A('                mempool_start_benchmark();')
        A('                flex_dma_async_wait_all();')
        if bench:
            A('                mempool_stop_benchmark();')
        A('            }')
    else:
        A('            /* (1) dm_core transfers the previous item -- BLOCKING (N_ITER == 1, so no')
        A('             * compute of this cluster shares the step and there is nothing to overlap).')
        A('             * dma_2d/push_to_next carry their own benchmark markers when --bench. */')
        A('            if (flex_is_dm_core() && outbound_active) {')
        A('                if (stage == N_STAGES - 1u) {')
        A('                    dma_2d(hbm_addr(HBM_OUTPUT_OFFSET + lane * STAGE_DOUT_BYTES[stage]),')
        A('                           (uint64_t)(uintptr_t)Y_BUF[obuf],')
        A('                           STAGE_DOUT_BYTES[stage]);')
        A('                } else {')
        A('                    push_to_next(stage, lane,')
        A('                                 (uint32_t)(uintptr_t)X_BUF[obuf],')
        A('                                 (uint64_t)(uintptr_t)Y_BUF[obuf]);')
        A('                }')
        A('            }')
        A('            /* (2) ALL cores (incl. dm_core) compute this stage: X_BUF[cbuf] -> Y_BUF[cbuf]. */')
        A('            if (compute_active) {')
        A('                switch (stage) {')
        for i in range(n):
            A(f'                    case {i}: compute_stage_{i}(cbuf); break;')
        A('                    default: break;')
        A('                }')
        A('            }')
    A('        }')
    A('        flex_global_barrier_xy();   /* joins compute & DMA across all cores, then clusters */')
    A('    }')
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


_C_RENDERERS = {'ctt': _render_ctt_c, 'cwt': _render_cwt_c}


# ---------------------------------------------------------------------------
# App generation
# ---------------------------------------------------------------------------

# Apps root of the mempool multi-cluster build.  Its Makefile discovers apps with
# `find $(APPS_DIR) -name "main.c"` and derives the app name from the directory,
# so each generated app is a FLAT directory holding a single main.c.
_APPS_DIR = (Path(__file__).resolve().parent.parent.parent.parent
             / 'software' / 'apps' / 'multi_cluster')


def app_name(mode: str, pool: str) -> str:
    """Directory (and binary) name of the generated app: cevit_<mode>_<pool>."""
    return f'cevit_{mode}_{pool.replace(" ", "_")}'


# Per-pool L1 geometry.  NUM_BANKS = banking_factor * num_cores (config/*.mk): 8 x 256
# for tensorpool256, 4 x 1024 for terapool -- and both L1s come to 4 MiB.
#   stacks  -- the sequential region the linker puts BELOW .l1 (arch.ld.c):
#              __seq_end = NUM_BANKS * STACK_SIZE / BANKING_FACTOR = num_cores * stack_size
#   runtime -- the runtime's own .l1 objects sharing the region with ours:
#              synchronization.c's log_barrier[NUM_BANKS] and partial_barrier[NUM_BANKS]
#              (4 B each, NUM_BANKS-aligned) plus the scalar barrier and .l1_prio allocator
#   align   -- our buffers' alignment attribute, NUM_BANKS * sizeof(int32_t)
def _l1_geometry(num_cores: int, banking: int, bank_size: int,
                 stack: int = 512) -> Dict[str, int]:
    banks = banking * num_cores
    return {
        'total':   banks * bank_size,
        'stacks':  num_cores * stack,
        'runtime': 2 * banks * 4 + 2 * banks + 64,   # arrays + their alignment + scalars
        'align':   banks * 4,
    }


L1_GEOMETRY = {
    'TensorPool': _l1_geometry(num_cores=256,  banking=8, bank_size=2048),
    'Terapool':   _l1_geometry(num_cores=1024, banking=4, bank_size=1024),
}


def _l1_footprint(clusters: List[ClusterSplit], mode: str, wait: bool,
                  pool: str) -> Tuple[int, str]:
    """(bytes, breakdown) of the per-cluster L1 the generated app declares.

    X and Y are doubled in cwt (item parity); W is resident either way.  Each buffer
    is placed on the pool's alignment boundary, so the gaps count too -- with tensors
    this close to the TCDM ceiling a few KiB of padding decides the link.
    """
    nx, nw, ny = _scratch_elems(clusters, wait)
    dup  = 2 if mode == 'cwt' else 1
    geo  = L1_GEOMETRY.get(pool, {'stacks': 0, 'align': 64})
    # walk the declarations in emission order from the start of .l1, aligning each
    sizes = [dup * nx * BYTES_PER_ELEM, nw * BYTES_PER_ELEM, dup * ny * BYTES_PER_ELEM]
    at    = geo['stacks']
    for sz in sizes:
        at = ((at + geo['align'] - 1) // geo['align']) * geo['align'] + sz
    total = at - geo['stacks']
    d     = '2x' if dup == 2 else ''
    parts = (f'{d}X {nx * BYTES_PER_ELEM}B + W {nw * BYTES_PER_ELEM}B '
             f'+ {d}Y {ny * BYTES_PER_ELEM}B + {(total - sum(sizes)) / 1024:.0f} KiB align')
    return total, parts


def generate_apps(res: Results, apps_dir: str, cycles_per_ms: int,
                  mode: str, n_iter: int, bench: bool = False,
                  wait: bool = False, l1_bytes: int = 4 * 1024 * 1024) -> None:
    # cwt overlaps compute & DMA, so it uses the max(compute, transfer) split;
    # ctt serializes them and uses the compute + transfer split.
    src = res.clusters_overlap if mode == 'cwt' else res.clusters
    if not src:
        print('Warning: no cluster splits found.')
        return

    compute_kind = 'mempool_wait (timing-only)' if wait else 'real baremetal kernels'
    compute_kind += ', benchmarked' if bench else ', uninstrumented'
    if mode == 'ctt':
        print(f'\nGenerating compute-then-transfer pipeline apps '
              f'({n_iter} iterations, {compute_kind}) -> {apps_dir}/')
    else:  # cwt
        print(f'\nGenerating compute-while-transfer pipeline apps '
              f'({n_iter} iterations, {compute_kind}) -> {apps_dir}/')

    for pool, clusters in src.items():
        if not clusters:
            continue
        app     = app_name(mode, pool)
        app_dir = Path(apps_dir) / app
        app_dir.mkdir(parents=True, exist_ok=True)

        c_src = _C_RENDERERS[mode](pool, clusters, res.row_latencies[pool],
                                   cycles_per_ms, n_iter, bench, wait,
                                   res.gather_fold)
        (app_dir / 'main.c').write_text(c_src)

        print(f'  [{pool}]  {app_dir / "main.c"}')
        l1, parts = _l1_footprint(clusters, mode, wait, pool)
        geo       = L1_GEOMETRY.get(pool, {})
        stacks    = geo.get('stacks', 0)
        runtime   = geo.get('runtime', 0)
        # what a generated app can actually declare: TCDM minus the stacks below .l1
        # and minus the runtime's own objects inside it
        budget    = min(l1_bytes, geo.get('total', l1_bytes)) - stacks - runtime
        print(f'  [{pool}]  L1 per cluster: {l1 / 1024:.1f} KiB of {budget / 1024:.0f} KiB '
              f'({parts};  budget = {geo.get("total", l1_bytes) / 1024:.0f} KiB TCDM '
              f'- {stacks / 1024:.0f} KiB stacks - {runtime / 1024:.0f} KiB runtime)')
        if l1 > budget:
            print(f'  [{pool}]  WARNING: over budget by {(l1 - budget) / 1024:.1f} KiB -- '
                  f'the .l1 section will not link.  The attention stage sets this: its '
                  f'MxM score matrix sits in X and again in Y, and head parallelism does '
                  f'not shrink it (M is per head).  Use a smaller template.')
        print(f'  [{pool}]  build with:  make -C {apps_dir} {app}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_cluster_split(plt, pool: str, clusters: List[ClusterSplit],
                        overlap: bool, out_path: Path) -> None:
    """Bar plot of per-cluster wall time: compute (red) vs transfer (green).

    One x position per PIPELINE STAGE.  A head-parallel stage owns several clusters,
    so its clusters are drawn SIDE BY SIDE inside the stage's slot (shaded, and
    labelled with their cluster ids) -- the picture then shows directly that the
    stage's wall is the height of one lane, not the sum of them.

    Serialized split -> stacked bars   (wall = compute + transfer).
    Overlapped split -> grouped bars    (wall = max(compute, transfer)).

    In the overlapped picture stage 0 carries a third quantity: its LPDDR data_in read,
    which the generated loop serializes AHEAD of the compute and so cannot overlap.  It
    is drawn hatched as the pedestal both the compute bar and the overlapped hand-off
    bar stand on, which is exactly how it enters the wall: din + max(compute, dout).
    Every other stage has din = 0 and reduces to the plain grouped pair.
    """
    bases    = _stage_bases(clusters)
    n_stages = len(clusters)
    tag      = 'CWT' if overlap else 'CTT'
    n_phys   = sum(cs.n_par for cs in clusters)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.3 * n_phys + 2.5), 4.4))

    span      = 0.82                  # width of a stage slot
    first     = True                  # label only the first bars, for one legend entry
    first_din = True                  # ... and only the first exposed-data_in bar
    tick_pos, tick_lab = [], []

    for s, (cs, base) in enumerate(zip(clusters, bases)):
        p = cs.n_par
        # lanes of a stage are identical replicas -- one head each -- so every bar in
        # the group has the same height; drawing them all is the point of the picture.
        slots = 2 * p if overlap else p            # bars in this stage's slot
        bw    = span / slots
        # keep a lone cluster's bar from ballooning to the full slot width
        draw  = min(bw * 0.9, 0.30 if overlap else 0.34)
        left  = s - span / 2
        wall  = _overlap_wall(cs) if overlap else cs.latency + cs.transfer_ms

        if p > 1:
            ax.axvspan(left - 0.03, left + span + 0.03, color='0.92', zorder=0)
        for lane in range(p):
            if overlap:
                xmid = left + (2 * lane + 1.0) * bw
                # stage 0's data_in read is exposed: it delays BOTH the compute and the
                # overlapped hand-off, so both bars start on top of it.
                dout = cs.transfer_ms - cs.din_ms
                if cs.din_ms > 0.0:
                    for x in (xmid - bw / 2, xmid + bw / 2):
                        ax.bar(x, cs.din_ms, width=draw, color='green', hatch='//',
                               edgecolor='white', zorder=2,
                               label=('data_in (not overlapped)'
                                      if first_din and x < xmid else None))
                    first_din = False
                ax.bar(xmid - bw / 2, cs.latency, width=draw, bottom=cs.din_ms,
                       color='red', zorder=2, label='compute' if first else None)
                ax.bar(xmid + bw / 2, dout, width=draw, bottom=cs.din_ms,
                       color='green', zorder=2, label='transfer' if first else None)
            else:
                xmid = left + (lane + 0.5) * bw
                ax.bar(xmid, cs.latency,     width=draw, color='red', zorder=2,
                       label='compute'  if first else None)
                ax.bar(xmid, cs.transfer_ms, width=draw, bottom=cs.latency,
                       color='green', zorder=2, label='transfer' if first else None)
            first = False
            # cluster id above its own bar(s), clear of the stage ticks below the axis
            ax.annotate(f'c{base + lane}', xy=(xmid, wall), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7, color='0.35')

        tick_pos.append(s)
        tick_lab.append(f'stage {s}' + (f'\n({p} clusters || )' if p > 1 else ''))

    # y range follows the data: tallest bar of this split plus headroom for the
    # cluster-id annotations sitting on top of it and for the legend box.
    top = max((_overlap_wall(cs) if overlap else cs.latency + cs.transfer_ms)
              for cs in clusters)
    ax.set_ylim(0.0, top * 1.20 if top > 0.0 else 1.0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_xlim(-0.6, n_stages - 0.4)
    ax.set_xlabel('pipeline stage  (clusters of a stage run in parallel)')
    ax.set_ylabel('wall time [ms]')
    ax.set_title(f'{pool} -- {tag}  |  {n_stages} stages over {n_phys} clusters')
    ax.legend(loc='upper right', fontsize='small')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_plots(res: Results, mode: Optional[str], out_dir: Path) -> None:
    """Save per-pool bar plots of the relevant cluster split(s) to out_dir.

    Plots the serialized split for ctt, the overlapped split for cwt, and both
    when mode is None -- one PNG per (pool, split).  matplotlib is imported
    lazily; if it is missing the plots are skipped with a warning.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless: write PNGs, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print('Warning: matplotlib not installed; --plot skipped '
              '(pip install matplotlib).')
        return

    want_serial  = mode in (None, 'ctt')
    want_overlap = mode in (None, 'cwt')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nSaving cluster-split plots -> {out_dir}/')
    for p in res.clusters:
        psafe = p.replace(' ', '_')
        if want_serial and res.clusters.get(p):
            path = out_dir / f'CEViT_CTT_split_{psafe}.png'
            _plot_cluster_split(plt, p, res.clusters[p], False, path)
            print(f'  [plot] {path}')
        if want_overlap and res.clusters_overlap.get(p):
            path = out_dir / f'CEViT_CWT_split_{psafe}.png'
            _plot_cluster_split(plt, p, res.clusters_overlap[p], True, path)
            print(f'  [plot] {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_summary(res: Results, path: str, csv_name: str) -> None:
    """Write a machine-readable per-(pool, mode) split summary for antenna_analysis.py.

    One row per pool per mode.  ctt uses the serialized split
    (res.clusters / res.best_throughput); cwt uses the overlapped split
    (res.clusters_overlap / res.best_throughput_overlap).  Pools without a split are
    skipped.  Columns:
    pool, mode, throughput_ms (= T_pipe, the bottleneck stage wall), n_stages,
    n_clusters, latency_ms.

    latency_ms is the end-to-end time of ONE item IN STEADY STATE, and that is not the
    critical path sum_s(compute_s + transfer_s): both generated loops rendezvous every
    cluster on a flex_global_barrier_xy() per step, so a stage that finishes early waits
    for the slowest one and every stage costs a full period T_pipe.  An item therefore
    spends T_pipe per step it is resident, and the wavefront fixes how many steps that is:

      * ctt -- stage s handles item m at step m+s, and the item's last transfer is in
        step m+n_stages-1:  latency = n_stages * T_pipe.
      * cwt -- the double-buffered schedule puts stage s's compute of item m at step
        m+2s (push lags compute by one step, and the consumer may only start the step
        after the push), with the final HBM write at step m+2*n_stages-1:
        latency = 2 * n_stages * T_pipe.

    So cwt shortens the period but stretches the number of steps an item is in flight --
    the overlap buys THROUGHPUT, and here it costs latency rather than leaving it alone.

    T_pipe is the SLOWEST STAGE's wall in cwt, and that wall charges stage 0 its LPDDR
    data_in read outside the overlap max, since the generated loop serializes that read
    ahead of the compute (see ``_overlap_wall``).  That is where the exposed first load
    is paid -- it is NOT an extra step on top of the 2*n_stages an item is resident.
    """
    # (mode, splits-dict, best-throughput-dict, steps an item is resident per stage)
    views = (
        ('ctt', res.clusters,         res.best_throughput,         1),
        ('cwt', res.clusters_overlap, res.best_throughput_overlap, 2),
    )
    lines = [
        '# CEViT pipeline summary  (read by antenna_analysis.py)',
        f'# csv: {csv_name}',
        '# columns: pool  mode  throughput_ms  n_stages  n_clusters  latency_ms',
    ]
    for mode, splits, best, steps_per_stage in views:
        for pool, stages in splits.items():
            if not stages:
                continue
            n_stages   = len(stages)
            n_clusters = sum(cs.n_par for cs in stages)
            # steady state: every step costs the bottleneck period, barriers included
            latency    = steps_per_stage * n_stages * best[pool]
            lines.append(f'{pool}\t{mode}\t{best[pool]:.6g}\t{n_stages}\t{n_clusters}'
                         f'\t{latency:.6g}')

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')
    print(f'\nWrote pipeline summary -> {path}')


def run(template_path: str, config_path: str,
        split_model: str = 'ctt') -> Results:
    """Programmatic entry point: returns the populated Results struct."""
    cfg  = load_config(config_path)
    grid = read_grid(template_path)

    header_row = find_header_row(grid)
    cols       = build_column_map(grid[header_row])
    data_start = header_row + 1
    data_end   = find_data_end(grid, cols, data_start)   # exclusive
    check_for_errors(grid, cols, data_start, data_end - 1)
    layers     = parse_layers(grid, cols, data_start, data_end - 1)
    if not layers:
        raise ValueError("No data rows found below the header row")

    # The hand-off DMA re-lays these out on its way to the next stage, so they never
    # reach the model.  Say so: they are in the template but in none of the tables.
    layers, folded = drop_folded_layout_layers(layers)
    for r in folded:
        print(f'Folded into the stage hand-off: row {r.excel_row} '
              f"'{r.op}' [{r.kernel}], {int(r.dout_bytes)} B")

    # Same on the way out of the head-parallel stage: the gather re-lays the heads out as
    # its address pattern, so those rows leave no local copy behind either.
    layers, gather_fold = fold_gathered_layout_layers(layers, HEAD_PARALLEL_CLUSTERS)
    if gather_fold:
        for r in gather_fold.rows:
            print(f'Folded into the gather hand-off: row {r.excel_row} '
                  f"'{r.op}' [{r.kernel}], {int(r.dout_bytes)} B")
        print(f'  gather descriptor: chunk {gather_fold.row_bytes} B x '
              f'{gather_fold.reps} reps, dst_stride '
              f'{gather_fold.row_bytes * HEAD_PARALLEL_CLUSTERS} B')

    # The bookend layout rows are the patchify / unpatchify and run on the cores, not on the
    # local DMA.  Must precede compute(): it changes what those rows cost.
    for r in mark_pe_layout_rows(layers):
        g = _patch_geometry(r)
        print(f'PE-run layout ({r.pe_layout}): row {r.excel_row} '
              f"'{r.op}' [{r.kernel}], grid [{g['Nf']}, {g['Nt']}, {g['C']}], "
              f"patch {g['Nh']}x{g['Nw']}")

    res = compute(layers, cfg, split_model)
    res.gather_fold = gather_fold
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
        help='Limit to one pipeline flavour: '
             'ctt = compute-then-transfer, cwt = compute-while-transfer.  '
             "Prints only that mode's cluster split and generates only its "
             'apps.  Omit to print both splits and generate both.',
    )
    ap.add_argument('--iterations', type=int, default=1, metavar='N',
                    help='Items streamed through the pipeline for ctt/cwt '
                         '(default: 1)')
    ap.add_argument('--out', metavar='DIR', default=str(_APPS_DIR),
                    help='Apps root directory under which the generated '
                         'cevit_<mode>_<pool>/main.c app dirs are created '
                         f'(default: {_APPS_DIR}).')
    ap.add_argument('--summary', metavar='PATH', default='pipeline_summary.txt',
                    help='Write a machine-readable per-(pool, mode) split summary '
                         '(throughput_ms, n_stages, n_clusters) for antenna_analysis.py '
                         '(default: ./pipeline_summary.txt).')
    ap.add_argument('--bench', action='store_true',
                    help='Bracket every layer and every DMA of the generated app with '
                         'mempool_start_benchmark()/mempool_stop_benchmark(), so an RTL run '
                         'traces exactly the kernels and the transfers.  The markers are two '
                         '`trace` CSR writes -- they read no counter and print nothing.  Off '
                         'by default: the app then runs the raw kernels with no '
                         'instrumentation and no output at all.  flex_timer_start/end (the '
                         'SoC perf counter for the whole execution region) is emitted either '
                         'way.')
    ap.add_argument('--wait', action='store_true',
                    help='Retro-compatibility: emit the timing-only pipeline where every '
                         'layer is a mempool_wait() spin (the pre-real-kernel behaviour), '
                         'instead of the default which lowers each layer to a real '
                         'baremetal kernel call (GEMM on the RedMulE tensor engines for '
                         'TensorPool / on the PE matmul for Terapool; softmax, layernorm, '
                         'gelu, add and the bookend patchify/unpatchify on the PE path; the '
                         'remaining reshape/split/transpose as DMA copies).  --wait needs no '
                         'tensor-shape columns in the template.')
    ap.add_argument('--split-model', choices=['ctt', 'cwt'], default='ctt',
                    metavar='{ctt,cwt}',
                    help='Wall model the STAGE BOUNDARIES are optimized for (default: '
                         'ctt).  Both modes always run the SAME split -- the same layers '
                         'on the same stages -- so ctt and cwt differ only in their '
                         'schedule; this only picks which of the two the cuts are '
                         'balanced for.  The other mode then reports its true period on '
                         'those cuts.')
    ap.add_argument('--l1-bytes', type=int, default=4 * 1024 * 1024, metavar='N',
                    help='Cluster TCDM budget in bytes, used only to warn when the '
                         'generated app\'s X/W/Y buffers do not fit (default: 4 MiB, the '
                         'cluster_tcdm_size of the TensorPool and Terapool arch configs).')
    ap.add_argument('--plot', action='store_true',
                    help='Save a per-pool bar plot of the cluster split(s) as PNGs '
                         'under ./cevit_split_plots/: compute (red) and transfer (green), '
                         'stacked for the serialized split (ctt) and side-by-side '
                         'for the overlapped split (cwt); the y-axis scales to the '
                         'tallest bar of each plot.  '
                         'Which split(s) follow --mode (both when omitted).')
    args = ap.parse_args()

    if args.iterations < 1:
        ap.error('--iterations must be >= 1')

    cfg = load_config(args.config)
    res, layers = run(args.template, args.config, args.split_model)

    # Always print the tables; the cluster split shows both flavours when no
    # --mode is given, or only the selected one otherwise.
    print_results(layers, res, args.mode)

    # Export the machine-readable summary for the antenna-capacity analysis.
    # compute() always builds both splits, so every (pool, mode) is available
    # regardless of --mode.
    if res.clusters or res.clusters_overlap:
        write_summary(res, args.summary, Path(args.template).name)

    modes = [args.mode] if args.mode else ['ctt', 'cwt']
    if not res.clusters:
        print('\nWarning: no cluster splits found -- nothing generated.')
    else:
        frequency     = cfg.get('frequency', 1000000000)
        cycles_per_ms = int(frequency // 1000)
        for m in modes:
            generate_apps(res, args.out, cycles_per_ms, m, args.iterations,
                          args.bench, args.wait, args.l1_bytes)

    if args.plot:
        if not res.clusters:
            print('\nWarning: no cluster splits to plot.')
        else:
            generate_plots(res, args.mode, Path.cwd() / 'cevit_split_plots')


if __name__ == '__main__':
    main()
