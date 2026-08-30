# AI-PHY pipeline parsers

Two performance models, one per network.

| tool | what it does |
| --- | --- |
| `CEViT/cevit_pipeline.py` | costs the CEViT layer template and generates its GVSOC pipeline apps |
| `CELERE/celere_pipeline.py` | the same for CELERE |

Each generator reads a layer template (`.csv`, `.xlsx`, `.xlsm` or `.ods`) and a JSON hardware
config, prints a per-row and per-stage latency report, writes a machine-readable summary, and
emits one multi-cluster app per `(mode, pool)` under `software/apps/multi_cluster/`.

**The partition is not searched for.** It lives in the `STAGES` table inside each script,
together with the buffer-binding tables beside it. The config carries hardware numbers only —
peaks, utilisations, bandwidths, clock. Changing how the work is split means editing the
script, and each script's module docstring explains its own partition in full.

## Scope and status

**Both partitions are tuned for TensorPool, not Terapool.** The stage widths were chosen
against TensorPool's RedMulE GEMM peak, where the periods come out ~5–7x lower (CEViT ctt:
0.071 ms against Terapool's 0.510; CELERE ctt: 0.135 against 0.788). On Terapool a GEMM is a
PE matmul and the balance moves — in CEViT the attention core becomes the wall instead of the
FFN block — so a Terapool-first partition would want different widths, not just a different
config. The Terapool numbers are honest output of the same model; they are simply not what the
shape was optimised for.

**CELERE was taken as far as ctt on TensorPool.** That is the configuration whose stage widths,
buffer bindings and LPDDR layout were actually tuned; the other three apps are generated and
modelled but are a starting point rather than a result. Its ctt build is 3691 KiB and fits
TensorPool's 3948 KiB budget, but **the same app comes out 3699 KiB on Terapool, 155 KiB over
that pool's 3544** — Terapool spends 512 KiB of TCDM on stacks against TensorPool's 128.
Raising the last stage's `tiles` does not help there: what puts it over is stage 5's doubled
*inbound* landing zone, and `tiles` only divides the drain. **cwt fits neither pool** — the
CELERE emitter doubles the whole `X_BUF`, which lands 1839 KiB over on TensorPool and 2251 KiB
over on Terapool. CEViT is the more finished of the two: all four of its apps fit, between
2663 and 2718 KiB.

**cwt is not worth its cost on either model as they stand.** It buys 2.1% of the period on CEViT
/ TensorPool, 0.3% on CEViT / Terapool, 1.4% on CELERE / Terapool and *nothing at all* on CELERE
/ TensorPool, where the tiled last stage is the bottleneck in both modes and drains inside its
own compute either way. Quote ctt unless the transfer column grows.

## Running them

```bash
cd scripts/parser/CEViT
python3 cevit_pipeline.py CEViT_273PRBs.csv config.json

cd scripts/parser/CELERE
python3 celere_pipeline.py CELERE_273PRBs_64ant.csv config.json
```

Both take the same flags:

| flag | meaning |
| --- | --- |
| `--mode {ctt,cwt}` | generate only one flavour; omit for both |
| `--iterations N` | items streamed through the pipeline (default 1) |
| `--out DIR` | apps root (default `software/apps/multi_cluster`) |
| `--summary PATH` | where to write `pipeline_summary.txt` |
| `--wait` | emit the timing-only twin: every layer a `mempool_wait()` spin sized to the modelled latency, instead of a real kernel |
| `--bench` | bracket every layer with `mempool_start/stop_benchmark()` for an RTL trace |
| `--l1-bytes N` | TCDM budget for the over-budget warning (default 4 MiB) |
| `--plot` | save per-stage bar plots under `./<model>_split_plots/` |

### The two modes

Both stream items as a wavefront: stage `s` (0-based) is skewed by a fixed `phase`, and at
step `step` it works on item `step - phase`. The two modes differ in that skew.

* **ctt** — compute-then-transfer. **Skew `s` per stage**, so stage `s` works on item `m` at
  step `m + s`. A cluster computes its stage and then transfers, back to back with **no barrier
  between them**, and one global barrier closes the step. Period = `max_s(compute_s +
  transfer_s)`: a cluster serialises its own two halves, but two stages with different
  bottlenecks cover for each other, so the slowest compute and the slowest transfer are only
  added when they fall on the same stage.
* **cwt** — compute-while-transfer. **Skew `2s` per stage**, so stage `s` works on item `m` at
  step `m + 2s`. That leaves a spare step between a producer and its consumer for the transfer
  to run in, so the transfer runs *beside* the compute rather than after it. One barrier a step
  as well. Period = the slowest stage's own wall, `din + max(compute, transfer)`.

(`phase` in the generated `main.c` is `stage - 1` for ctt and `2 * (stage - 1)` for cwt.)

**Neither mode has a barrier between compute and transfer**, so in both a predecessor can be
pushing while its consumer computes, and **both double the hand-off landing zone** — indexed by
the item's low bit, so a producer writes the copy its consumer is not reading. What separates
them is the *outbound* end: under ctt a push completes before the barrier that ends the step, so
its source is never live across a compute and needs no twin, while cwt deliberately leaves the
drain running beside the next compute and does need one. Only what crosses an item boundary is
ever doubled; the working set, the LPDDR landing zones and the weights are single in both.

cwt shortens the period and lengthens the time in flight: it buys throughput and pays in
latency. Both are modelled and both are generated, for both pools (TensorPool and Terapool) —
four apps per model.

## Building the generated apps

Each generator prints the exact `make` line per app. In short:

```bash
make -C software/apps/multi_cluster cevit_ctt_TensorPool  COMPILER=llvm config=terapool
make -C software/apps/multi_cluster celere_ctt_TensorPool COMPILER=llvm config=tensorpool256
```

* `COMPILER=llvm` is required — the FP16 kernels do not build with GCC.
* **`config=` picks the grid, not the pool.** `num_cluster_x` / `num_cluster_y` in
  `software/runtime/multi_cluster/configs/arch_*.py` are `_Static_assert`ed by the emitted app:
  CEViT is laid out for **4 x 3** and CELERE for **4 x 8**. So every CEViT app builds under
  `config=terapool` and every CELERE app under `config=tensorpool256` — *whichever pool it is
  modelled against*. The pool in the app's name is the cost model; the config is the machine it
  is placed on, and the two are named alike only by accident.
* The default `config=mempool` builds neither: 4 x 4 and a 1 MiB L1, so the grid assert fires
  or the link fails for an unrelated-looking reason.
* `hbm_chan_placement` must stay `[0,4,0,4]` (four LPDDR nodes on the north edge and four on
  the south); `terapool` and `tensorpool256` both already carry it.

## Plotting

`--plot` on either generator saves per-stage bar plots under `./<model>_split_plots/` —
per-cluster compute against transfer, stage by stage, which is where a bottleneck is actually
visible.

Both generators also write a `pipeline_summary.txt` with one row per `(pool, mode)`, for
whatever wants to read the headline numbers without re-running the model:

```
# columns: pool  mode  throughput_ms  n_stages  n_clusters  latency_ms
```

`throughput_ms` is `T_pipe`, the bottleneck period. `latency_ms` is **not** the critical-path
sum of the stages: the generated pipelines barrier every step, so each step costs `T_pipe` and
an item is resident for one step per stage in ctt and two in cwt.

## Config keys

Both `config.json` files carry the same keys:

| key | meaning |
| --- | --- |
| `PE_peak`, `GEMM_peak` | per-pool peaks in TFLOPS |
| `PE_utilization` | derates the PE path: non-GEMM everywhere, plus GEMMs on a pool with no tensor engine |
| `TE_utilization` | derates `GEMM_peak` only on pools that have a tensor engine (TensorPool's RedMulE) |
| `bandwidth_c2c_gbs` | on-chip cluster-to-cluster link, GB/s |
| `bandwidth_lpddr_gbs` | **per channel**, GB/s |
| `lpddr_channels` | channels an off-chip transfer is striped across |
| `frequency` | clock, Hz |

An off-chip leg is charged at `bandwidth_lpddr_gbs * lpddr_channels`, **capped by**
`bandwidth_c2c_gbs`: the bytes still cross the on-chip network to reach the memory-side node,
so extra channels stop helping once the aggregate passes that link. CELERE's config also
carries `pipeline_clusters`, which is only a cross-check against its stage table.

## What the models do and do not claim

* The emitted apps move **placeholder data** (fp16 1.0) and check no result. What they measure
  is cycles and DMA traffic, not numerical correctness.
* **Distance is not charged.** Every transfer is billed its bytes and nothing for the hops it
  takes. Both scripts name the specific legs that reach sideways; re-measure against RTL before
  treating those as free.
* Layout rows (reshape / split / transpose) have no baremetal kernel. Where a transfer was
  crossing that boundary anyway it performs the row for free; otherwise the row is a
  `mempool_wait()` spin sized by a fit taken on `patchify_f16`, which over-counts a contiguous
  permutation. Each script's `DROPPED_ROWS` says which rows are in which case, and CEViT's row
  13 is flagged there as a deliberate optimism.
