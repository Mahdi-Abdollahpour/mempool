# AI-PHY pipeline parsers

Two performance models, one per network, plus a shared plotting helper.

| tool | what it does |
| --- | --- |
| `CEViT/cevit_pipeline.py` | costs the CEViT layer template and generates its GVSOC pipeline apps |
| `CELERE/celere_pipeline.py` | the same for CELERE |
| `latency_throughput_bars.py` | plots the `pipeline_summary.txt` either generator writes |

Each generator reads a layer template (`.csv`, `.xlsx`, `.xlsm` or `.ods`) and a JSON hardware
config, prints a per-row and per-stage latency report, writes a machine-readable summary, and
emits one multi-cluster app per `(mode, pool)` under `software/apps/multi_cluster/`.

**The partition is not searched for.** It lives in the `STAGES` table inside each script,
together with the buffer-binding tables beside it. The config carries hardware numbers only —
peaks, utilisations, bandwidths, clock. Changing how the work is split means editing the
script, and each script's module docstring explains its own partition in full.

## Scope and status

**Both partitions are tuned for TensorPool, not Terapool.** The stage widths were chosen
against TensorPool's RedMulE GEMM peak, where the periods come out ~5–6x lower (CEViT ctt:
0.086 ms against Terapool's 0.526; CELERE ctt: 0.159 against 0.787). On Terapool a GEMM is a
PE matmul and the balance moves — in CEViT the attention core becomes the wall instead of the
FFN block — so a Terapool-first partition would want different widths, not just a different
config. The Terapool numbers are honest output of the same model; they are simply not what the
shape was optimised for.

**CELERE was taken as far as ctt on TensorPool.** That is the configuration whose stage widths,
buffer bindings and LPDDR layout were actually tuned; the other three apps are generated and
modelled but are a starting point rather than a result. Its ctt build fits both pools at
2803 KiB, but **cwt fits neither** — it doubles the whole `X_BUF` and comes out 943 KiB over on
TensorPool and 1355 KiB over on Terapool, so those two apps will not link until the last
stage's `tiles` is raised or the doubling is made partial the way `cevit_pipeline.py` does it.
CEViT is the more finished of the two: all four of its apps fit, between 2206 and 2581 KiB.

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
  step `m + s`. Two global barriers a step, one L1 buffer set, compute and DMA serialized.
  Period = slowest compute + slowest transfer, taken over the whole partition, because the
  barrier stops the two covering for each other.
* **cwt** — compute-while-transfer. **Skew `2s` per stage**, so stage `s` works on item `m` at
  step `m + 2s`. That leaves a spare step between a producer and its consumer for the transfer
  to run in, and needs only one barrier a step. Period = the slowest stage's own wall. Only
  what crosses an item boundary takes a second L1 copy.

(`phase` in the generated `main.c` is `stage - 1` for ctt and `2 * (stage - 1)` for cwt.)

cwt shortens the period and lengthens the time in flight: it buys throughput and pays in
latency. Both are modelled and both are generated, for both pools (TensorPool and Terapool) —
four apps per model.

## Building the generated apps

Each generator prints the exact `make` line per app. In short:

```bash
make -C software/apps/multi_cluster cevit_ctt_TensorPool COMPILER=llvm config=tensorpool256
```

* `COMPILER=llvm` is required — the FP16 kernels do not build with GCC.
* `config=tensorpool256` is required — the default `config=mempool` gives a 1 MiB L1 and the
  link fails for an unrelated-looking reason.
* **The grid must match the partition.** `num_cluster_x` / `num_cluster_y` in
  `software/runtime/multi_cluster/configs/arch_*.py` are `_Static_assert`ed by the emitted app:
  CEViT needs **4 x 3**, CELERE needs **4 x 8**. The configs currently carry 4 x 3, so building
  the CELERE apps means raising `num_cluster_y` to 8 first. `hbm_chan_placement` must stay
  `[0,4,0,4]` (four LPDDR nodes on the north edge and four on the south) for either.

## Plotting

Both generators write a `pipeline_summary.txt` with one row per `(pool, mode)`:

```
# columns: pool  mode  throughput_ms  n_stages  n_clusters  latency_ms
```

`latency_throughput_bars.py` turns that into two bar charts — latency per execution on the
left, throughput at the bottleneck on the right:

```bash
python3 latency_throughput_bars.py CEViT/pipeline_summary.txt --out lt_cevit.png
python3 latency_throughput_bars.py CELERE/pipeline_summary.txt --cwt
```

Latency is **not** the critical-path sum of the stages: the generated pipelines barrier every
step, so each step costs the bottleneck period `T_pipe` and an item is resident for one step
per stage in ctt and two in cwt.

`--plot` on either generator is the complementary view — per-cluster compute against transfer,
stage by stage, which is where a bottleneck is actually visible.

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
