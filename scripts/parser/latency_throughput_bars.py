#!/usr/bin/env python3
"""
Fixed latency and throughput of each CEViT pipeline configuration, as two bar charts.

The model is fixed, so every (pool, mode) configuration has exactly one latency and one
throughput -- there is nothing to sweep.  Reads the cevit_pipeline.py --summary file and,
per config:

    latency    = steps * T_pipe  [ms]        -- one execution end-to-end in steady state
                 (the summary's latency_ms column)
    throughput = 1 / T_pipe  [antennas/ms]   -- completions per ms at the bottleneck

Latency is NOT the critical path sum_s(compute_s + transfer_s): the generated pipelines
put a global barrier on every step, so a stage that finishes early waits for the slowest
one and each step costs the bottleneck period T_pipe.  An item is resident for one step
per stage in ctt (latency = n_stages * T_pipe) and two in cwt, whose double-buffered
schedule offsets consecutive stages by two steps (latency = 2 * n_stages * T_pipe).  So
cwt shortens the period and lengthens the flight: the two modes separate in BOTH panels.

T_pipe itself is the slowest STAGE in cwt -- and that stage's wall charges stage 0 the
LPDDR data_in read on top of max(compute, dout), because the generated loop serializes
that read in front of the compute.  The pipeline's one exposed load is paid inside
T_pipe, which is why cwt's flight stays at exactly 2 * n_stages steps.

Left panel = latency per config; right panel = throughput per config.  Values labelled to
4 decimals.

Usage
-----
    python latency_throughput_bars.py pipeline_summary.txt
    python latency_throughput_bars.py pipeline_summary.txt --out lt.png
    python latency_throughput_bars.py pipeline_summary.txt --cwt      # one mode only
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class PipelineRecord:
    pool:       str
    mode:       str
    t_pipe:     float   # bottleneck stage wall [ms] = steady-state period
    n_stages:   int
    n_clusters: int
    latency:    float   # end-to-end wall of one item [ms] = steps * T_pipe


def parse_summary(path: str) -> Tuple[List[PipelineRecord], str]:
    """Parse the cevit_pipeline --summary file into (records, csv_name).

    The 6th column (latency_ms) is optional: summaries written before it existed fall
    back to n_stages * T_pipe, which is the ctt steady-state latency and under-reports
    cwt by its factor of 2.
    """
    records: List[PipelineRecord] = []
    csv_name = ''
    stale    = False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            if line[1:].strip().lower().startswith('csv:'):
                csv_name = line.split(':', 1)[1].strip()
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f'Malformed summary row (need >= 5 columns): {raw!r}')
        pool, mode, t_pipe, n_stages, n_clusters = fields[:5]
        if len(fields) >= 6:
            latency = float(fields[5])
        else:
            latency = int(n_stages) * float(t_pipe)
            stale   = True
        records.append(PipelineRecord(pool, mode, float(t_pipe),
                                      int(n_stages), int(n_clusters), latency))
    if not records:
        raise ValueError(f'No pipeline rows found in {path}')
    if stale:
        print(f'Warning: {path} has no latency_ms column; approximating latency as '
              'n_stages * T_pipe.  Re-run cevit_pipeline.py to refresh it.')
    return records, csv_name


def latency_ms(rec: PipelineRecord) -> float:
    return rec.latency


def throughput_ant_ms(rec: PipelineRecord) -> float:
    return 1.0 / rec.t_pipe if rec.t_pipe > 0 else 0.0


def config_label(rec: PipelineRecord) -> str:
    return f'{rec.pool}\n{rec.mode.upper()}'


def print_metrics(records: List[PipelineRecord]) -> None:
    print(f'\n{"pool":<11}{"mode":<5}{"latency[ms]":>13}{"throughput[ant/ms]":>21}')
    for r in records:
        print(f'{r.pool:<11}{r.mode.upper():<5}'
              f'{latency_ms(r):>13.4f}{throughput_ant_ms(r):>21.4f}')


def plot_bars(records: List[PipelineRecord], csv_name: str, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless: write a PNG, never open a window
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print('Warning: matplotlib not installed; PNG skipped '
              '(pip install matplotlib).  Text table above still printed.')
        return

    order = {'ctt': 0, 'cwt': 1}
    recs  = sorted(records, key=lambda r: (r.pool, order.get(r.mode, 99)))

    pools  = sorted({r.pool for r in recs})
    cmap   = LinearSegmentedColormap.from_list('bluered', ['#1f77b4', '#d62728'])
    colour = [cmap(pools.index(r.pool) / max(1, len(pools) - 1)) for r in recs]
    hatch  = ['//' if r.mode == 'cwt' else '' for r in recs]
    labels = [config_label(r) for r in recs]
    xpos   = list(range(len(recs)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    def draw_decimal(ax, values, ylabel, title):
        """Plain bars with a 4-decimal label (used for latency, which is a time)."""
        bars = ax.bar(xpos, values, color=colour, edgecolor='white')
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        for x, v in zip(xpos, values):
            ax.text(x, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(values) * 1.15 if values else 1)      # headroom for labels
        _finish(ax, ylabel, title)

    def draw_antennas(ax, values, ylabel, title):
        """Antennas are countable: draw floor(v) whole antennas solid + the fractional
        antenna as a lighter cap, headline the integer, keep the exact 4-decimal value."""
        for x, v, c, h in zip(xpos, values, colour, hatch):
            whole = math.floor(v)
            ax.bar(x, whole, color=c, edgecolor='white', hatch=h)
            frac = v - whole
            if frac > 1e-9:
                ax.bar(x, frac, bottom=whole, color=c, edgecolor='white',
                       hatch=h, alpha=0.30)
            ax.annotate(f'{whole:d}', (x, v), textcoords='offset points', xytext=(0, 2),
                        ha='center', va='bottom', fontsize=13, fontweight='bold')
            ax.annotate(f'{v:.4f}', (x, v), textcoords='offset points', xytext=(0, 19),
                        ha='center', va='bottom', fontsize=8, color='0.35')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(values) * 1.24 if values else 1)          # room for both labels
        _finish(ax, ylabel, title)

    def _finish(ax, ylabel, title):
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    draw_decimal(axL, [latency_ms(r) for r in recs],
                 'latency [ms]', 'latency per execution  =  steps $\\times$ $T_{pipe}$')
    draw_antennas(axR, [throughput_ant_ms(r) for r in recs],
                  'throughput [antennas/ms]  (whole + partial)', 'throughput  =  1 / $T_{pipe}$')

    n_clusters = recs[0].n_clusters if recs else '?'
    csv_disp   = (csv_name or 'CEViT').removesuffix('.csv')
    fig.suptitle(f'{csv_disp}  ({n_clusters}-cluster pipeline)', fontsize=13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'\nSaved plot -> {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary', nargs='?', default='pipeline_summary.txt',
                    help='Split summary from cevit_pipeline.py --summary '
                         '(default: ./pipeline_summary.txt).')
    ap.add_argument('--out', metavar='PNG', default='latency_throughput.png',
                    help='Output PNG (default: ./latency_throughput.png).')
    ap.add_argument('--ctt', action='store_true',
                    help='Plot only the ctt (compute-then-transfer) mode.')
    ap.add_argument('--cwt', action='store_true',
                    help='Plot only the cwt (compute-while-transfer) mode.')
    args = ap.parse_args()

    wanted = {m for m, on in (('ctt', args.ctt), ('cwt', args.cwt)) if on} or {'ctt', 'cwt'}

    records, csv_name = parse_summary(args.summary)
    records = [r for r in records if r.mode in wanted]
    if not records:
        raise SystemExit(f'No pipeline rows for mode(s) {sorted(wanted)} in {args.summary}')

    print_metrics(records)
    plot_bars(records, csv_name, args.out)


if __name__ == '__main__':
    main()
