#!/usr/bin/env python3
"""
Fixed latency and throughput of each pipeline configuration, as two bar charts.

Reads the pipeline_summary.txt that cevit_pipeline.py or celere_pipeline.py writes.  The
model and the partition are both fixed, so every (pool, mode) configuration has exactly one
latency and one throughput -- there is nothing to sweep:

    latency    = steps * T_pipe  [ms]        one execution end to end in steady state
                                             (the summary's latency_ms column)
    throughput = 1 / T_pipe  [antennas/ms]   completions per ms at the bottleneck

LATENCY IS NOT THE CRITICAL-PATH SUM sum_s(compute_s + transfer_s).  The generated pipelines
put a global barrier on every step, so a stage that finishes early waits for the slowest one
and each step costs the bottleneck period T_pipe.  An item is resident for one step per stage
in ctt (latency = n_stages * T_pipe) and two in cwt, whose double-buffered schedule offsets
consecutive stages by two steps (latency = 2 * n_stages * T_pipe).  So cwt shortens the period
and lengthens the flight, and the two modes separate in BOTH panels.

Left panel = latency per config; right panel = throughput per config.

Usage
-----
    python latency_throughput_bars.py CEViT/pipeline_summary.txt
    python latency_throughput_bars.py CELERE/pipeline_summary.txt --out lt.png
    python latency_throughput_bars.py CEViT/pipeline_summary.txt --cwt      # one mode only
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
    """Parse a pipeline_summary.txt into (records, csv_name)."""
    records: List[PipelineRecord] = []
    csv_name = ''
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            if line[1:].strip().lower().startswith('csv:'):
                csv_name = line.split(':', 1)[1].strip()
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f'Malformed summary row (need 6 columns): {raw!r}')
        pool, mode, t_pipe, n_stages, n_clusters, latency = fields[:6]
        records.append(PipelineRecord(pool, mode, float(t_pipe), int(n_stages),
                                      int(n_clusters), float(latency)))
    if not records:
        raise ValueError(f'No pipeline rows found in {path}')
    return records, csv_name


def throughput_ant_ms(rec: PipelineRecord) -> float:
    return 1.0 / rec.t_pipe if rec.t_pipe > 0 else 0.0


def print_metrics(records: List[PipelineRecord]) -> None:
    print(f'\n{"pool":<11}{"mode":<5}{"latency[ms]":>13}{"throughput[ant/ms]":>21}')
    for r in records:
        print(f'{r.pool:<11}{r.mode.upper():<5}'
              f'{r.latency:>13.4f}{throughput_ant_ms(r):>21.4f}')


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
    labels = [f'{r.pool}\n{r.mode.upper()}' for r in recs]
    xpos   = list(range(len(recs)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    def draw_decimal(ax, values, ylabel, title):
        """Plain bars with a 4-decimal label, for latency, which is a time."""
        bars = ax.bar(xpos, values, color=colour, edgecolor='white')
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        for x, v in zip(xpos, values):
            ax.text(x, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(values) * 1.15 if values else 1)      # headroom for labels
        _finish(ax, ylabel, title)

    def draw_antennas(ax, values, ylabel, title):
        """Antennas are countable: floor(v) whole antennas solid plus the fractional one
        as a lighter cap, headlining the integer and keeping the exact value below it."""
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
        ax.set_ylim(0, max(values) * 1.24 if values else 1)      # room for both labels
        _finish(ax, ylabel, title)

    def _finish(ax, ylabel, title):
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    draw_decimal(axL, [r.latency for r in recs],
                 'latency [ms]', 'latency per execution  =  steps $\\times$ $T_{pipe}$')
    draw_antennas(axR, [throughput_ant_ms(r) for r in recs],
                  'throughput [antennas/ms]  (whole + partial)',
                  'throughput  =  1 / $T_{pipe}$')

    n_clusters = recs[0].n_clusters if recs else '?'
    csv_disp   = (csv_name or 'pipeline').removesuffix('.csv')
    fig.suptitle(f'{csv_disp}  ({n_clusters}-cluster pipeline)', fontsize=13)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'\nSaved plot -> {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('summary', nargs='?', default='pipeline_summary.txt',
                    help='Summary written by cevit_pipeline.py / celere_pipeline.py '
                         '--summary (default: ./pipeline_summary.txt).')
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
