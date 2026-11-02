"""Compute C4 sweeps and pure-RR baselines directly as combined figures."""
import argparse
import csv
from reproduction.io import digest
from reproduction.experiments import c4, randomized_response
from contextlib import ExitStack
import gzip
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from reproduction.plotting import style_figure, plt, curves


ROOT = Path(__file__).resolve().parents[1]


def generate(output, settings, sweeps=None):
    """Build pools, sample and estimate C4 mixtures, then render the selected baselines."""
    sweeps = ['epsilon', 'n'] if sweeps is None else list(dict.fromkeys(sweeps))
    output = Path(output)
    target = output / 'c4-ablation'
    target.mkdir(parents=True, exist_ok=True)
    (output / 'replicates').mkdir(parents=True, exist_ok=True)
    errors = defaultdict(list)
    replicate_paths = {}
    with ExitStack() as stack:
        writers = {}

        def emit(name, row):
            if name not in writers:
                path = output / 'replicates' / f'{name}_pure_rr_overlay.csv.gz'
                replicate_paths[name] = path
                stream = stack.enter_context(gzip.open(path, 'wt', newline=''))
                writers[name] = csv.DictWriter(stream, fieldnames=list(row))
                writers[name].writeheader()
            writers[name].writerow(row)
            lam = float(row['true_lambda'])
            identity = (name, row['model'], float(row['panel']), float(row['x']))
            errors[identity].append(100*abs(np.clip(float(row['lambda_hat']), 0, 1)-lam)/lam)

        # Pool construction, randomized mixture draws, and frequency estimation
        # run in c4.run. No standalone epsilon/n figures or tables are created.
        results = c4.run(sweeps, settings, emit)

    rr_curves = {}
    grids = {}
    for name in sweeps:
        rows = results[name]
        grid = sorted({row['x'] for row in rows})
        grids[name] = grid
        # Simulate the randomized-response reports and invert their reporting
        # channel in memory; the baseline is part of this overlay only.
        rr = randomized_response.curve(name, grid, settings['figure_seeds']['pure-rr'])
        rr_curves[name] = rr
        band_rows = []
        for row in rows:
            values = errors[name, row['model'], row['panel'], row['x']]
            row['mape'] = float(np.mean(values))
            row['lower'], row['upper'] = map(float, np.quantile(values, [.05, .95]))
            band_rows.append(dict(curve=row['model'], panel=row['panel'], x=row['x'],
                                  mape=row['mape'], lower=row['lower'], upper=row['upper']))
        for row in rr:
            row['lower'], row['upper'] = row['mape_p05'], row['mape_p95']
            band_rows.append(dict(curve='pure_rr', panel=row['prevalence'], x=row[name],
                                  mape=row['mape'], lower=row['lower'], upper=row['upper']))
        fig, axes = plt.subplots(1, 3, figsize=(12.99, 4.4), sharey=True)
        for ax, panel in zip(axes, [0.1, 0.5, 0.9]):
            group = [r for r in rows if r['panel'] == panel]
            curves(ax, group, metric='mape', bands=False)
            reference = sorted([r for r in rr if r['prevalence'] == panel],
                               key=lambda r: r[name])
            ticks = sorted({r['x'] for r in group})
            assert ticks == [r[name] for r in reference]
            ax.plot(ticks, [r['mape'] for r in reference], color='black',
                    linestyle='--', marker='x', markersize=6, linewidth=2,
                    label='Pure RR', zorder=5)
            ax.set_title(rf'$\lambda={panel:g}$', fontsize=18)
            ax.set_xlabel(r'$\epsilon$' if name == 'epsilon' else r'$n$', fontsize=20)
            if name == 'n':
                ax.set_xscale('log')
            ax.set_xticks(ticks, [f'{x:g}' for x in ticks], rotation=30 if name == 'n' else 0)
            ax.set_box_aspect(1)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(alpha=0.2)
        axes[0].set_ylim(bottom=0)
        axes[0].set_ylabel('MAPE (%)', fontsize=20)
        axes[1].legend(frameon=False, fontsize=11, loc='upper right')
        style_figure(fig)
        fig.tight_layout(w_pad=0.8)
        for extension in ['png']:
            path = output / f'c4-ablation/{name}_pure_rr_overlay.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(path)
        plt.close(fig)
        with (output / f'c4-ablation/{name}_pure_rr_overlay_bands.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(band_rows[0]))
            writer.writeheader()
            writer.writerows(band_rows)
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    inputs = {path: checksum for path, checksum in manifest.items()
              if path.startswith('raw/c4/') and '/temp_1/' in path}
    inputs['settings.json'] = digest(ROOT / 'settings.json')
    metadata = dict(
        command=shlex.join([sys.executable, '-m', 'reproduction.overlays',
                           '--output-dir', str(output), '--sweeps', *sweeps]),
        input_sha256=inputs,
        c4_sampling=dict(keys=settings['keys'], repeats_per_key=settings['repeats'],
                         seeds={name: settings['figure_seeds'][name] for name in results},
                         sampling='Without replacement within each component',
                         estimator='clip((favored_fraction-pi0)/(p1-pi0), 0, 1)'),
        c4_replicates={name: str(path.relative_to(output)) for name, path in replicate_paths.items()},
        rr_settings=dict(seed=settings['figure_seeds']['pure-rr'], repeats=1000, clip=True,
                         fixed_n=2000, fixed_epsilon=2, prevalence=[0.1, 0.5, 0.9],
                         grids=grids, pi0=0.5, pi1='sigmoid(epsilon)',
                         population='Fixed m=round(n*prevalence) ones and n-m zeros'),
        rr_curves=rr_curves,
        bands=None,
        overlay='Simulated pure-RR MAPE; no exponential fit')
    metadata_name = ('pure_rr_overlay_metadata.json' if len(sweeps) == 2
                     else f'{sweeps[0]}_pure_rr_overlay_metadata.json')
    (target / metadata_name).write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--sweeps', nargs='+', choices=['epsilon', 'n'], default=['epsilon', 'n'])
    args = parser.parse_args()
    settings = json.loads((ROOT / 'settings.json').read_text())
    generate(args.output_dir.resolve(), settings, args.sweeps)


if __name__ == '__main__':
    main()
