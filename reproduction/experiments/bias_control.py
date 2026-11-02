"""Generate the pi0 bias-control figure with direct seeded mixture sampling."""
import argparse
from collections import defaultdict
import csv
from reproduction.io import write_csv
import gzip
import json
from pathlib import Path
import shlex
import sys

import numpy as np
from core.pools import ROOT
from core.hashing import prepare, hashes, colors
from core.sampling import select, mixture_plan, figure_rng
from core.estimation import estimate
from reproduction.io import digest
from reproduction.plotting import style_figure, plt
from matplotlib.ticker import ScalarFormatter, NullLocator

NAME = 'pi0-bias-control'
RAW = 'raw/c4-pi0-bias-control/llama31_8b_temp_0p2_t20.jsonl.gz'
P0S = [.1, .2, .3, .4, .5]
LAMBDAS = [.2, .3, .4]
PLOTTED = [50, 100, 250, 500, 800, 1000, 1500, 2000, 3000, 5000]
EXTENDED = {
    .2: [50, 100, 250, 500, 1000, 2000, 3500, 6000, 10000, 15000],
    .3: [50, 100, 250, 500, 1000, 2000, 3500, 5000, 7500, 10000],
    .4: [50, 100, 250, 500, 1000, 2000, 3500, 4500, 6000, 7500],
}


def draw_plan(lambdas, key_index, seed, extended):
    """Draw only the sizes plotted in each panel, paired across pi0 values."""
    for lam in lambdas:
        sizes = EXTENDED[lam] if extended else PLOTTED
        sizes = [n for n in sizes if round(n * lam) <= 3000]
        yield from mixture_plan(64000, 3000, sizes, [lam], key_index, seed)


def render(rows, path, ymax, temperature=None, sampler="rejection", lambdas=LAMBDAS, epsilon=2, extended=False, p0s=P0S):
    errors = defaultdict(list)
    with gzip.open(path.parent.parent/'replicates'/f'{path.stem}.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            lam = float(row['true_lambda'])
            errors[float(row['target_p0']), lam, int(row['sample_size'])].append(
                100*abs(np.clip(float(row['lambda_hat']), 0, 1)-lam)/lam)
    for row in rows:
        row['mape_p05'], row['mape_p95'] = np.quantile(errors[
            float(row['target_p0']), float(row['true_lambda']), int(row['sample_size'])], [.05, .95])
    ymax = 1.05*max(float(row['mape']) for row in rows)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, lam in zip(axes, lambdas):
        plotted = EXTENDED[lam] if extended else PLOTTED
        for i, p0 in enumerate(p0s):
            style = {0.1: 0, 0.2: 1, 0.27: 0, 0.3: 2, 0.4: 3, 0.5: 4}.get(p0, i)
            marker = ['o', 's', '^', 'D', 'v'][style]
            group = sorted((r for r in rows if float(r['target_p0']) == p0 and
                            float(r['true_lambda']) == lam and int(r['sample_size']) in plotted),
                           key=lambda r: int(r['sample_size']))
            expected = [n for n in plotted if round(n*lam) <= 3000]
            assert [int(r['sample_size']) for r in group] == expected
            ax.plot(expected, [float(r['mape']) for r in group], color=f'C{style}', marker=marker,
                    linewidth=2, label=rf'$\pi_0={p0:g}$')
        ax.set(xscale='log', ylim=(0, ymax))
        ax.set_xlabel(r'Mixture sample size $n$', fontsize=18)
        ax.set_xticks([50, 100, 500, 2000, plotted[-1]] if extended else [50, 100, 500, 1000, 5000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='both', labelsize=14)
        ax.set_box_aspect(.75)
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1.), frameon=False, fontsize=15)
    axes[0].set_ylabel('Mean absolute\npercentage error', fontsize=18)
    title = ('Parallel' if sampler == 'parallel' else 'Rejection') + rf' RGS ($\epsilon={epsilon:g}$)'
    if temperature is not None:
        title += f', temperature={temperature:g}'
    axes[1].set_title(title, fontsize=19, pad=14)
    if extended or lambdas != LAMBDAS:
        for ax, lam in zip(axes, lambdas):
            ax.text(.38, .95, rf'$\lambda={lam:g}$', transform=ax.transAxes,
                    va='top', fontsize=16)
    style_figure(fig)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def recompute(output, settings, raw=RAW, name=NAME, sampler="rejection", lambdas=LAMBDAS, epsilon=2, extended=False, p0s=P0S):
    parallel = sampler == "parallel"
    # Pool construction: reserve 64 base responses per prompt and collect
    # the remaining 60 candidates for blocks of 20.
    base, candidates, seen = [], [], []
    with gzip.open(ROOT/raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            seen.append(row['prompt_id'])
            assert row['source_indices'] == list(range(124))
            tokens = [tuple(c['token_ids']) for c in row['completions']]
            assert len(tokens) == 124 and all(len(t) <= 20 for t in tokens)
            base.extend(tokens[:64])
            candidates.extend(tokens[64:])
    assert seen == list(range(1000)) and len(base) == 64000 and len(candidates) == 60000
    prepared = prepare(base + candidates)
    groups, diagnostics = defaultdict(list), []
    path = output/'replicates'/f'{name}.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['sampler', 'key_index', 'key', 'target_p0', 'sample_size', 'true_lambda',
              'repeat', 'favored_count', 'favored_fraction', 'lambda_hat', 'absolute_percentage_error']
    with gzip.open(path, 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ki, key in enumerate(settings['keys']):
            cache = hashes(prepared, key)
            for p0 in p0s:
                base_bits = colors(base, cache, p0)
                blocks = colors(candidates, cache, p0).reshape(3000, 20)
                # Pool construction: select one watermarked response per block under
                # the current key, pi0, epsilon, and sampler.
                marked = select(blocks, epsilon, figure_rng(settings['figure_seeds'][NAME], ki), parallel=parallel)
                diagnostics.append(dict(sampler=sampler, key_index=ki, key=key, target_p0=p0,
                    base_fraction=float(base_bits.mean()), marked_fraction=float(marked.mean())))
                # Randomized samples: draw_plan delegates to core.sampling.mixture_plan for
                # seeded component sampling without replacement.
                for n, lam, repeat, wm, bg in draw_plan(lambdas, ki, settings['figure_seeds'][NAME], extended):
                    count = int(base_bits[bg].sum()) + int(marked[wm].sum())
                    # Estimator: invert the observed mixed favored fraction using the
                    # nominal pi0 and sampler-specific p1, then clip to [0, 1].
                    hat = float(estimate(count/n, p0, epsilon, parallel=parallel))
                    error = 100*abs(hat-lam)/lam
                    groups[p0, n, lam].append(hat)
                    writer.writerow(dict(zip(fields, [sampler, ki, key, p0, n, lam,
                        repeat, count, count/n, hat, error])))
            print(f'{name}: key {ki+1}/{len(settings["keys"])}', flush=True)
    rows = []
    for (p0, n, lam), values in sorted(groups.items()):
        assert len(values) == 1000
        values = np.asarray(values)
        rows.append(dict(sampler=sampler, target_p0=p0, sample_size=n, true_lambda=lam,
            repeats=len(values), lambda_hat_mean=float(values.mean()),
            mape=float(np.mean(100*np.abs(values-lam)/lam))))
    expected_points = sum(round(n*lam) <= 3000 for lam in lambdas
                          for n in (EXTENDED[lam] if extended else PLOTTED))
    assert len(rows) == len(p0s)*expected_points
    write_csv(output/'c4-ablation'/f'{name}.csv', rows)
    write_csv(output/'c4-ablation'/f'{name}_diagnostics.csv', diagnostics)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    parser.add_argument('--pi0', type=float, nargs=5, default=P0S)
    parser.add_argument('--extend', action='store_true', help='Plot ten sizes per panel up to available pool limits')
    parser.add_argument('--epsilon', type=float, default=2.)
    parser.add_argument('--temperature', type=float, choices=[.2, 1.], default=.2)
    parser.add_argument('--sampler', choices=['rejection', 'parallel'], default='rejection')
    parser.add_argument('--lambdas', type=float, nargs=3, default=LAMBDAS)
    parser.add_argument('--replot', action='store_true', help='Render existing output summary without recomputing raw draws')
    args = parser.parse_args()
    if len(set(args.pi0)) != 5 or any(not 0 < p0 < 1 for p0 in args.pi0):
        parser.error('--pi0 requires five distinct probabilities in (0, 1)')
    if any(not 0 < lam <= 1 for lam in args.lambdas) or len(set(args.lambdas)) != 3:
        parser.error('--lambdas requires three distinct values in (0, 1]')
    if not np.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error('--epsilon must be finite and positive')
    name = NAME if args.temperature == .2 else NAME+'-temp1'
    if args.epsilon != 2:
        name += '-eps' + f'{args.epsilon:g}'.replace('.', 'p')
    if args.sampler == 'parallel':
        name += '-parallel'
    if args.lambdas != LAMBDAS:
        name += '-lambda' + '_'.join(f'{lam:g}'.replace('.', 'p') for lam in args.lambdas)
    if args.extend:
        if args.lambdas != LAMBDAS:
            parser.error('--extend requires the original lambdas 0.2, 0.3, 0.4')
        name += '-extended10'
    if args.pi0 != P0S:
        name += '-pi0-' + '_'.join(f'{p0:g}'.replace('.', 'p') for p0 in args.pi0)
    raw = RAW if args.temperature == .2 else 'raw/c4-pi0-bias-control/llama31_8b_temp_1_t20.jsonl.gz'
    output = args.output.resolve()
    (output/'c4-ablation').mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT/'settings.json').read_text())
    assert len(settings['keys']) == 20 and settings['repeats'] == 50
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    assert digest(ROOT/raw) == manifest[raw]
    inputs = {raw: manifest[raw], 'settings.json': digest(ROOT/'settings.json')}
    path = output/'c4-ablation'/f'{name}.png'
    if args.replot:
        with path.with_suffix('.csv').open() as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = recompute(output, settings, raw, name, sampler=args.sampler, lambdas=args.lambdas, epsilon=args.epsilon, extended=args.extend, p0s=args.pi0)
    ymax = 1.05*max(float(r['mape']) for r in rows)
    render(rows, path, ymax, temperature=1 if args.temperature == 1 else None, sampler=args.sampler, lambdas=args.lambdas, epsilon=args.epsilon, extended=args.extend, p0s=args.pi0)
    command = shlex.join([sys.executable, *sys.argv])
    path.with_name(name+'_COMMAND.txt').write_text(command+'\n')
    files = dict(summary_csv=name+'.csv', replicates='../replicates/'+name+'.csv.gz',
                 diagnostics=name+'_diagnostics.csv')
    checksums = {key: digest(path.parent/relative) for key, relative in files.items()}
    checksums['png'] = digest(path)
    plotted_by_lambda = {str(lam): EXTENDED[lam] if args.extend else PLOTTED for lam in args.lambdas}
    plotted_sizes = sorted({n for sizes in plotted_by_lambda.values() for n in sizes})
    metadata = dict(figure=path.name, dataset='C4', parameters=dict(model='Llama-3.1-8B-Instruct',
        temperature=args.temperature, analyzed_token_prefix=20, epsilon=[args.epsilon], pi0=args.pi0, M=[20],
        n=plotted_sizes, **{'lambda': args.lambdas}),
        axes=dict(x_parameter='n', x_values=plotted_sizes, x_values_by_lambda=plotted_by_lambda, panel_parameter='lambda', panel_values=args.lambdas,
                  y_metric='MAPE (%)', bands=None, shared_y_limits=[0, ymax]),
        method=dict(selection=('likelihood-ratio optimal redistribution within each candidate block'
                               if args.sampler == 'parallel' else 'bounded rejection; final-candidate fallback'),
            calibration='nominal; no empirical component rates',
            p1=('min(exp(epsilon)*pi0, 1-exp(-epsilon)*(1-pi0))' if args.sampler == 'parallel'
                else 'pi0/(pi0+exp(-epsilon)*(1-pi0))'),
            estimator='clip((favored_fraction-pi0)/(p1-pi0), 0, 1)',
            mape='mean(100*abs(clipped_lambda_hat-lambda)/lambda)',
            prompt_count=1000, base_pool_responses=64000, selected_pool_responses=3000,
            source_draws_disjoint=True, duplicates_retained=True,
            base_draw_indices=[0,64], candidate_draw_indices=[64,124],
            candidate_construction='Three same-prompt blocks of 20 from the 60 candidate responses, disjoint from 64 base responses.',
            mixture_sampling='without replacement within each component'),
        sampling=dict(keys=settings['keys'], repeats_per_key=50, replicates_per_point=1000,
            seed=settings['figure_seeds'][NAME],
            selection_seed='SeedSequence([figure_seed, key_index, 0])',
            mixture_seed='SeedSequence([figure_seed, key_index, 1, n, round(lambda*1000)])',
            size_plan='Sample only plotted sizes, independently per (n, lambda).'),
        plotted_points=sum(int(r['sample_size']) in plotted_by_lambda[str(float(r['true_lambda']))] for r in rows),
        summary_points=len(rows), replicate_rows=sum(int(r['repeats']) for r in rows),
        reproduction_command=command, files=files, sha256=checksums, input_sha256=inputs,
        rendering=dict(numpy=np.__version__, matplotlib=plt.matplotlib.__version__))
    path.with_name(name+'_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'{path}: generated', flush=True)


if __name__ == '__main__':
    main()
