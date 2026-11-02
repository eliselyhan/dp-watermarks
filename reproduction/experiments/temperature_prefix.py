"""Joint temperature/prefix ablation on the common 500-prompt Llama pools."""
import argparse
from pathlib import Path
import csv
import gzip
import json
import shlex
import sys
from collections import defaultdict
import numpy as np
from core.pools import ROOT, load_c4_pools
from core.hashing import prepare, hashes, colors
from core.sampling import select, mixture_plan, figure_rng
from core.estimation import estimate
from reproduction.io import digest
from reproduction.io import write_csv, read_csv
from reproduction.plotting import plt, finish

NAME = 'temperature-prefix'
TEMPERATURES = [.2, .4, .6, .8, 1.]
LENGTHS = [5, 10, 20, 50]
LAMBDAS = [.1, .5, .9]


def pools(temperature):
    """Use the same candidate order as the main C4 figures, capped at M=20."""
    pools = load_c4_pools('llama31_8b', temperature)
    return pools.base, [block[:20] for block in pools.candidates]


def render(summaries, output):
    fig, axes = plt.subplots(1,3,figsize=(13,4.4),sharey=True)
    for ax, lam in zip(axes,LAMBDAS):
        for i, (T,marker) in enumerate(zip(LENGTHS,['o','s','^','D'])):
            rows = [r for r in summaries if r['T']==T and r['true_lambda']==lam]
            ax.plot(TEMPERATURES,[r['mape'] for r in rows],color=f'C{i}',marker=marker,label=f'T = {T}',linewidth=2)
        ax.set(xlabel=r'Temperature $\tau$',ylim=(0,None))
        ax.set_xticks(TEMPERATURES, [f'{temp:g}' for temp in TEMPERATURES])
        ax.set_box_aspect(1)
    axes[0].set_ylabel('MAPE (%)')
    axes[0].legend(frameon=False)
    finish(fig, axes, output/(NAME+'.png'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output'/NAME)
    parser.add_argument('--replot', action='store_true', help='Render saved summaries without replaying raw draws')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.replot:
        render(read_csv(output/'summary.csv'), output)
        metadata = json.loads((output/'metadata.json').read_text())
        metadata['output_sha256'][NAME+'.png'] = digest(output/(NAME+'.png'))
        metadata['replot_command'] = shlex.join([sys.executable, *sys.argv])
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        return
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    (output/'COMMAND.txt').write_text(command+'\n')
    settings = json.loads((ROOT/'settings.json').read_text())
    assert len(settings['keys']) == 20 and settings['repeats'] == 50
    inputs = {'settings.json': digest(ROOT/'settings.json')}
    summaries, diagnostics = [], []
    bootstrap = figure_rng(settings['figure_seeds'][NAME], stream=2).integers(0, 20, (5000, 20))
    fields = ['temperature','T','key_index','true_lambda','repeat','favored_count','lambda_hat','absolute_percentage_error']
    with gzip.open(output/'replicates.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for temp in TEMPERATURES:
            tag = 'temp_'+f'{temp:g}'.replace('.', 'p')
            for kind in ['base', 'candidate_blocks']:
                rel = f'raw/c4/llama31_8b/{tag}/{kind}.jsonl.gz'
                inputs[rel] = digest(ROOT/rel)
            # Pool construction: load disjoint base responses and candidate blocks
            # for this temperature; truncate token sequences before keyed hashing.
            raw_base, raw_blocks = pools(temp)
            # Truncate before hashing so each prefix length defines its own colors.
            for T in LENGTHS:
                base = [t[:T] for t in raw_base]
                candidates = [t[:T] for block in raw_blocks for t in block]
                prepared = prepare(base+candidates)
                errors, hats = defaultdict(list), defaultdict(list)
                for ki, key in enumerate(settings['keys']):
                    cache = hashes(prepared,key)
                    b = colors(base,cache,.5)
                    c = colors(candidates,cache,.5).reshape(2000,20)
                    # Pool construction: select one watermarked response per block using
                    # colors computed for the current token-prefix length.
                    w = select(c,2,figure_rng(settings['figure_seeds'][NAME], ki))
                    diagnostics.append(dict(temperature=temp,T=T,key_index=ki,base_fraction=float(b.mean()),marked_fraction=float(w.mean())))
                    per_key = defaultdict(list)
                    # Randomized samples: draw base and watermarked component indices
                    # without replacement, pairing draws across temperature/prefix conditions.
                    for n, lam, repeat, wm, bg in mixture_plan(32000,2000,[2000],LAMBDAS,ki,settings['figure_seeds'][NAME],settings['repeats']):
                        count = int(w[wm].sum()+b[bg].sum())
                        # Estimator: invert the combined favored fraction using pi0=0.5 and
                        # epsilon=2; core.estimation.estimate clips the watermark share to [0, 1].
                        hat = float(estimate(count/n,.5,2))
                        assert 0 <= hat <= 1
                        error = 100*abs(hat-lam)/lam
                        per_key[lam].append(error)
                        hats[lam].append(hat)
                        writer.writerow(dict(zip(fields,[temp,T,ki,lam,repeat,count,hat,error])))
                    for lam in LAMBDAS:
                        assert len(per_key[lam]) == 50
                        errors[lam].append(np.mean(per_key[lam]))
                for lam in LAMBDAS:
                    # Bootstrap key-level means, keeping resamples paired across conditions.
                    means = np.asarray(errors[lam])
                    lo, hi = np.quantile(means[bootstrap].mean(axis=1),[.025,.975])
                    assert len(hats[lam]) == 1000
                    summaries.append(dict(temperature=temp,T=T,true_lambda=lam,count=1000,lambda_hat_mean=float(np.mean(hats[lam])),mape=float(means.mean()),ci_low=float(lo),ci_high=float(hi)))
                stream.flush()
                print(f'Completed temperature={temp:g}, T={T}',flush=True)
    assert len(summaries) == 60
    write_csv(output/'summary.csv',summaries)
    write_csv(output/'diagnostics.csv',diagnostics)
    render(summaries, output)
    metadata = dict(command=command,temperatures=TEMPERATURES,prefix_lengths=LENGTHS,lambdas=LAMBDAS,n=2000,epsilon=2,M=20,pi0=.5,prompts=500,base_responses=32000,selected_responses=2000,keys=settings['keys'],repeats_per_key=50,replicate_rows=60000,source_draws_disjoint=True,duplicates_retained=True,truncation='first min(T, length) tokens before hashing and selection',estimator='nominal, clipped to [0,1], unlabeled mixture only',bands='95% bootstrap over 20 key means; 5000 paired resamples; figure seed, stream 2; conditional on fixed prompt pools',pairing='Same mixture indices and initial selection seeds across all conditions; selection RNG consumption may vary',seed=settings['figure_seeds'][NAME], candidate_construction='First 20 candidates from each ready-to-use candidate block',input_sha256=inputs)
    metadata['output_sha256'] = {p.name:digest(p) for p in output.iterdir() if p.suffix in ['.csv','.gz','.png']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output,flush=True)


if __name__ == '__main__':
    main()
