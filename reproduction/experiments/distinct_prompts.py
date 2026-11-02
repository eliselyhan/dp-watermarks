"""Generate the C4 distinct-prompt ablation with fresh seeded mixtures from raw token IDs."""
import argparse
import csv
import gzip
from reproduction.io import digest
import json
from pathlib import Path
import shlex
import sys
import numpy as np
from reproduction.experiments.distinct_prompts_config import (
    BASE_POOL_SIZE, COMPLETIONS_PER_PROMPT, WATERMARKED_POOL_SIZE,
    CANDIDATES_PER_BLOCK, LAMBDA_TENTHS, MAX_TOKENS, MIXTURE_SIZE,
    NUM_PROMPTS, PROMPT_COUNTS, REPEATS_PER_KEY,
)
from core.hashing import prepare, hashes, colors
from core.sampling import select, figure_rng
from reproduction.plotting import style_figure, plt

ROOT = Path(__file__).resolve().parents[2]
RAW = 'raw/c4-distinct-prompts/completions.jsonl.gz'
NAME = 'distinct-prompts'
LAMBDAS = [t / 10 for t in LAMBDA_TENTHS]
MS = list(PROMPT_COUNTS)

# Fixed experiment settings; watermark keys and seeds come from settings.json.
CONFIG = {
    "temperature": 1.0,
    "epsilon": 2.0,
    "n": MIXTURE_SIZE,
    "p0": 0.5,
    "prompt_counts": MS,
    "rejection_block_size": CANDIDATES_PER_BLOCK,
}


def sample_mixtures(base_counts, marked_counts, prompt_count, tenth, rng, repeats=REPEATS_PER_KEY):
    """Uniformly choose distinct prompts, then sample responses without replacement.

    Every chosen prompt contributes MIXTURE_SIZE / prompt_count responses. Allocate the
    watermarked responses as evenly as possible, randomizing remainder slots.
    Hypergeometric draws sample the favored counts exactly without materializing
    response indices; the estimator sees only their unlabeled aggregate.
    """
    prompt_total = len(base_counts)
    # Randomized samples: choose distinct prompts for each mixture, then
    # randomize which prompts receive the remainder of the watermarked quota.
    prompts = np.argsort(rng.random((repeats, prompt_total)), axis=1)[:, :prompt_count]
    marked_total = MIXTURE_SIZE * tenth // 10
    marked_draws = np.full(prompts.shape, marked_total // prompt_count)
    remainder_slots = np.argsort(rng.random(prompts.shape), axis=1)[:, :marked_total % prompt_count]
    marked_draws[np.arange(repeats)[:, None], remainder_slots] += 1
    base_draws = MIXTURE_SIZE // prompt_count - marked_draws
    # Draw favored counts without replacement within each prompt/component.
    # Return their unlabeled total; distinct_prompts.main applies the estimator.
    base_favored = rng.hypergeometric(base_counts[prompts], BASE_POOL_SIZE-base_counts[prompts], base_draws)
    marked_favored = rng.hypergeometric(marked_counts[prompts], WATERMARKED_POOL_SIZE-marked_counts[prompts], marked_draws)
    return (base_favored + marked_favored).sum(axis=1)


def replay(config, output):
    """Build watermarked pools and draw fresh mixtures using the figure seed."""
    key_count = len(config["keys"])
    base=np.empty((key_count,NUM_PROMPTS),dtype=np.int64); marked=np.empty_like(base)
    seen=[]
    # Pool construction: read each prompt's disjoint base and candidate ranges
    # from bundled token IDs, preserving source order and duplicate responses.
    with gzip.open(ROOT/RAW,'rt') as stream:
        for line in stream:
            row=json.loads(line); pid=row['prompt_id']; seen.append(pid)
            assert row['source_indices']==list(range(COMPLETIONS_PER_PROMPT))
            assert row['base_range']==[0,BASE_POOL_SIZE] and row['candidate_range']==[BASE_POOL_SIZE,COMPLETIONS_PER_PROMPT]
            tokens=[tuple(t) for t in row['token_ids']]
            assert len(tokens)==COMPLETIONS_PER_PROMPT and all(0<len(t)<=MAX_TOKENS for t in tokens)
            prepared=prepare(tokens)
            for ki,key in enumerate(config['keys']):
                bits=colors(tokens,hashes(prepared,key),config["p0"])
                blocks=bits[BASE_POOL_SIZE:].reshape(WATERMARKED_POOL_SIZE,CANDIDATES_PER_BLOCK)
                # Pool construction: select one watermarked response per candidate block
                # and retain the base/marked favored counts for each prompt and key.
                chosen=select(blocks,config["epsilon"],figure_rng(config['seed'], ki, 0, pid))
                base[ki,pid]=bits[:BASE_POOL_SIZE].sum(); marked[ki,pid]=chosen.sum()
            if (pid+1)%10==0: print(f'Replayed hashing and M={CANDIDATES_PER_BLOCK} selection: {pid+1}/{NUM_PROMPTS} prompts',flush=True)
    assert seen==list(range(NUM_PROMPTS))
    # Each plotted condition has its own reproducible stream.
    result=np.empty((len(LAMBDA_TENTHS),len(PROMPT_COUNTS),key_count,REPEATS_PER_KEY),dtype=np.uint16)
    for li,tenth in enumerate(LAMBDA_TENTHS):
        for mi,m in enumerate(PROMPT_COUNTS):
            for ki in range(key_count):
                rng=figure_rng(config['seed'], ki, 1, m, tenth)
                # Randomized samples: generate repeated mixture counts for this
                # (lambda, prompt count, key) using its own seeded stream.
                green=sample_mixtures(base[ki],marked[ki],m,tenth,rng)
                result[li,mi,ki]=green
            print(f'Replayed mixtures: lambda={tenth/10:g}, m={m}',flush=True)
    target=output/'replicates';target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/'distinct-prompts.npz',green_count=result,
                        base_green=base,watermarked_green=marked,lambdas=[t / 10 for t in LAMBDA_TENTHS],prompt_counts=PROMPT_COUNTS)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output')
    args = parser.parse_args()
    output = args.output.resolve()
    target = output/'c4-ablation'
    target.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    settings = json.loads((ROOT/'settings.json').read_text())
    config = {**CONFIG, "keys": settings["keys"]}
    assert config['temperature']==1 and config['epsilon']==2 and config['n']==MIXTURE_SIZE
    assert config['p0']==.5 and config['rejection_block_size']==CANDIDATES_PER_BLOCK
    assert config['prompt_counts']==MS and len(config['keys'])==20
    config['seed'] = settings['figure_seeds'][NAME]
    assert digest(ROOT/RAW) == manifest[RAW]
    # Pool construction and randomized samples: distinct_prompts.replay
    # builds per-prompt pools and returns sampled, unlabeled favored counts.
    counts = replay(config, output)
    raw_inputs = {RAW: manifest[RAW], 'settings.json': digest(ROOT/'settings.json')}
    for code in ['core/hashing.py', 'core/sampling.py',
                 'reproduction/experiments/distinct_prompts.py',
                 'reproduction/experiments/distinct_prompts_config.py']:
        raw_inputs[code] = digest(ROOT/code)
    p0,epsilon,n = config['p0'],config['epsilon'],config['n']
    # Estimator: derive the rejection channel probability p1, invert each
    # mixture's favored fraction, and clip the estimated share to [0, 1].
    p1 = p0/(p0+np.exp(-epsilon)*(1-p0))
    hats = np.clip((counts/n-p0)/(p1-p0),0,1)
    rows = []
    for li,lam in enumerate(LAMBDAS):
        for mi,m in enumerate(MS):
            values = hats[li,mi].ravel()
            rows.append(dict(m=m,n=n,true_lambda=lam,estimates=len(values),
                mape_p05=float(np.quantile(100*np.abs(values-lam)/lam,.05)),
                mape_p95=float(np.quantile(100*np.abs(values-lam)/lam,.95)),
                lambda_hat_mean=float(values.mean()),mape=float(np.mean(100*np.abs(values-lam)/lam))))
    csv_path = target/f'{NAME}.csv'
    with csv_path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,axes = plt.subplots(1,3,figsize=(16.5,4.8))
    for ax,lam in zip(axes,LAMBDAS):
        group = [r for r in rows if r['true_lambda']==lam]
        ys = [r['mape'] for r in group]
        ax.plot(MS,ys,'o-',linewidth=2)
        ax.set_xscale('log')
        ax.set_xticks(MS,labels=[str(m) for m in MS])
        ax.minorticks_off()
        ax.set_ylim(0,max(ys)*1.15)
        ax.set_title(rf'$\lambda={lam:g}$',fontsize=19)
        ax.set_xlabel('Number of distinct prompts $m$',fontsize=17)
        ax.tick_params(axis='both',labelsize=14)
        ax.grid(alpha=.2)
        ax.set_box_aspect(.75)
    axes[0].set_ylabel('MAPE (%)',fontsize=18)
    fig.suptitle(r'C4 / Qwen3-4B; temperature=1; 20 tokens; $n=500$; $\epsilon=2$; $M=20$',fontsize=19)
    style_figure(fig)
    fig.tight_layout(w_pad=1.2)
    png = target/f'{NAME}.png'
    fig.savefig(png,dpi=300,bbox_inches='tight')
    plt.close(fig)
    command = shlex.join([sys.executable,*sys.argv])
    (target/f'{NAME}_COMMAND.txt').write_text(command+'\n')
    metadata = dict(figure=png.name,dataset='C4',parameters=dict(model='Qwen3-4B',
        temperature=config["temperature"],max_tokens=MAX_TOKENS,n=n,
        epsilon=epsilon,pi0=p0,M=CANDIDATES_PER_BLOCK,lambdas=LAMBDAS,
        distinct_prompt_counts=MS),axes=dict(x_scale='log',y_metric='MAPE (%)',shared_y=False),
        sampling=dict(keys=config['keys'],estimates_per_point=len(config["keys"])*REPEATS_PER_KEY,
            source_prompts=NUM_PROMPTS,repeats_per_condition_per_key=REPEATS_PER_KEY,
            seed=config['seed'], mixture='Uniform subsets of distinct prompts; equal responses per prompt; balanced watermarked allocation',component_sampling='Without replacement within each component; hypergeometric favored counts'),
        estimator='clip((unlabeled_mixed_green_count/n-pi0)/(pi0/(pi0+exp(-epsilon)*(1-pi0))-pi0),0,1)',
        reproduction_scope='Raw token IDs: hashing, M=20 rejection selection, fresh mixture sampling, clipped estimates, and MAPE',
        input_sha256=raw_inputs,
        sha256=dict(png=digest(png),summary=digest(csv_path)),reproduction_command=command,
        computation=dict(estimates=int(hats.size),summary_points=len(rows),
            raw_replayed=True,clipped=True),
        rendering=dict(numpy=np.__version__,matplotlib=plt.matplotlib.__version__))
    (target/f'{NAME}_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{png}: generated {len(rows)} summary points from {hats.size:,} clipped estimates.')

if __name__ == '__main__':
    main()
