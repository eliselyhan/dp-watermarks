"""Replay capped rejection selection and plot mean candidate evaluations by task/epsilon."""
import argparse
from collections import defaultdict
from reproduction.io import write_csv
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np
from core.pools import ROOT, MODELS, LABELS, records, load_c4_pools
from core.hashing import prepare, hashes, colors
from core.sampling import select, figure_rng
from reproduction.io import digest
from reproduction.plotting import style_figure, plt

EPSILONS = [.5, 1, 2, 3]
TASKS = {'C4': 'c4', 'OUTFOX': 'outfox', 'LiveCodeBench': 'livecodebench'}


def rejection_cost(blocks, epsilon, rng):
    """Same random draws and short-circuit rule as core.sampling.select, with diagnostics."""
    # Randomized selection: replay rejection acceptance within each block
    # and count candidates examined; no mixed-response samples are drawn.
    costs, fallback, selected = [], [], []
    for block in blocks:
        accepted = False
        for index, favored in enumerate(block):
            if favored or rng.random() < math.exp(-epsilon):
                accepted = True
                break
        costs.append(index+1)
        fallback.append(not accepted)
        selected.append(block[index])
    return np.asarray(costs), np.asarray(fallback), np.asarray(selected)


def pools(dataset, model):
    # Pool construction: load candidate blocks for C4 or split each instruction
    # prompt's candidate responses into blocks after excluding its base pool.
    if dataset == 'c4':
        candidates = load_c4_pools(model).candidates
        yield '', [block[:20] for block in candidates]
    else:
        base_count = 248 if dataset == 'outfox' else 160
        for row in records(f'raw/{dataset}/{model}.jsonl.gz'):
            tokens = [tuple(c['token_ids']) for c in row['completions']]
            assert len(tokens) == base_count+1800
            candidates = tokens[base_count:]
            yield row['prompt_id'], [candidates[i:i+20] for i in range(0,1800,20)]


def render(summary, path):
    fig, axes = plt.subplots(1,3,figsize=(11.5,3.6),sharey=True)
    for ax, task in zip(axes,TASKS):
        for i, (model,label) in enumerate(zip(MODELS,LABELS)):
            rows = [r for r in summary if r['task'] == task and r['model'] == model]
            if rows:
                ax.plot([r['epsilon'] for r in rows], [r['mean_candidates'] for r in rows],
                        marker=['o','s','^','D'][i], color=f'C{i}', linewidth=2, label=label)
        ax.set_title(task, fontsize=18)
        ax.set_xlabel(r'$\epsilon$', fontsize=20)
        ax.set_xticks(EPSILONS, [f'{e:g}' for e in EPSILONS])
        ax.set_ylim(0, 1.08*max(r['mean_candidates'] for r in summary))
    axes[0].set_ylabel('Mean candidates\nexamined', fontsize=18)
    axes[0].legend(frameon=False,fontsize=10)
    fig.suptitle(r'$M=20,\quad \pi_0=0.5$',fontsize=16)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=.2)
    style_figure(fig)
    fig.tight_layout(pad=.6)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/rejection-cost')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((ROOT/'settings.json').read_text())
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    inputs = {p:h for p,h in manifest.items() if
              (p.startswith('raw/c4/') and '/temp_1/' in p and p.endswith(('base.jsonl.gz','candidate_blocks.jsonl.gz')))
              or p.startswith(('raw/outfox/','raw/livecodebench/')) or p == 'settings.json'}
    for relative, expected in inputs.items():
        assert digest(ROOT/relative) == expected, relative
    inputs['settings.json'] = digest(ROOT/'settings.json')
    totals = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    details = []
    for task, dataset in TASKS.items():
        models = MODELS[:2] if dataset == 'livecodebench' else MODELS
        for mi, model in enumerate(models):
            for prompt, blocks in pools(dataset, model):
                prepared = prepare([t for b in blocks for t in b])
                for ki, key in enumerate(settings['keys']):
                    cache = hashes(prepared, key)
                    for epsilon in EPSILONS:
                        p0 = .5
                        c = colors([t for b in blocks for t in b], cache, p0).reshape(-1,20)
                        seed = np.random.SeedSequence([settings['figure_seeds']['rejection-cost'], ki, mi, int(prompt or 0)])
                        rng = np.random.default_rng(seed)
                        costs, fallback, chosen = rejection_cost(c, epsilon, rng)
                        # Verify the instrumented replay preserves the selection and RNG stream.
                        check_rng = np.random.default_rng(seed)
                        assert np.array_equal(chosen, select(c, epsilon, check_rng))
                        assert rng.bit_generator.state == check_rng.bit_generator.state
                        assert np.all((costs >= 1) & (costs <= 20))
                        counts = np.array([len(costs), int(costs.sum()), int((costs == 20).sum()), int(fallback.sum())])
                        totals[task, model, epsilon, ki] += counts
                        details.append(dict(task=task, model=model, prompt_id=prompt, key_index=ki,
                            epsilon=epsilon, pi0=p0, M=20, blocks=int(counts[0]), candidate_evaluations=int(counts[1]),
                            blocks_reaching_cap=int(counts[2]), forced_fallbacks=int(counts[3])))
            print(f'Replayed {task}: {model}', flush=True)
    # Cost summary: average candidates examined per selection. This figure
    # measures sampling cost and does not estimate a watermark mixture share.
    summary = []
    for task, model, epsilon in sorted({k[:3] for k in totals}):
        per_key = [totals[task,model,epsilon,ki] for ki in range(len(settings['keys']))]
        total = np.sum(per_key, axis=0)
        expected = (2000 if task == 'C4' else 20*90)*len(settings['keys'])
        assert total[0] == expected
        summary.append(dict(task=task, model=model, epsilon=epsilon, pi0=.5, M=20,
            blocks=int(total[0]), mean_candidates=float(total[1]/total[0]),
            fraction_reaching_cap=float(total[2]/total[0]), forced_fallback_fraction=float(total[3]/total[0]),
            key_mean_p05=float(np.quantile([r[1]/r[0] for r in per_key], .05)),
            key_mean_p95=float(np.quantile([r[1]/r[0] for r in per_key], .95))))
    assert len(summary) == 10*len(EPSILONS)
    path = args.output/'rejection_samples_vs_epsilon.png'
    render(summary, path)
    write_csv(path.with_suffix('.csv'),summary)
    write_csv(args.output/'rejection_cost_by_prompt_key.csv',details)
    command = shlex.join([sys.executable,*sys.argv])
    path.with_name(path.stem+'_COMMAND.txt').write_text(command+'\n')
    metadata = dict(figure=path.name, tasks=list(TASKS), epsilon=EPSILONS, M=20,
        pi0_rule='fixed 0.5', keys=settings['keys'],
        models_by_task={task:MODELS[:2] if task == 'LiveCodeBench' else MODELS for task in TASKS},
        metric='Mean candidate evaluations per returned response, including the accepted or final fallback candidate.',
        interpretation='Sequential rejection cost replayed from pre-generated raw candidates; not wall-clock runtime or total offline generation cost.',
        averaging='Equal weight per candidate block and hash key; 2000 blocks/key for C4 and 20 prompts x 90 blocks/key for OUTFOX/LiveCodeBench.',
        cap_behavior='Return candidate 20 if none accepted. Reaching the cap includes acceptance on candidate 20; forced fallback counts only no acceptance.',
        seed=settings['figure_seeds']['rejection-cost'], selection_rng='SeedSequence([figure_seed, key_index, model_index, prompt_id]); reset for each epsilon.',
        calibration='Fixed pi0=0.5, matching the main and fixed-pi0 ablation figures.',
        analyzed_tokens={'C4':'first 50 tokens','OUTFOX':'complete sequence','LiveCodeBench':'first 100 post-thinking tokens'},
        lambda_dependence='None: cost measured before mixing with base responses.',
        plotted_points=len(summary), bands=None, y_axis_minimum=0,
        validation=dict(matches_engine_select=True, matches_rng_state=True, counts_within_cap=True, block_counts_checked=True),
        command=command, input_sha256=inputs,
        output_sha256={p.name:digest(p) for p in [path,path.with_suffix('.csv'),args.output/'rejection_cost_by_prompt_key.csv']})
    path.with_name(path.stem+'_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(path,flush=True)


if __name__ == '__main__':
    main()
