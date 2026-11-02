"""OUTFOX and LiveCodeBench: per-prompt pools, sampling, and estimation."""
from collections import defaultdict
import numpy as np
from core.hashing import prepare, hashes, colors
from core.sampling import select, mixture_plan, figure_rng
from core.estimation import estimate, statistics
from core.pools import MODELS, records


def run(dataset, settings, emit):
    """Rebuild rejection selections before drawing per-prompt mixtures."""
    models = MODELS if dataset == 'outfox' else MODELS[:2]
    base_count = 248 if dataset == 'outfox' else 160
    p0 = .5
    groups = defaultdict(list)
    for mi, model in enumerate(models):
        for row in records(f'raw/{dataset}/{model}.jsonl.gz'):
            tokens = [tuple(c['token_ids']) for c in row['completions']]
            if dataset == 'livecodebench':
                assert row['source_indices'] == list(range(1960))
            assert len(tokens) == base_count+1800
            prepared = prepare(tokens)
            for ki, key in enumerate(settings['keys']):
                cache = hashes(prepared, key)
                all_colors = colors(tokens, cache, p0)
                selection_seed, mixture_seed = np.random.SeedSequence(
                    [settings['figure_seeds']['instruction-main'], mi, row['prompt_id'], ki, int(dataset == 'livecodebench')]).spawn(2)
                # Pool construction: split each prompt into its base responses and
                # 90 blocks of 20 candidates, then select one response per block.
                selected = select(all_colors[base_count:].reshape(90, 20), 2,
                                  np.random.default_rng(selection_seed))
                bg, wg = int(all_colors[:base_count].sum()), int(selected.sum())
                for tenth in range(1, 10):
                    lam = tenth/10
                    rng = np.random.default_rng(np.random.SeedSequence(mixture_seed.entropy,
                        spawn_key=mixture_seed.spawn_key+(100, tenth)))
                    # Randomized samples: hypergeometric counts are equivalent to drawing
                    # responses without replacement from each pool and summing favored bits.
                    counts = rng.hypergeometric(bg, base_count-bg, 100-10*tenth, 50)
                    counts += rng.hypergeometric(wg, 90-wg, 10*tenth, 50)
                    # Estimator: invert the combined favored fraction for each 100-response
                    # mixture and clip the estimated watermark share to [0, 1].
                    values = estimate(counts/100, p0, 2)
                    groups[model, lam].extend(values)
                    for repeat, (count, value) in enumerate(zip(counts, values)):
                        emit(dataset, dict(model=model, prompt_id=row['prompt_id'], key_index=ki,
                            epsilon=2, p0=p0, n=100, true_lambda=lam, repeat=repeat,
                            favored_count=int(count), lambda_hat=float(value), x=lam, panel=''))
        print(f'Recomputed {dataset} {model} (M=20)', flush=True)
    rows = []
    for (model, lam), values in sorted(groups.items()):
        assert len(values) == 20000
        rows.append(dict(model=model, x=lam, **statistics(values, lam)))
    return rows
