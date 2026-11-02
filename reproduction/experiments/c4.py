"""C4 main figure and epsilon, sample-size, and candidate-count ablations."""
from collections import defaultdict
import numpy as np
from core.hashing import prepare, hashes, colors
from core.sampling import select, mixture_plan, figure_rng
from core.estimation import estimate, statistics
from core.pools import MODELS, load_c4_pools

LAMBDAS = [.1, .5, .9]
SIZES = [50, 100, 500, 1000, 2000, 5000]


def configurations(name):
    """Yield epsilon, pi0, sizes, lambdas, and candidate count."""
    if name == 'c4-main':
        return [(e, .5, [2000],
                 [i/10 for i in range(1, 10)], 20) for e in [1, 2, 3]]
    if name == 'epsilon':
        return [(e, .5, [2000], LAMBDAS, 20) for e in [.5, 1, 2, 3, 4, 5]]
    if name == 'n':
        return [(2, .5, SIZES, LAMBDAS, 20)]
    if name == 'parallel':
        return [(e, .5, [2000], [i/10 for i in range(1, 10)], 20) for e in [1, 2, 3]]
    if name in ['rejection_m', 'parallel_m']:
        p0 = .5
        return [(2, p0, [2000], LAMBDAS, m) for m in [2, 4, 8, 20, 32]]
    raise ValueError(f'Unknown C4 figure: {name}')


def run(names, settings, emit):
    """Build disjoint pools, replay each key/configuration, then summarize estimates."""
    groups = {name: defaultdict(list) for name in names}
    if not names:
        return {}
    for model in MODELS:
        pools = load_c4_pools(model, include_n_sweep='n' in names)
        base = pools.base
        # Hash each token sequence once across all requested configurations.
        candidate_blocks = pools.candidates + pools.n_sweep_candidates
        prepared = prepare(base + [tokens for block in candidate_blocks for tokens in block])
        for key_index, key in enumerate(settings['keys']):
            cache = hashes(prepared, key)
            plans = {}
            for name in names:
                configs = configurations(name)
                for epsilon, p0, sizes, lambdas, candidate_cap in configs:
                    pool = pools.n_sweep_candidates if name == 'n' else pools.candidates
                    pool = [block[:candidate_cap] for block in pool]
                    if any(len(block) != candidate_cap for block in pool):
                        raise ValueError(f'{name}: insufficient draws for M={candidate_cap}')
                    base_colors = colors(base, cache, p0)
                    candidate_colors = colors([t for block in pool for t in block], cache, p0).reshape(-1, candidate_cap)
                    parallel = name in ['parallel', 'parallel_m']
                    # Pool construction: keyed selection yields one watermarked favored bit
                    # per candidate block; base_colors represents the unwatermarked pool.
                    selected_colors = select(candidate_colors, epsilon, figure_rng(settings['figure_seeds'][name], key_index), parallel)
                    identity = (name, tuple(sizes), tuple(lambdas))
                    # Randomized samples: cache seeded mixture indices so compared
                    # configurations reuse draws for the same figure, sizes, and lambdas.
                    if identity not in plans:
                        plans[identity] = list(mixture_plan(len(base_colors), len(selected_colors), sizes, lambdas, key_index,
                            settings['figure_seeds'][name], settings['repeats']))
                    # Mix base and selected responses, then estimate the hidden mixture share.
                    for n, lam, repeat, marked_indices, base_indices in plans[identity]:
                        count = int(selected_colors[marked_indices].sum() + base_colors[base_indices].sum())
                        # Estimator: pass only the combined favored fraction to core.estimation.estimate;
                        # the true lambda is used for constructing mixtures and scoring results.
                        value = float(estimate(count/n, p0, epsilon, parallel))
                        x = dict(epsilon=epsilon, n=n, parallel=lam,
                                 rejection_m=candidate_cap, parallel_m=candidate_cap).get(name, lam)
                        if name == 'c4-main':
                            x = lam
                        panel = epsilon if name in ['parallel', 'c4-main'] else lam
                        groups[name][model, panel, x, lam].append(value)
                        emit(name, dict(model=model, prompt_id='', key_index=key_index, epsilon=epsilon,
                            p0=p0, n=n, M=candidate_cap, true_lambda=lam, repeat=repeat,
                            favored_count=count, lambda_hat=value, x=x, panel=panel))
        print(f'Recomputed C4 {model}, temperature=1', flush=True)
    result = {}
    for name, grouped in groups.items():
        result[name] = []
        for (model, panel, x, lam), values in sorted(grouped.items()):
            assert len(values) == 1000
            result[name].append(dict(model=model, panel=panel, x=x, **statistics(values, lam)))
    return result
