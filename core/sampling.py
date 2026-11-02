"""Watermark selection and seeded sampling of response mixtures."""
import math
import numpy as np


def select_index(block, epsilon, rng, parallel=False):
    """Select a candidate index, preserving duplicates and draw order."""
    if parallel:
        p = float(np.mean(block))
        probabilities = np.ones(len(block)) / len(block)
        if 0 < p < 1:
            decay = math.exp(-epsilon)
            q = min(p/decay if decay else float('inf'), 1-decay*(1-p))
            probabilities = np.where(block, probabilities*q/p,
                                     probabilities*(1-q)/(1-p))
            probabilities /= probabilities.sum()
        return int(rng.choice(len(block), p=probabilities))
    for j, favored in enumerate(block):
        if favored or rng.random() < math.exp(-epsilon):
            return j
    return len(block)-1


def select(blocks, epsilon, rng, parallel=False):
    """Select one raw candidate per block; retain multiplicity and draw order."""
    return np.asarray([block[select_index(block, epsilon, rng, parallel)]
                       for block in blocks], dtype=np.int8)


def figure_rng(seed, key_index=0, stream=0, *coordinates):
    """Independent deterministic stream: [figure seed, key, purpose, coordinates].

    Purpose 0 selects watermarked responses, 1 samples mixtures, and 2 bootstraps.
    Parameters being compared are omitted so their random draws are paired.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, key_index, stream, *coordinates]))


def mixture_plan(base_size, marked_size, sizes, lambdas, key_index, seed, repeats=50):
    """Sample only requested points; each (n, lambda) has an independent stream.

    Sampling is without replacement within each component. Adding or reordering
    other plotted points cannot change the samples for an existing point.
    """
    # Randomized samples: draw component indices without replacement within
    # each mixture; pools may be reused across independent repeats.
    for n in sizes:
        for lam in lambdas:
            marked_count = round(n * lam)
            base_count = n - marked_count
            if marked_count > marked_size or base_count > base_size:
                raise ValueError(f'Insufficient responses for n={n}, lambda={lam}')
            rng = figure_rng(seed, key_index, 1, n, round(lam * 1000))
            for repeat in range(repeats):
                marked = rng.choice(marked_size, size=marked_count, replace=False)
                base = rng.choice(base_size, size=base_count, replace=False)
                yield n, lam, repeat, marked, base
