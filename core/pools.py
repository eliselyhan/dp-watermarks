"""Load disjoint response pools from bundled token IDs."""
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['qwen3_1p7b', 'qwen3_4b', 'qwen3_8b', 'llama31_8b']
LABELS = ['Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Llama3.1-8B-Instruct']


def records(path):
    with gzip.open(ROOT / path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def load_blocks(model, temperature, kind):
    """Load ordered candidate blocks and check that source draws are unique."""
    tag = 'temp_' + f'{temperature:g}'.replace('.', 'p')
    blocks, prompts, identities = [], [], set()
    for row in records(f'raw/c4/{model}/{tag}/{kind}.jsonl.gz'):
        block = [tuple(c['token_ids']) for c in row['completions']]
        assert len(block) == len(row['source_indices']) == 16
        for index in row['source_indices']:
            identity = (row['draw_batch'], row['prompt_id'], index)
            assert identity not in identities
            identities.add(identity)
        blocks.append(block)
        prompts.append(row['prompt_id'])
    return blocks, prompts, identities


@dataclass
class C4Pools:
    """Unwatermarked responses and ordered candidate blocks before selection."""

    base: list[tuple[int, ...]]
    candidates: list[list[tuple[int, ...]]]
    n_sweep_candidates: list[list[tuple[int, ...]]]


def load_c4_pools(model, temperature=1, *, include_n_sweep=False):
    """Load ready-to-use, disjoint response pools; no historical batch assembly."""
    tag = 'temp_' + f'{temperature:g}'.replace('.', 'p')
    directory = f'raw/c4/{model}/{tag}'
    # Pool construction: load the disjoint base responses and candidate
    # blocks; watermark selection happens later for each key and configuration.
    base_blocks, _, _ = load_blocks(model, temperature, 'base')
    base = [tokens for block in base_blocks for tokens in block]

    def candidate_blocks(filename):
        return [[tuple(tokens) for tokens in row['token_ids']]
                for row in records(f'{directory}/{filename}.jsonl.gz')]

    candidates = candidate_blocks('candidate_blocks')
    n_sweep = candidate_blocks('n_sweep_blocks') if include_n_sweep else []
    assert len(base) == 32000 and len(candidates) == 2000
    assert all(len(block) == (32 if temperature == 1 else 20) for block in candidates)
    if include_n_sweep:
        assert len(n_sweep) == 4500 and all(len(block) == 20 for block in n_sweep)
    return C4Pools(base, candidates, n_sweep)
