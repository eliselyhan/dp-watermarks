#!/usr/bin/env python3
"""Generate response-level RGS watermarked samples from a Hugging Face model."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from core.hashing import colors, hashes, prepare
from core.sampling import select_index


def favored(completions, key, p0):
    """Hash full completion token sequences, excluding the input prompt."""
    sequences = [tuple(c.token_ids) for c in completions]
    return colors(sequences, hashes(prepare(sequences), key), p0)


def watermarked_sample(draw, sampler, epsilon, candidates, key, p0, rng):
    """Return (completion, favored bit, draws, cap fallback) for one sample."""
    if sampler == 'parallel':
        completions = draw(candidates)
        bits = favored(completions, key, p0)
        index = select_index(bits, epsilon, rng, parallel=True)
        return completions[index], bool(bits[index]), candidates, False
    for attempt in range(1, candidates + 1):
        completion, = draw(1)
        bit = bool(favored([completion], key, p0)[0])
        if bit or rng.random() < math.exp(-epsilon):
            return completion, bit, attempt, False
    return completion, bit, candidates, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Hugging Face model ID or local checkpoint')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='One prompt')
    source.add_argument('--prompts-file', type=Path, help='JSON array of prompt strings')
    parser.add_argument('--sampler', choices=['rejection', 'parallel'], default='rejection')
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--key', type=int, required=True, help='Integer watermark key (same convention as core.hashing)')
    parser.add_argument('--p0', type=float, default=.5, help='Nominal favored fraction (default: 0.5)')
    parser.add_argument('--num-samples', type=int, default=1, help='Samples per prompt')
    parser.add_argument('--M', '--candidates', dest='candidates', type=int, default=20,
                        help='Candidate budget M: rejection cap or parallel batch size (default: 20)')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--raw-prompt', action='store_true', help='Skip the chat template for text completion')
    parser.add_argument('--output', type=Path, default=Path('generated/samples.jsonl'))
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.9)
    parser.add_argument('--dry-run', action='store_true', help='Validate and print the plan without loading a model')
    args = parser.parse_args(argv)
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error('--epsilon must be finite and nonnegative')
    if not 0 < args.p0 < 1:
        parser.error('--p0 must be in (0, 1)')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive')
    if min(args.num_samples, args.candidates, args.max_tokens,
           args.tensor_parallel_size, args.max_model_len) < 1:
        parser.error('Sample counts, candidate counts, token limits and parallel size must be positive')
    if args.max_tokens >= args.max_model_len:
        parser.error('--max-tokens must leave room for the prompt in --max-model-len')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('--gpu-memory-utilization must be in (0, 1)')
    prompts = ([args.prompt] if args.prompt is not None
               else json.loads(args.prompts_file.read_text()))
    if not isinstance(prompts, list) or not prompts or any(
            not isinstance(p, str) or not p.strip() for p in prompts):
        parser.error('Provide a nonempty prompt or a nonempty JSON array of nonempty strings')
    settings = {k: v for k, v in vars(args).items()
                if k not in ('key', 'prompt', 'prompts_file', 'output', 'dry_run')}
    print(json.dumps(dict(settings, prompts=len(prompts), output=str(args.output)), indent=2))
    if args.dry_run:
        return
    metadata_path = args.output.with_suffix('.metadata.json')
    for path in (args.output, metadata_path):
        if path.exists():
            parser.error(f'Output already exists: {path}; choose another --output')

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype='auto', seed=args.seed,
              tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              generation_config='vllm')
    tokenizer = llm.get_tokenizer()
    encoded = []
    for prompt in prompts:
        if args.raw_prompt or not tokenizer.chat_template:
            ids = tokenizer.encode(prompt)
        else:
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, return_dict=False)
        if len(ids) + args.max_tokens > args.max_model_len:
            parser.error('Prompt plus --max-tokens exceeds --max-model-len')
        encoded.append(ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        with metadata_path.open('x') as metadata:
            json.dump({'key': args.key}, metadata, indent=2)
            metadata.write('\n')
        for prompt_id, (prompt, ids) in enumerate(zip(prompts, encoded)):
            for sample_id in range(args.num_samples):
                generation_rng = np.random.default_rng([args.seed, prompt_id, sample_id, 0])
                selection_rng = np.random.default_rng([args.seed, prompt_id, sample_id, 1])

                def draw(count):
                    params = SamplingParams(n=count, temperature=args.temperature,
                        top_p=1., top_k=-1, min_p=0., max_tokens=args.max_tokens,
                        seed=int(generation_rng.integers(0, 2**32)), ignore_eos=False)
                    result, = llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)
                    completions = sorted(result.outputs, key=lambda c: c.index)
                    if len(completions) != count:
                        raise RuntimeError('Unexpected completion count')
                    return completions

                completion, bit, draws, fallback = watermarked_sample(
                    draw, args.sampler, args.epsilon, args.candidates,
                    args.key, args.p0, selection_rng)
                row = dict(prompt_id=prompt_id, sample_id=sample_id, prompt=prompt,
                           text=completion.text, token_ids=list(completion.token_ids),
                           favored=bit, candidates_drawn=draws, cap_fallback=fallback,
                           settings=settings)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                print(f'Prompt {prompt_id}, sample {sample_id}: {draws} candidate(s)', flush=True)
    print(f'Saved {len(prompts) * args.num_samples} samples to {args.output}')


if __name__ == '__main__':
    main()
