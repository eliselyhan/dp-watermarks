"""Write paper-figure metadata from the actual plotted and replicate data."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
from reproduction.io import digest
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['parallel', 'parallel_m', 'rejection_m']


# Metadata only: describe and validate saved estimates. Pool construction
# and randomized sampling occur in experiments/, with estimation in core/estimation.py.
def write_metadata(path, settings):
    if path.parent.name in ['c4-main', 'instruction-main']:
        return write_main_metadata(path, settings)
    name = path.stem
    if name not in NAMES:
        raise ValueError(f'Unsupported standalone ablation: {name}')
    dataset = name
    replicates = path.parent.parent/'replicates'/f'{dataset}.csv.gz'
    values = defaultdict(set)
    model_x = defaultdict(set)
    groups = Counter()
    with gzip.open(replicates, 'rt') as stream:
        for row in csv.DictReader(stream):
            for field in ['epsilon', 'p0', 'n', 'M', 'true_lambda', 'panel', 'x', 'key_index', 'repeat']:
                values[field].add(float(row[field]))
            model_x[row['model']].add(float(row['x']))
            groups[row['model'], float(row['panel']), float(row['x'])] += 1
    with path.with_suffix('.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(groups)
    assert all(groups[r['model'], float(r['panel']), float(r['x'])] == int(r['count']) for r in rows)
    assert values['key_index'] == set(range(len(settings['keys'])))
    assert values['repeat'] == set(range(settings['repeats']))
    parallel = dataset in ['parallel', 'parallel_m']
    lambda_plot = dataset == 'parallel'
    x_parameter = dict(parallel='lambda',
                       rejection_m='M', parallel_m='M')[dataset]
    parameters = {label: sorted(values[field]) for label, field in
                  [('epsilon','epsilon'), ('pi0','p0'), ('n','n'), ('M','M'), ('lambda','true_lambda')]}
    parameters['temperature_by_model'] = {model: [1.0] for model in model_x}
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    inputs = {'settings.json': digest(ROOT/'settings.json')}
    kinds = ['base', 'candidate_blocks']
    for model in model_x:
        for kind in kinds:
            relative = f'raw/c4/{model}/temp_1/{kind}.jsonl.gz'
            inputs[relative] = manifest[relative]
    metadata = dict(
        figure=path.name, dataset='C4', parameters=parameters,
        axes=dict(x_parameter=x_parameter, x_values=sorted(values['x']),
                  panel_parameter='epsilon' if lambda_plot else 'lambda', panel_values=sorted(values['panel']),
                  y_metric='mean clipped lambda estimate' if lambda_plot else 'MAPE (%)',
                  bands='empirical 5th and 95th percentiles of estimates' if lambda_plot else None),
        method=dict(selection='parallel' if parallel else 'rejection',
                    estimator='clip((favored_fraction - pi0)/(p1 - pi0), 0, 1)',
                    p1='min(pi0/exp(-epsilon), 1-exp(-epsilon)*(1-pi0))' if parallel else 'pi0/(pi0+exp(-epsilon)*(1-pi0))',
                    calibration='nominal; no empirical component rates',
                    mape='mean(100 * abs(clipped_lambda_hat - lambda) / lambda)',
                    analyzed_token_prefix=50, prompt_count=500,
                    base_pool_responses=32000, selected_pool_responses=2000,
                    source_draws_disjoint=True, mixture_sampling='without replacement within each component',
                    candidate_construction='2000 ready-to-use blocks of 32 candidates, truncated to M.'),
        sampling=dict(key_count=len(settings['keys']), keys=settings['keys'],
                      repeats_per_key=settings['repeats'], replicates_per_point=sorted(set(groups.values())),
                      seed=settings['figure_seeds'][dataset],
                      selection_seed='SeedSequence([figure_seed, key_index, 0])',
                      mixture_seed='SeedSequence([figure_seed, key_index, 1, n, round(lambda*1000)])',
                      size_plan='Only plotted sizes and lambdas; independent streams per point'),
        plotted_points=len(rows), replicate_rows=sum(groups.values()),
        reproduction_command=path.with_name(name+'_COMMAND.txt').read_text().strip(),
        files=dict(summary_csv=path.with_suffix('.csv').name, replicates='../replicates/'+replicates.name),
        sha256=dict(png=digest(path), summary_csv=digest(path.with_suffix('.csv')), replicates=digest(replicates)),
        input_sha256=inputs)
    target = path.with_name(name+'_metadata.json')
    target.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def write_main_metadata(path, settings):
    c4 = path.parent.name == 'c4-main'
    datasets = {'C4': 'c4-main'} if c4 else {'OUTFOX': 'outfox', 'LiveCodeBench': 'livecodebench'}
    groups, values, prompts, models = Counter(), defaultdict(set), defaultdict(set), defaultdict(set)
    conditions = set()
    replicate_paths = {}
    for label, dataset in datasets.items():
        replicates = path.parent.parent/'replicates'/f'{dataset}.csv.gz'
        replicate_paths[label] = replicates
        with gzip.open(replicates, 'rt') as stream:
            for row in csv.DictReader(stream):
                panel = float(row['panel']) if c4 else label
                groups[panel, row['model'], float(row['x'])] += 1
                for field in ['epsilon', 'p0', 'n', 'true_lambda', 'key_index', 'repeat']:
                    values[field].add(float(row[field]))
                if row.get('M'):
                    assert int(row['M']) == 20
                assert 0 <= float(row['lambda_hat']) <= 1
                models[label].add(row['model'])
                if row['prompt_id']:
                    prompts[label, row['model']].add(row['prompt_id'])
                conditions.add((float(row['epsilon']), float(row['p0']), int(row['n'])))
    with path.with_suffix('.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(groups)
    for row in rows:
        panel = float(row['panel']) if c4 else row['dataset']
        assert groups[panel, row['model'], float(row['x'])] == int(row['count']) == (1000 if c4 else 20000)
    assert values['key_index'] == set(range(len(settings['keys'])))
    assert values['repeat'] == set(range(settings['repeats']))
    from reproduction.experiments.c4 import configurations
    expected = {(eps, p0, sizes[0]) for eps, p0, sizes, _, cap in configurations('c4-main')} if c4 else {(2., .5, 100)}
    assert conditions == expected
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    inputs = {'settings.json': digest(ROOT/'settings.json')}
    for label, model_names in models.items():
        for model in model_names:
            paths = ([f'raw/c4/{model}/temp_1/{kind}.jsonl.gz' for kind in ['base', 'candidate_blocks']]
                     if c4 else [f'raw/{datasets[label]}/{model}.jsonl.gz'])
            for relative in paths:
                inputs[relative] = manifest[relative]
    method = dict(selection='rejection', calibration='nominal; no empirical component rates',
                  estimator='clip((favored_fraction - pi0)/(p1 - pi0), 0, 1)',
                  p1='pi0/(pi0+exp(-epsilon)*(1-pi0))',
                  source_draws_disjoint=True, mixture_sampling='without replacement within each component',
                  bands='empirical 5th and 95th percentiles; not confidence intervals')
    sampling = dict(key_count=len(settings['keys']), keys=settings['keys'], repeats_per_key_per_prompt=settings['repeats'],
                    replicates_per_point=sorted(set(groups.values())), seed=settings['figure_seeds']['c4-main' if c4 else 'instruction-main'])
    if c4:
        method.update(analyzed_token_prefix=50, prompt_count=500, base_pool_responses=32000,
                      selected_pool_responses=2000,
                      candidate_construction='2000 ready-to-use blocks of 32 candidates; take the first 20.')
        sampling['repeats_per_key'] = sampling.pop('repeats_per_key_per_prompt')
        sampling.update(selection_seed='SeedSequence([figure_seed, key_index, 0])',
                        mixture_seed='SeedSequence([figure_seed, key_index, 1, n, round(lambda*1000)])',
                        size_plan='Only plotted sizes and lambdas')
    else:
        assert all(len(ids) == 20 for ids in prompts.values())
        method['datasets'] = {
            label: dict(prompt_count_per_model=20, base_pool_responses_per_prompt=248 if label == 'OUTFOX' else 160,
                        candidate_draws_per_prompt=1800, selected_pool_responses_per_prompt=90,
                        analyzed_tokens='complete token sequence' if label == 'OUTFOX' else 'first 100 post-thinking tokens',
                        candidate_construction='Split base draws first, then partition remaining 1800 draws into 90 disjoint blocks of 20.')
            for label in datasets}
        sampling.update(seed_sequence='SeedSequence([figure_seed, model_index, prompt_id, key_index, is_livecodebench]).spawn(2)',
                        model_order=['qwen3_1p7b', 'qwen3_4b', 'qwen3_8b', 'llama31_8b'],
                        mixture_seed='Second spawned sequence with spawn_key extended by (100, round(lambda*10))',
                        count_sampling='Independent hypergeometric draws for base and selected pools, 50 replicates per condition')
    metadata = dict(figure=path.name, dataset='C4' if c4 else ['OUTFOX', 'LiveCodeBench'],
        parameters=dict(epsilon=sorted(values['epsilon']), pi0=sorted(values['p0']), pi0_rule='fixed 0.5',
                        n=sorted(values['n']), M=[20], **{'lambda': sorted(values['true_lambda'])},
                        coupled_conditions=[dict(epsilon=e, pi0=p, n=n, M=20) for e,p,n in sorted(conditions)],
                        models_by_dataset={label: sorted(names) for label,names in models.items()}),
        axes=dict(x_parameter='lambda', x_values=sorted(values['true_lambda']),
                  panel_parameter='epsilon' if c4 else 'dataset',
                  panel_values=sorted(values['epsilon']) if c4 else list(datasets),
                  y_metric='mean clipped lambda estimate', bands=method['bands']),
        method=method, sampling=sampling, plotted_points=len(rows), replicate_rows=sum(groups.values()),
        reproduction_command=path.with_name(path.stem+'_COMMAND.txt').read_text().strip(),
        files=dict(summary_csv=path.with_suffix('.csv').name,
                   replicates={label:'../replicates/'+p.name for label,p in replicate_paths.items()}),
        sha256=dict(png=digest(path), summary_csv=digest(path.with_suffix('.csv')),
                    replicates={label:digest(p) for label,p in replicate_paths.items()}), input_sha256=inputs)
    if c4:
        metadata['parameters']['temperature'] = [1.0]
    path.with_name(path.stem+'_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    args = parser.parse_args()
    settings = json.loads((ROOT/'settings.json').read_text())
    paths = [args.output/'c4-ablation'/f'{name}.png' for name in NAMES]
    for directory in ['c4-main', 'instruction-main']:
        paths.extend(sorted((args.output/directory).glob('*.png')))
    for path in paths:
        metadata = write_metadata(path, settings)
        print(f'{path.stem}: {metadata["plotted_points"]} points; metadata written')


if __name__ == '__main__':
    main()
