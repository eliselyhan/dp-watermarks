"""Reproduce all paper figures from bundled raw completions into output/."""
import argparse
from reproduction.io import write_csv
import csv
import gzip
from contextlib import ExitStack
import json
from pathlib import Path
import shlex
import subprocess
import shutil
import sys

# Data and outputs stay at the bundle root, one level above these scripts.
HERE = Path(__file__).resolve().parents[1]
FIGURES = {
    'c4-main': 'c4-main/c4-main.png',
    'instruction-main': 'instruction-main/instruction-main.png',
    'c4-main-parallel': 'c4-ablation/parallel.png',
    **{name: f'c4-ablation/{name}.png' for name in
       ['parallel_m', 'rejection_m']},
}


def generate(selected=None, output=None):
    from .experiments import c4, instruction
    from . import plotting
    from reproduction.metadata import write_metadata

    selected = list(FIGURES) if selected is None else selected
    output = HERE / 'output' if output is None else output
    settings = json.loads((HERE / 'settings.json').read_text())
    command = shlex.join([sys.executable, *sys.argv])
    # Stream individual estimates to disk; retain only summaries for plotting.
    with ExitStack() as stack:
        writers = {}
        def emit(dataset, row):
            if dataset not in writers:
                path = output / 'replicates' / f'{dataset}.csv.gz'
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context(gzip.open(path, 'wt', newline=''))
                writers[dataset] = csv.DictWriter(stream, fieldnames=list(row))
                writers[dataset].writeheader()
            writers[dataset].writerow(row)
        # Pool construction, randomized samples, and estimation are delegated
        # to instruction.run for instruction tasks and c4.run for C4 figures.
        # Their call sites identify each stage; core/ supplies shared methods.
        outfox = instruction.run('outfox', settings, emit) if 'instruction-main' in selected else None
        coding = instruction.run('livecodebench', settings, emit) if 'instruction-main' in selected else None
        ablations = ['parallel' if name == 'c4-main-parallel' else name
                     for name in selected if name != 'instruction-main']
        results = c4.run(ablations, settings, emit) if ablations else {}
    # Render each summary and record the inputs and command used to produce it.
    for name in selected:
        path = output / FIGURES[name]
        if name == 'c4-main':
            rows = results[name]
            plotting.c4_main(rows, path)
        elif name == 'instruction-main':
            plotting.main_figure(outfox, coding, path)
            rows = [dict(dataset='OUTFOX', **r) for r in outfox]
            if coding:
                rows += [dict(dataset='LiveCodeBench', **r) for r in coding]
        else:
            dataset = 'parallel' if name == 'c4-main-parallel' else name
            rows = results[dataset]
            plotting.ablation(dataset, rows, path)
        write_csv(path.with_suffix('.csv'), rows)
        path.with_name(path.stem + '_COMMAND.txt').write_text(command + '\n')
        write_metadata(path, settings)
        print(f'{name}: generated -> {path}', flush=True)


# Specialized figures implement their stages in the helper modules below;
# this runner coordinates execution and collects the finished figures.
STEPS = [
    ('pi0-bias-control', ['-m', 'reproduction.experiments.bias_control']),
    ('rejection-cost', ['-m', 'reproduction.experiments.rejection_cost']),
    ('epsilon-rr-overlay', ['-m', 'reproduction.overlays', '--sweeps', 'epsilon']),
    ('n-rr-overlay', ['-m', 'reproduction.overlays', '--sweeps', 'n']),
    ('distinct-prompts', ['-m', 'reproduction.experiments.distinct_prompts']),
    ('temperature-prefix', ['-m', 'reproduction.experiments.temperature_prefix']),
]


# Public figure names map directly to their PNG outputs.
# Each overlay computes its C4 sweep and pure-RR baseline.
GALLERY = {
    'c4-main': [FIGURES['c4-main']],
    'instruction-main': [FIGURES['instruction-main']],
    **{name: [FIGURES[name]] for name in ['c4-main-parallel', 'parallel_m', 'rejection_m']},
    'temperature-prefix': ['temperature-prefix/temperature-prefix.png'],
    'pi0-bias-control': ['c4-ablation/pi0-bias-control.png'],
    'rejection-cost': ['rejection-cost/rejection_samples_vs_epsilon.png'],
    'epsilon-rr-overlay': ['c4-ablation/epsilon_pure_rr_overlay.png'],
    'n-rr-overlay': ['c4-ablation/n_pure_rr_overlay.png'],
    'distinct-prompts': ['c4-ablation/distinct-prompts.png'],
}


def helper_output(command, output):
    """Pass the analysis directory explicitly to each standalone helper."""
    command = list(command)
    module = command[1]
    if module == 'reproduction.overlays':
        command += ['--output-dir', str(output)]
    else:
        subdir = {'reproduction.experiments.rejection_cost': 'rejection-cost',
                  'reproduction.experiments.temperature_prefix': 'temperature-prefix'}.get(module, '')
        command += ['--output', str(output/subdir)]
    return command


def collect_figures(selected, output, destination):
    """Publish the selected seeded figures under their concise paper names."""
    destination.mkdir(parents=True, exist_ok=True)
    for name, paths in GALLERY.items():
        if name in selected:
            for relative in paths:
                filename = 'temperature_prefix.png' if name == 'temperature-prefix' else Path(relative).name
                source, target = output/relative, destination/filename
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
    print(f'Paper PNGs: {destination}', flush=True)


def main(argv=None):
    choices = [*FIGURES, *dict(STEPS)]
    parser = argparse.ArgumentParser(
        description=__doc__, epilog='Figures: ' + ', '.join(choices),
    )
    parser.add_argument(
        '--figures', nargs='+', metavar='NAME', choices=choices,
        help='Rebuild only these figures (default: all). Each overlay computes its C4 sweep and pure-RR baseline.',
    )
    parser.add_argument('--output-dir', type=Path, help='Directory for new analysis tables, replicates, and plots')
    parser.add_argument('--figure-dir', type=Path, default=HERE/'figures',
                        help='Paper PNG folder (default: figures)')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve() if args.output_dir else HERE/'output'
    selected = set(args.figures) if args.figures else set(FIGURES) | set(dict(STEPS))
    core = [name for name in FIGURES if name in selected]
    steps = [command for name, command in STEPS if name in selected]
    total = bool(core) + len(steps)
    if core:
        print(f'[1/{total}] Rebuilding {", ".join(core)}', flush=True)
        generate(core, output) if args.output_dir else generate(core)
    for index, step in enumerate(steps, start=1 + bool(core)):
        if args.output_dir:
            step = helper_output(step, output)
        print(f'[{index}/{total}] {shlex.join(step)}', flush=True)
        subprocess.run([sys.executable, '-u', *step], cwd=HERE, check=True)
    collect_figures(selected, output, args.figure_dir.resolve())
    print(f'Done: {output}', flush=True)


if __name__ == '__main__':
    main()
