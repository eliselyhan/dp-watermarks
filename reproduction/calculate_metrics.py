"""Calculate headline metrics from existing reproduction outputs."""
import csv
import gzip
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "output"

    # Read estimates already produced by c4.run and instruction.run.
    # Pool construction, randomized mixtures, and estimation happen there;
    # this script only aggregates their saved lambda_hat values into metrics.
    for task, epsilon in [('c4-main', 1), ('outfox', 2)]:
        groups = defaultdict(list)
        with gzip.open(root / 'replicates' / f'{task}.csv.gz', 'rt') as f:
            for row in csv.DictReader(f):
                if float(row['epsilon']) == epsilon:
                    condition = (row['model'], float(row['true_lambda']))
                    groups[condition].append(float(row['lambda_hat']))

        assert len(groups) == 36
        mse = statistics.mean(
            statistics.mean((value - lam) ** 2 for value in values)
            for (_, lam), values in groups.items()
        )
        bias = statistics.mean(
            abs(statistics.mean(values) - lam)
            for (_, lam), values in groups.items()
        )
        print(f'{task}, epsilon={epsilon}, conditions={len(groups)}')
        print(f'  MSE={mse:.12f}; 100*MSE={100*mse:.10f}')
        print(f'  absolute bias={bias:.12f}; percentage points={100*bias:.10f}')

    summary = root / 'c4-main' / (
        'c4-main.csv'
    )
    with summary.open() as f:
        rows = list(csv.DictReader(f))

    for epsilon in (1, 2, 3):
        conditions = [r for r in rows if float(r['panel']) == epsilon]
        assert len(conditions) == 36
        bias = statistics.mean(
            abs(float(r['mean']) - float(r['x'])) for r in conditions
        )
        halfwidth = statistics.mean(
            (float(r['p95']) - float(r['p05'])) / 2 for r in conditions
        )
        print(f'C4 epsilon={epsilon}: bias={bias:.10f}, halfwidth={halfwidth:.10f}')


if __name__ == "__main__":
    main()
