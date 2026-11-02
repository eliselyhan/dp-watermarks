# Figure guide

Run commands from `dp-watermarks/`.

| Figure in `figures/` | Command |
| --- | --- |
| `c4-main.png` | `python -m reproduction.run --figures c4-main` |
| `instruction-main.png` | `python -m reproduction.run --figures instruction-main` |
| `parallel.png` | `python -m reproduction.run --figures c4-main-parallel` |
| `parallel_m.png` | `python -m reproduction.run --figures parallel_m` |
| `rejection_m.png` | `python -m reproduction.run --figures rejection_m` |
| `temperature_prefix.png` | `python -m reproduction.run --figures temperature-prefix` |
| `pi0-bias-control.png` | `python -m reproduction.run --figures pi0-bias-control` |
| `rejection_samples_vs_epsilon.png` | `python -m reproduction.run --figures rejection-cost` |
| `epsilon_pure_rr_overlay.png` | `python -m reproduction.run --figures epsilon-rr-overlay` |
| `n_pure_rr_overlay.png` | `python -m reproduction.run --figures n-rr-overlay` |
| `distinct-prompts.png` | `python -m reproduction.run --figures distinct-prompts` |

`python -m reproduction.run` rebuilds all figures and collects them in `figures/`.
Use `--output-dir` and `--figure-dir` to override the analysis and PNG destinations.
Each run records the figure seeds and sampling rules in its analysis metadata.

The `epsilon-rr-overlay` and `n-rr-overlay` commands each compute their C4
sweep and pure-RR baseline directly. It writes only the combined figures, their band tables, metadata, and
C4 replicate samples named `epsilon_pure_rr_overlay.csv.gz` and
`n_pure_rr_overlay.csv.gz`. Standalone epsilon/n figures and a separate
pure-RR output directory are not generated.
