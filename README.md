# Reconciling Privacy with Provenance for Language Model Watermarks
[arxiv link]
## Reproduce the paper figures

Rebuild all 11 paper figures from bundled completion token IDs on CPU. Use Python 3.13.

From this directory:

```bash
python -m pip install -r requirements.txt
python -m reproduction.run
python -m reproduction.calculate_metrics
```

Figures are saved to `figures/`; analysis tables and metadata go to `output/`.

To rebuild selected figures:

```bash
python -m reproduction.run --figures c4-main c4-main-parallel
python -m reproduction.run --figures epsilon-rr-overlay n-rr-overlay
python -m reproduction.run --help
```

See the [figure guide](docs/figures.md) and [metric definitions](docs/metrics.md).

## Generate watermarked responses

`generate_rgs_samples.py` samples candidate completions from a Hugging Face model using vLLM, then
applies the RGS watermark with the keyed hash to obtain watermarked responses at a desired privacy level epsilon. This requires an NVIDIA GPU with a compatible CUDA driver and model access; generation dependencies are included in `requirements.txt`.

For example, run Llama 3.1 8B Instruct with the rejection sampler:

```bash
python generate_rgs_samples.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompt "Explain why the sky is blue." \
  --num-samples 5 --sampler rejection --epsilon 2 --p0 0.5 --M 20 --key 42
```

Use a Hugging Face model ID or local checkpoint for `--model`. To use the parallel
sampler, pass `--sampler parallel`. Set desired privacy level with `--epsilon`. `--p0` sets the nominal unwatermarked green
fraction (default 0.5). Set `--M` to change the candidate budget for the sampler (default 20).
Use `--key` to specify the integer watermark key (for example, `--key 42`).

Outputs are saved to `generated/`.

## Layout

| Path | Contents |
| --- | --- |
| `reproduction/run.py` | Figure reproduction entry point |
| `reproduction/calculate_metrics.py` | Metric calculation entry point |
| `reproduction/experiments/` | C4, OUTFOX/LiveCodeBench, and other experiments |
| `core/` | Pool loading, hashing, sampling, and estimation |
| `reproduction/plotting.py` | Figure rendering |
| `reproduction/overlays.py` | Combined C4 and pure-RR experiments and figures |
| `reproduction/metadata.py`, `io.py` | Metadata and file helpers |
| `tasks/` | Prompt inputs for model generation |
| `raw/` | Ordered completion token IDs and reference counts |
| `settings.json`, `data_manifest.json` | Experiment settings and input checksums |
| `figures/` | 11 paper PNGs |
| `docs/` | Figure guide and metric definitions |
| `output/` | Generated results (ignored by Git) |
