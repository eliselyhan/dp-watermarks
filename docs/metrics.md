# How the frequency-estimation numbers are calculated

The paper's headline numbers summarize frequency estimates from C4 and the OUTFOX essay-writing task. Each metric gives equal weight to every model–true-frequency pair. Bias means **average absolute bias**, not the average absolute error of individual estimates.

## Reproduce the calculations

Run from the repository root (`dp-watermarks/`):

```bash
python -m reproduction.calculate_metrics
```

The script uses only the Python standard library and reads existing results from
`output/`. If those results have not been generated, run:

```bash
python -m reproduction.run --figures c4-main instruction-main
python -m reproduction.calculate_metrics
```

It reads `output/replicates/c4-main.csv.gz` and
`output/replicates/outfox.csv.gz` to calculate MSE and average absolute bias for
C4 at ε = 1 and OUTFOX at ε = 2. Each calculation groups estimates into 36
model–true-frequency conditions. Within each condition, MSE is the mean squared
difference between the estimate and the true frequency; absolute bias is the
absolute difference between the mean estimate and the true frequency. The
script then averages each metric equally across conditions.

It also reads `output/c4-main/c4-main.csv` to calculate average absolute bias
and average percentile-band half-width for C4 at ε = 1, 2, and 3. Half-width is
`(p95 - p05) / 2`, averaged equally across the 36 conditions. These are empirical
5th–95th percentile bands of replicate estimates, not confidence intervals for
their mean.

## Results

The existing bundled-input reproduction outputs give:

| Task | Privacy parameter, ε | MSE | MSE × 100 | Average absolute bias | Bias in percentage points |
| --- | ---: | ---: | ---: | ---: | ---: |
| C4 | 1 | 0.0019404390 | 0.1940438992 | 0.0038069324 | 0.3806932381 |
| OUTFOX | 2 | 0.0113060733 | 1.1306073331 | 0.0039807535 | 0.3980753454 |

Bias in percentage points is `100 × absolute bias`. The script also prints
`100 × MSE` as a scaled reporting value.

| C4 privacy parameter, ε | Average absolute bias | Average percentile-band half-width |
| --- | ---: | ---: |
| 1 | 0.0038069324 | 0.0717876518 |
| 2 | 0.0036693615 | 0.0402965058 |
| 3 | 0.0009578488 | 0.0311129204 |
