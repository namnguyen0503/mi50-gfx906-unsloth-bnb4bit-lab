# FineTome Token Stats (train[:3000])

Dataset sample analyzed: 3000 rows.

## Summary statistics

- count: 3000
- mean: 568.05
- p50: 457
- p75: 662
- p90: 1003
- p95: 1320
- p99: 2195
- max: 6531

## Threshold ratios

- `>2048`: 44 / 3000 = 1.47%
- `>3072`: 10 / 3000 = 0.33%
- `>4096`: 4 / 3000 = 0.13%
- `>8192`: 0

## Practical impact

For this sampled split, very long sequences are rare. This supports focusing on stable mid-length settings unless task requirements require explicit long-context coverage.
