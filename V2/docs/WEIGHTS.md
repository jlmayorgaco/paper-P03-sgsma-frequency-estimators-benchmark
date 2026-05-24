# PI-GRU Weights

OpenFreqBench includes the active PI-GRU checkpoint for reproducible local
validation.

The reproducibility manifest records a SHA-256 hash for the included
`pi_gru_weights_hybrid.pt` checkpoint. Inspect it with:

```bash
openfreqbench quality-gate --skip-tests
```

## Packaging Policy

For MVP 2.0.0, the wheel includes only the active default checkpoint:

- `pi_gru_weights_hybrid.pt`

Historical and experimental checkpoints are intentionally left in `V0/` rather
than the public MVP package. If a researcher wants to use another checkpoint,
they should place it next to `estimators/pi_gru.py` in a source checkout or pass
a custom estimator implementation.

Any paper or leaderboard claim using PI-GRU must report the checkpoint hash and
torch version from `benchmark_report.json`.
