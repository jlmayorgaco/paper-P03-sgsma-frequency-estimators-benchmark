# QR Card Copy

Short title:

OpenFreqBench

Short description:

Open source benchmark platform for grid frequency estimators. Install it, add
your estimator, run standard scenarios, and compare results with locked metrics.

Suggested QR target:

`https://github.com/openfreqbench/openfreqbench`

First command:

```bash
python -m pip install openfreqbench
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1
```

