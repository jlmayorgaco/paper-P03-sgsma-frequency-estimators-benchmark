# QR Card Copy

Short title:

OpenFreqBench

Short description:

Open source benchmark platform for grid frequency estimators. Install it, add
your estimator, run standard scenarios, and compare results with locked metrics.

Suggested QR target:

`https://github.com/jlmayorgaco/paper-P03-sgsma-frequency-estimators-benchmark/tree/MVP2.0.0`

First command:

```bash
python -m pip install "openfreqbench @ git+https://github.com/jlmayorgaco/paper-P03-sgsma-frequency-estimators-benchmark.git@MVP2.0.0"
openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1
```
