# Release Checklist (`v1.0.0-rc1` and `v1.0.0`)

1. Freeze environment
- Confirm Python version and dependency lock state.
- Run `openfreqbench env doctor`.

2. Validate canonical gates
- Run `openfreqbench quality-gate --profile canonical`.
- Run `openfreqbench benchmark validate`.

3. Validate nightly flow
- Run local smoke:
  - `openfreqbench quality-gate --profile manual-nightly` (or filtered with `BENCHMARK_ALLOW_PARTIAL_CANONICAL=1` for smoke).
- Run:
  - `openfreqbench andes run-ieee39`
  - `openfreqbench stats run`
  - `openfreqbench report build`

4. Reproducibility evidence
- Attach:
  - commit hash
  - `reports/benchmark_run_manifest.json`
  - `artifacts/full_mc_benchmark/statistical_tests_report.json`

5. Release metadata
- Update `CHANGELOG.md`.
- Verify `CITATION.cff`.
- Verify `.zenodo.json`.

6. Tag and publish
- Create `vX.Y.Z-rc1` then `vX.Y.Z`.
- Confirm GitHub Action `release-pypi` succeeds.
