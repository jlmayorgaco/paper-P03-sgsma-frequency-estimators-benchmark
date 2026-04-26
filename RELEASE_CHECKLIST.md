# Release Checklist (`vX.Y.Z-rc1` and `vX.Y.Z`)

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

4. Validate CLI surface contract
- Ensure help and command tree are stable:
  - `openfreqbench --help`
  - `openfreqbench benchmark --help`
  - `openfreqbench stats run --help`
  - `openfreqbench report build --help`
- Ensure roadmap CLI run modes are documented (current and planned).

5. Reproducibility evidence
- Attach:
  - commit hash
  - `reports/benchmark_run_manifest.json`
  - `artifacts/full_mc_benchmark/statistical_tests_report.json`

6. Release metadata
- Update `CHANGELOG.md`.
- Verify `CITATION.cff`.
- Verify `.zenodo.json`.

7. Tag and publish
- Create `vX.Y.Z-rc1` then `vX.Y.Z`.
- Confirm GitHub Action `release-pypi` succeeds.
