# Release Checklist

Use this checklist before a public GitHub release, PyPI upload, award submission,
or public leaderboard announcement. For the MVP 2.0.0 branch, the package lives
at the repository root.

## Repository

- The package checkout is clean.
- Public repository URL is correct in `pyproject.toml` and `CITATION.cff`.
- License, code of conduct, contributing guide, and security policy are present.
- GitHub Actions CI is present under `.github/workflows/`.
- Issue templates are present under `.github/ISSUE_TEMPLATE/`.
- `docs/AI_USAGE_DISCLOSURE.md` is present and reflected in the manuscript.
- A software-paper package is present under `paper/` or `manuscripts/softwarex/`.
- For PyPI, confirm that the `openfreqbench` name is available or update the
  package name before upload.

## Scientific Artifacts

- Run `openfreqbench quality-gate`.
- Run `openfreqbench quality-gate --release` from a clean tree.
- Run a tuned replay with `parameter_policy: artifact_tuned`.
- Archive `benchmark_report.json`, `analysis_summary.json`, and generated plots.
- Run preregistered hypotheses and archive statistical outputs.
- Store the reproducibility manifest with source hashes and checkpoint hashes.

## Installation

- Build a wheel in a clean environment.
- Install the wheel in a new virtual environment.
- Run `openfreqbench doctor`.
- Run `openfreqbench quick-test --scenario IEEE_Single_SinWave --estimator ZCD --n-runs 1`.
- Run `scripts/verify_local.ps1` or `scripts/verify_local.sh`.
- Create an annotated release tag on the exact release commit.
- Run `scripts/verify_release.ps1` or `scripts/verify_release.sh` from the tagged commit.

## Communication

- Claims cite report paths or hashes.
- CPU claims state hardware/runtime.
- PI-GRU claims state checkpoint hash and torch version.
- The README clearly labels unsupported three-phase/WAMS features as roadmap.
- The release notes distinguish software registry scope from any paper-grade
  demonstration matrix.
