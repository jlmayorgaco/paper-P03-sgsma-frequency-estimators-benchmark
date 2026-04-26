# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Layered quality gates (`canonical`, `legacy`, `manual-nightly`).
- Canonical artifact validator for benchmark completeness.
- OSS packaging (`pyproject.toml`) and CLI entry points.
- CI workflows for canonical PR checks and nightly benchmark runs.
- Governance and contribution docs.
- Targeted CLI benchmark execution modes:
  - `openfreqbench benchmark run-scenario`
  - `openfreqbench benchmark run-estimator`
  - `openfreqbench benchmark run-matrix`
  - `--dry-run` filter preview for scoped runs.

### Changed
- Canonical plotting path now fails fast when canonical reports are missing.
- Legacy fallback for plotting is restricted to explicit legacy mode.
