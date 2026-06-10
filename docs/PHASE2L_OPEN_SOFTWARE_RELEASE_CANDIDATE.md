# Phase 2-L Open Software Release Candidate

Date: 2026-06-10

Purpose: convert the Phase 2-K software-first audit into concrete release
candidate infrastructure.

## Changes

- Added GitHub Actions CI at `.github/workflows/ci.yml`.
- Added issue templates for bug reports and feature requests.
- Added `docs/AI_USAGE_DISCLOSURE.md`.
- Added a JOSS-oriented software-paper skeleton in `paper/paper.md`.
- Added release-file checks to `openfreqbench quality-gate --release`.
- Added unit tests for the release-file checks.
- Added estimator-level `REFERENCE_KEYS`, a DOI-backed reference registry, and
  `docs/ESTIMATOR_REFERENCES.md`.
- Replaced the initial software-paper bibliography placeholder with verified
  method-family and PMU-context citations.

## Release Gate Additions

`openfreqbench quality-gate --release` now checks:

- clean Git checkout;
- remote origin;
- `CITATION.cff`;
- at least one tag pointing at `HEAD`;
- CI workflow presence;
- issue template presence;
- software-paper package presence;
- AI disclosure presence;
- `.zenodo.json` presence.

The tag check is intentionally strict. A public release should run the release
gate from the exact commit that will be archived.

## Remaining Before Public Release

1. Create an annotated release tag, for example `v2.1.0`, when the release
   candidate is final.
2. Run `scripts/verify_release.ps1` or `scripts/verify_release.sh` from the
   tagged commit.
3. Create the GitHub release.
4. Archive the release in Zenodo.
5. Update `CITATION.cff` with the Zenodo DOI.
6. Confirm author affiliation/ORCID metadata before submission.
7. Decide whether to submit JOSS after public-history maturity or prepare a
   SoftwareX manuscript first.

## Evidence Boundary

Phase 2-G remains the software demonstration bundle:

- 14 scenarios;
- 17 estimators;
- 238 scenario-estimator pairs;
- 7140 raw Monte Carlo records;
- 234 main-comparison pairs;
- 4 diagnostic appendix pairs.

It should not be expanded into stronger journal claims without new dedicated
evidence for timing, n=100 uncertainty, PI-GRU inclusion, or ATLAS severity
sweeps.
