# Scientific Readiness

OpenFreqBench 2.0.0 is designed for graduate-level reproducible benchmarking, but
public claims must follow this checklist.

## Minimum Claim Standard

Every claim should include:

- benchmark profile: `canonical-single-phase-v1`,
- parameter policy,
- run id,
- scenario set,
- estimator set,
- Monte Carlo run count,
- base seed,
- report hash or archived report path,
- dependency versions,
- checkpoint hashes when neural estimators are used.

## Evidence Levels

Level 1: quick smoke result. Useful for debugging, not for scientific claims.

Level 2: YAML Monte Carlo result with reproducibility manifest. Acceptable for
internal comparison.

Level 3: tuned artifact replay with preregistered hypotheses and archived report.
Acceptable for paper supplement and public leaderboard.

Level 4: independently reproduced result from a fresh clone/wheel environment.
Target level for awards, competitions, and external review.

