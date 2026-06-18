# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.0.x   | Yes       |
| < 2.0   | No        |

## Reporting a Vulnerability

OpenFreqBench is designed to run user-provided custom estimator code. This is
an intentional design choice: custom estimators are Python modules that execute
locally. Do not run untrusted custom estimator files.

For vulnerabilities in the OpenFreqBench platform itself (CLI, runner, metrics,
or scenario generators), please report them privately:

1. Open a GitHub security advisory at the repository's Security tab, or
2. Contact the maintainer directly via the email listed in pyproject.toml.

We aim to acknowledge reports within 5 business days and provide an initial
assessment within 14 days.

## Scope

- Core platform code: `src/openfreqbench/`, `src/analysis/`, `src/scenarios/`
- Installed package behavior under normal use
- Schema validation and output integrity

## Out of Scope

- Custom estimator code provided by users (runs locally by design)
- Benchmark results from untrusted or modified scenarios
- Issues in third-party dependencies (please report upstream)
