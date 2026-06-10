---
title: "OpenFreqBench: A reproducible benchmark platform for grid frequency estimators"
tags:
  - Python
  - power systems
  - frequency estimation
  - PMU
  - benchmarking
authors:
  - name: Juan Lugo
    orcid: "TODO"
    affiliation: 1
affiliations:
  - name: "TODO: add institutional affiliation"
    index: 1
date: 10 June 2026
bibliography: paper.bib
---

# Summary

OpenFreqBench is an open-source Python benchmark platform for single-phase grid
frequency estimator evaluation. It provides a command-line workflow for running
Monte Carlo benchmark matrices, comparing canonical and custom estimators,
generating reproducible artifacts, building report tables and plots, validating
public output schemas, and archiving evidence for software and paper claims.

The current software release candidate exposes 32 single-phase scenarios, 18
canonical estimators, and a locked metric profile,
`canonical-single-phase-v1`. The platform is designed so that users can modify
estimators, scenarios, run matrices, and hypotheses while preserving canonical
metric definitions and artifact traceability.

# Statement of Need

Frequency-estimator studies often mix signal-generation assumptions, estimator
parameter policies, metric definitions, and report post-processing in ways that
make comparisons difficult to reproduce. OpenFreqBench addresses this by
separating the benchmark contract from individual experiments:

- scenarios generate voltage traces and ground-truth frequency trajectories;
- estimators implement a public `step(...)` or `step_vectorized(...)` contract;
- metrics are implemented by the platform, not redefined in YAML;
- Monte Carlo runs store seeds, parameters, source hashes, environment metadata,
  raw records, aggregate metrics, and traceability files.

This makes OpenFreqBench useful for researchers who need to compare estimator
families, document failure modes, and preserve the evidence chain from command
line to manuscript table.

# State of the Field

TODO: Add concise citations for PMU frequency estimation, IEEE/NERC-style test
conditions, benchmark reproducibility, and prior estimator-comparison work.

This section should position OpenFreqBench against ad hoc simulation scripts,
single-estimator studies, and broader power-system simulation tools. The key
claim should be software scope, not superiority of any one estimator.

# Software Design

OpenFreqBench is organized around a stable public workflow:

1. Discover the platform state with `openfreqbench doctor`, `manifest`, and
   `list`.
2. Define a benchmark matrix in YAML or use CLI smoke commands.
3. Run the matrix with fixed Monte Carlo seeds and a declared parameter policy.
4. Generate machine-readable and manuscript-facing outputs with `report build`.
5. Freeze evidence with artifact indexes, hashes, environment reports, and
   paper traceability files.

The core design elements are:

- a scenario registry for single-phase grid-event and disturbance generators;
- an estimator registry with active and experimental estimator classifications;
- a locked metric profile with accuracy, tail-error, protection, timing,
  latency, memory, startup, and invalid-output metrics;
- a validity-aware report layer that moves non-finite or post-startup invalid
  pairs into a diagnostic appendix without deleting raw data;
- public schemas for benchmark reports, manifests, run configs, and ATLAS
  readiness reports.

# Research Impact Statement

OpenFreqBench supports reproducible estimator-comparison research by giving
users a common benchmark contract and a citable artifact workflow. The Phase 2-G
demonstration run contains 14 scenarios, 17 estimators, 238
scenario-estimator pairs, and 7140 raw Monte Carlo records. It demonstrates how
the software generates main-comparison tables, diagnostic failure appendices,
confidence intervals, Pareto-style recommendations, and claim-traceability
files.

The Phase 2-G run is a demonstration bundle for the software paper. It should
not be presented as final journal-grade evidence for CPU ranking, PI-GRU
generalization, ATLAS severity sweeps, or n=100 uncertainty claims.

# AI Usage Disclosure

Generative AI tools were used to assist with code audit summaries,
documentation organization, and early manuscript drafting. The author reviewed
and edited the resulting text, executed software checks, and remains responsible
for the accuracy, integrity, and reproducibility of the software and manuscript.

# Acknowledgements

TODO: Add funding, institutional, advisor, collaborator, and compute-resource
acknowledgements.

# References
