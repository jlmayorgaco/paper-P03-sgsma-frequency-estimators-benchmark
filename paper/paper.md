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
    orcid: 0000-0002-0774-217
    affiliation: 1
affiliations:
  - name: "Independent researcher"
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

Grid frequency estimation is a foundational measurement task in power systems,
underpinning protection relays, phasor measurement units (PMUs), and
inverter-based resource (IBR) control. As grids transition toward high IBR
penetration, frequency disturbances become faster (high RoCoF), more complex
(multi-event sequences), and more frequent, raising the stakes for estimator
selection.

The estimator literature is large but fragmented: studies typically evaluate a
single method or a single family against a small set of hand-crafted test cases,
with signal-generation assumptions, parameter policies, and error metrics defined
ad hoc. When two papers report RMSE for the "same" estimator under the "same"
scenario, the numbers often differ by an order of magnitude because the
underlying benchmark contracts are incompatible--different sampling rates,
different noise models, different trim windows, and different formulas for
"RMSE." This fragmentation makes it difficult for researchers, protection
engineers, and PMU developers to answer practical questions: which estimator
should I deploy for my grid conditions, and what failure modes should I expect?

OpenFreqBench addresses this by providing a common, locked benchmark contract
that separates concerns:

- **Scenarios** generate voltage traces and ground-truth frequency trajectories
  from a fixed taxonomy of disturbance families (steps, ramps, modulation, phase
  jumps, harmonics, interharmonics, noise, ringdown, and composite IBR events).
- **Estimators** implement a public `step(z, t_s, memory)` contract
  (one sample in, one frequency estimate out), instrumented by the platform to
  measure CPU time, detect invalid outputs, and track memory use.
- **Metrics** are implemented and locked by the platform in versioned profiles,
  preventing users from silently redefining what "RMSE" or "trip risk" means.
- **Monte Carlo runs** store seeds, parameters, source hashes, environment
  metadata, raw records, and aggregate metrics, producing a traceable evidence
  chain from command line to manuscript table.

The target audience includes power-systems researchers evaluating novel
estimators, protection engineers screening candidates for IBR-dominated grids,
and PMU developers seeking reproducible comparison baselines. The platform was
validated through a pilot study presented at SGSMA 2026 with 16 estimators, 34
scenarios, and blind peer review [@openfreqbench2026].

# State of the Field

Synchrophasor and PMU evaluation is normally anchored in standard measurement
requirements and dynamic test cases rather than in a single estimator family
[@ieee2018_synchrophasor; @martin2015_synchrophasor]. The estimator literature
is broad: zero-crossing and interpolated-DFT methods remain common references
for timing and windowed approaches [@djuric2008_zero_crossing;
@grandke1983_ipdft], PLL and SOGI structures are widely used for grid
synchronization [@kaura1997_pll_distorted; @ciobotaru2006_sogi_pll],
Kalman-family estimators cover linear, extended, unscented, adaptive, and robust
variants [@kalman1960_linear_filtering; @dash1999_extended_complex_kalman;
@julier2004_unscented_filtering; @mehra1970_adaptive_kalman], and subspace or
modal methods such as Prony, ESPRIT, MUSIC, matrix pencil, and Koopman/EDMD have
separate signal-processing foundations [@hauer1991_prony_power_system;
@roy1989_esprit; @schmidt1986_music; @hua1990_matrix_pencil;
@williams2015_edmd_koopman].

Several open-source tools exist for power-system simulation and PMU data
analysis, but none provides a locked benchmark contract specifically for
single-phase frequency estimator comparison. General-purpose power-system
simulators (e.g., ANDES, OpenDSS) generate waveforms but do not define
estimator APIs, metric profiles, or reproducibility chains. Specialized PMU
calibration tools focus on compliance testing against IEEE standards rather than
comparative estimator benchmarking. Ad hoc comparison scripts, while common in
the literature, conflate signal generation, parameter policies, and metric
definitions in unreproducible ways. OpenFreqBench fills this gap by offering a
versioned, schema-validated platform where scenarios, estimators, and metrics
are independently registered but compared under a common, traceable contract.
The platform does not compete with power-system simulators or PMU calibrators;
it consumes their output (or generates synthetic test signals) and provides the
missing layer of structured, reproducible estimator comparison.

The design philosophy follows a "build, not extend" rationale: existing tools
are optimized for simulation or compliance, not for estimator benchmarking with
locked metrics, validity gates, and artifact traceability. Adding these features
to a general-purpose simulator would require a fundamental redesign of its
abstraction layers. OpenFreqBench instead provides these guarantees natively
through a minimal, focused codebase.

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

Several design decisions distinguish OpenFreqBench from general-purpose
benchmark frameworks:

**Locked metric profiles.** Metric formulas are owned by the platform and
versioned (e.g., `canonical-single-phase-v1`). Users select metrics by ID in
YAML but cannot redefine formulas. This prevents the common pitfall where two
studies report incompatible "RMSE" values because they use different trim
windows, exclude different samples, or compute the mean differently. If a
researcher needs a new metric, it must be added in code with tests and
documentation, producing a new profile version.

**Validity gate with diagnostic appendix.** Some estimator-scenario pairs
produce non-finite outputs (NaN, Inf) or excessive post-startup invalid rates.
Rather than silently dropping these pairs or letting them corrupt aggregate
statistics, the platform classifies them as "diagnostic" and moves them to a
separate appendix. Raw data is preserved, and the classification rules are
preregistered and auditable. This is methodologically important for frequency
estimation, where windowed methods legitimately return NaN during their
structural latency window, but post-startup NaN indicates estimator failure.

**Parameter policy separation.** Three policies are supported: `default` (use
each estimator's published default parameters), `explicit` (override via YAML),
and `artifact_tuned` (load per-scenario, per-estimator tuned parameters from a
prior tuning run). This separation ensures that tuning provenance is traceable
and that a paper's "default parameter" claims are not silently conflated with
"best-case tuned" performance.

**Numba JIT compilation.** Performance-critical estimator cores (e.g., Kalman
filter updates, Prony/ESPRIT eigen-decompositions) use Numba's `@njit` just-in-
time compilation to approach compiled-language speed while keeping the estimator
code readable and Python-native. This is essential because the benchmark
measures per-sample CPU time as a deployment-cost proxy, and Python overhead
would dominate for fast estimators (e.g., SOGI-FLL at ~0.035 us/sample).

**Reproducibility chain.** Every benchmark run stores its YAML config, Monte
Carlo seeds, environment metadata (Python version, dependency hashes, OS, CPU),
source-code hashes, and output artifact hashes. The `archive` command freezes
this into a ZIP bundle with an artifact index and a paper-traceability CSV that
maps every quantitative claim to a specific file and hash. This allows reviewers
and readers to verify that a paper's numbers match the referenced artifact
without re-running the full benchmark.

# Research Impact Statement

OpenFreqBench has already demonstrated research impact through its pilot
deployment at the SGSMA 2026 conference [@openfreqbench2026], where 16
estimators across 10 disturbance families were benchmarked under blind peer
review. The conference presentation validated the benchmark methodology and
generated community interest in a reusable, open-source implementation.

The current Phase 2-G demonstration run contains 14 scenarios, 17 estimators,
238 scenario-estimator pairs, and 7,140 raw Monte Carlo records. It
demonstrates the full software workflow: main-comparison tables with confidence
intervals, diagnostic failure appendices for non-finite outputs, Pareto-style
accuracy-runtime recommendations, and claim-traceability files that link every
quantitative result to an artifact path and content hash.

The platform is designed to serve as a community benchmark hub. Researchers can
contribute new estimators by implementing the `step()` contract and registering
them in YAML, without modifying the metric or scenario infrastructure. Custom
scenarios can be added following the same registration pattern. The locked
metric profile ensures that contributed estimators are evaluated under the same
definitions as canonical ones, preventing "benchmark gaming" through metric
redefinition.

The Phase 2-G run is a demonstration bundle for this software paper. It should
not be presented as final journal-grade evidence for CPU ranking, PI-GRU
generalization, ATLAS severity sweeps, or n=100 uncertainty claims. The
software's impact lies in enabling those future studies under a common,
reproducible contract.

# AI Usage Disclosure

Generative AI tools were used to assist with code audit summaries,
documentation organization, and early manuscript drafting. The author reviewed
and edited the resulting text, executed software checks, and remains responsible
for the accuracy, integrity, and reproducibility of the software and manuscript.

# Acknowledgements

No external funding or institutional grant is declared for this release-candidate
manuscript. The SGSMA presentation is acknowledged as the pilot setting that
motivated the cleanup into a reusable open-source benchmark platform.

# References
