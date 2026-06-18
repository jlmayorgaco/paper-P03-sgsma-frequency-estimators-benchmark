# SGSMA Slide Claim Audit

Date: 2026-05-31

Scope: `slides/slide.tex`, `slides_script.tex`, `slides/generated_results.tex`, SGSMA PDF `C:/Users/walla/Downloads/2026149998 (1).pdf`, and benchmark artifacts under `artifacts/`.

## Verdict

The deck is defensible if the evidence layers are kept separate:

| Layer | Scope | Evidence status | Safe use in talk |
|---|---:|---|---|
| SGSMA benchmark matrix | 16 estimators, 34 scenarios, 10 families, 60 MC runs | SGSMA PDF | Main benchmark claims |
| OpenFreqBench expansion | 18 estimators, 33 scenarios, 5 seeds | local artifact `artifacts/full_mc_benchmark` | Additional platform results, not paper table values |
| ATLAS full sweep | 17 estimators, 113 severity levels, 30 MC runs | `paper_grade`, `paper_claims_allowed=true` | Mechanism and severity claims |
| Dense phase-jump sweep | 18 estimators, positive jumps, n=1 | diagnostic | Mechanism preview only |

## Corrections Applied

- Replaced the early scale slide with the SGSMA benchmark matrix: 16 estimators, 34 scenarios, 10 disturbance families, 60 MC runs, 544 summaries, and 32,640 records.
- Replaced the family-winner table with Table II values from the submitted SGSMA paper.
- Reworked ATLAS metric disagreement to handle trip-risk ties correctly:
  - Trip-risk has tied zero-exposure minima in 87/113 severity levels.
  - Among 26 levels with a unique trip-risk winner, 5 differ from the RMSE winner.
  - RoCoF max differs from the RMSE winner in 53/112 unique-winner levels.
  - RoCoF RMS differs from the RMSE winner in 48/112 unique-winner levels.
- Updated CPU values to the SGSMA benchmark values where the slide discusses deployability: SOGI-FLL 0.0348 us, SOGI-PLL 0.0867 us, EKF 0.813 us, RA-EKF 1.61 us, Koopman 390 us, ESPRIT 2838 us.
- Updated ATLAS readiness macros from stale diagnostic values to the current validated ATLAS artifact: n=30, fixed policy, 17/17 canonical estimators, 0 readiness issues.
- Corrected the harmonic endpoint table: ZCD is 0.27 mHz at 1% THD but 86.37 Hz at 30% THD, not 0.012 Hz.
- Marked the dense phase-jump map as diagnostic n=1, default policy, positive sign.
- Corrected FM/interharmonic text: low-rate FM winners rotate, ESPRIT leads from 1-10 Hz, and EKF leads interharmonics through 15% before PLL leads at 20%.
- Tightened RoCoF sign claim: several methods are down-ramp sensitive; the asymmetry is not universal.

## High-Risk Claims To Say Carefully

| Claim | Safe wording |
|---|---|
| "ATLAS proves a new standard" | "ATLAS supports a reference test profile; it is not a standard." |
| "Trip-risk changes the winner in 83/113 cases" | Do not say this. Use the tie-aware statement above. |
| "Phase-jump worst case is non-monotone" | "The dense diagnostic sweep suggests non-monotonicity; the validated phase-jump sweep only qualifies the tested angles." |
| "CPU is latency" | "CPU is a software-cost proxy, not hardware latency." |
| "One estimator is best" | "Estimator choice depends on event family, metric, and compute envelope." |
| "An estimator would have prevented a field disturbance" | Do not say this. Use field events only as stress-model motivation. |

## External Source Ledger

- NERC/WECC Blue Cut Fire report: 1,178 MW PV loss on August 16, 2016; false low-frequency trips caused by distorted fault waveforms are explicitly discussed. Source: https://www.nerc.com/globalassets/our-work/reports/event-reports/1200_mw_fault_induced_solar_photovoltaic_resource_interruption_final.pdf
- NERC/WECC Canyon 2 report: October 9, 2017, two solar PV reductions of 682 MW and 937 MW; report states erroneous frequency tripping was not a cause. Source: https://www.nerc.com/globalassets/our-work/reports/event-reports/900-mw-solar-photovoltaic-resource-interruption-disturbance-report.pdf
- NERC Odessa 2022 report: about 1,711 MW IBR reduction in West Texas and note that SCADA reductions may differ from high-resolution monitoring. Source: https://www.nerc.com/globalassets/our-work/reports/white-papers/nerc_2022_odessa_disturbance_report-1.pdf
- NERC/WECC CA BESS 2023 report: abnormal BESS loss events in Southern California on March 9 and April 6, 2022. Source: https://www.nerc.com/comm/RSTC/Documents/NERC_BESS_Disturbance_Report_2023.pdf
- NERC disturbance-monitoring white paper: recommends SCADA, SER, DFR, and inverter/controller data retention; DFR is kHz point-on-wave. Source: https://www.nerc.com/globalassets/our-work/reports/white-papers/white_paper_ibr_disturbance_monitoring.pdf
- IEEE/IEC 60255-118-1:2018: defines synchrophasor, frequency, and ROCOF measurement and compliance requirements; it does not prescribe a computation method. Source: https://standards.ieee.org/ieee/60255-118-1/5724/
- NISTIR 8106: NIST PMU performance assessment program. Source: https://www.nist.gov/publications/2014-nist-assessment-phasor-measurement-unit-performance
- Kaua'i oscillation analysis: November 21, 2021, 18-20 Hz oscillation following an oil-plant trip. Source: https://arxiv.org/abs/2301.05781
- ENTSO-E Iberian blackout page: final report and factual report for the April 28, 2025 Spain/Portugal incident. Use only as observability and event-analysis motivation. Source: https://www.entsoe.eu/publications/blackout/28-april-2025-iberian-blackout/
- SGSMA 2026 official site: June 1-4, 2026, Santiago, Chile. Source: https://sgsma2026.cl/home

## Speaker Guardrails

- Say "SGSMA benchmark matrix" for 16/34/60 results.
- Say "OpenFreqBench expansion" for 18/33/n=5 local results.
- Say "ATLAS validated n=30" only for `artifacts/atlas-papergrade-missing-v1`.
- Say "diagnostic" for dense phase-jump and old subset artifacts.
- Avoid ratios like "275x" unless directly tied to a table in the current deck; it was intentionally removed.
- When challenged on CPU differences, answer: "The SGSMA benchmark CPU values and the OpenFreqBench expansion values come from different Python runs, so each should be interpreted within its own artifact."
