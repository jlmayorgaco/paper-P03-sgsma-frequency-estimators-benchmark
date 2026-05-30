# SLIDES.md — SGSMA 2026 deck correction plan

> **Status (updated 2026-05-30):** Reorganized into **Motivation → Measurement gap → Benchmark platform → Method → Core evidence → Expanded/ATLAS → Deployment → Validation/conclusion**. The 6 source-verified "gold" findings now **ARE the expanded evidence in the main body**; the 3 generic ATLAS plot slides they replace (Sensitivity to disturbance severity, Rank shift when the metric changes, Harmonic-stress response) were moved to the **backup appendix**. Result: **49 main slides + 10 backup = 59 physical pages** (footer reads `N/59` throughout — this beamer's `\appendix` does NOT shrink `\inserttotalframenumber`). Expanded-benchmark scope numbers set to **16 estimators / 34 scenarios / 60 Monte Carlo runs** per the author's explicit instruction (note: this contradicts the repo, where the code shows 18/32 and the n=5/n=1 runs; "60" has no artifact source — recorded as the author's decision). Backup slides that describe the *full canonical set* / *representative ATLAS runs* still say 18 (truthful for those runs). The per-slide mapping table further down is historical; the QA criteria remain binding. Active order in `## Final deck order` below.

## Final deck order (shipped, 2026-05-30)

Main talk — 49 slides:
1 Title · 2 IBR disturbances expose a measurement gap · 3 Standard PMU streams show consequences, not always causes · 4 Research problem · 5 Operating claim · 6 Technical contributions · 7 Experiment scale · 8 Estimator selection logic · 9 Benchmark literature gap · 10 Benchmark comparison axes · 11 OpenFreqBench platform · 12 Experimental method · 13 Full benchmark workflow · 14 Scenario suite · 15 Estimator families · 16 Metrics and decision variables · 17 RA-EKF mechanism · 18 Standard scenario results · 19 Composite islanding: trip exposure under phase jump · 20 Composite IBR sequence: why it is hardest · 21 Scenario E: the leaderboard breaks · 22 Trip-risk evidence: core stress cases · 23 Event responses · 24 Result comparison · 25 Computational feasibility · 26 Deployment limits · 27 Balance profile · 28 Compliance heatmap · 29 Empirical claims · **30 Expanded stress analysis · 31 Expanded benchmark: family winners · 32 ATLAS severity analysis · 33 Reading the evidence: three levels · 34 Dynamic accuracy ranking · 35 When the textbook filter diverges · 36 RoCoF has a sign: down-ramps are harder · 37 Phase-jump severity is non-monotone · 38 No universal winner: champions rotate by regime · 39 The ZCD paradox · 40 Harmonic degradation differs by estimator family · 41 CPU–accuracy Pareto analysis** (30–41 = expanded/ATLAS, built around the gold findings) · 42 Three findings · 43 Validity limits and next steps · 44 Recommendation map · 45 Qualification profile · 46 Standardization profile · 47 IBR Event Qualification Score · 48 Toward realtime IBR estimators · 49 Validation agenda and conclusion.

Backup appendix (after `\appendix`): Backup divider · **Sensitivity to disturbance severity · Rank shift when the metric changes · Harmonic-stress response** (the 3 replaced generics) · Backup: metric registry · frequency-step stress map (n=30) · magnitude-step stress map (n=30) · noise robustness ranking · the paper-grade run (planned) · selected references.

Key moves: motivation (2–4) before operating claim/contributions; "Composite IBR sequence" (20) before "Scenario E: the leaderboard breaks" (21, renamed from "Multi-event trip-risk"); "Event responses" (23) before "Result comparison" (24); the 6 gold findings replace the 3 generic ATLAS plot slides as the expanded evidence (35–40). Renames: "IBR disturbances expose the gap"→"…expose a measurement gap"; "Standard PMU stream limits"→"Standard PMU streams show consequences, not always causes"; "Benchmark gap in prior work"→"Benchmark literature gap"; "Conclusion and validation agenda"→"Validation agenda and conclusion". Disabled `\ifbackupframes` alternates remain inert.

## Mission

Maintain the SGSMA 2026 Beamer/LaTeX presentation (`slides/slide.tex`) as a self-contained ~30-minute conference deck. The deck must **not remove substantive scientific content** and must stay visually clean and terminologically consistent. Disabled alternate frames live behind the `\ifbackupframes` switch (`\backupframesfalse`) and are intentionally not rendered.

The final deck should be able to **present itself**: a technically informed reader should understand the motivation, benchmark design, estimator evidence, RA-EKF result, limitations, and deployment recommendation even without listening to the speaker.

Central message to preserve:

> Frequency-estimator qualification in low-inertia IBR grids is conditional on event class, metric target, and compute envelope. There is no universal leaderboard; OpenFreqBench makes this comparison reproducible.

## Non-negotiable rules

1. ~~Exported deck must have **45 slides exactly**.~~ **Superseded:** the deck is the expanded ~30-min form (currently 59 pages incl. backup appendix). Keep the count stable across edits and verify with `pdfinfo` after every build; do not let edits silently add or drop pages.
2. Keep all substantive content: scientific claims, estimator names, scenario definitions, metric definitions, numeric results, validity limits, ATLAS/expanded-benchmark evidence, recommendation map, and source footnotes.
3. It is acceptable to remove standalone section-divider slides, merge redundant slides, shorten repeated prose, and redesign dense visuals.
4. Do not bury important content only in speaker notes. The slide itself must contain the claim, evidence, and takeaway.
5. Every non-title slide must have a visible **takeaway sentence** or equivalent result box.
6. Use claim-oriented slide titles where possible, not generic labels. Example: use “Scenario E breaks a single leaderboard” instead of only “Multi-event trip-risk”.
7. Do not use tiny text to force content to fit. Avoid main-body text below ~11 pt; tables may use smaller text only if still readable in a projected 16:9 deck.
8. No arrows, labels, callouts, legends, or source text may overlap plotted data, diagram boxes, or each other.
9. Keep terminology and capitalization consistent throughout the deck.
10. Build the final PDF and inspect rendered pages before declaring the work complete.

## Canonical terminology and spell-check list

Use these spellings exactly unless quoting a source title:

- **SGSMA 2026**
- **OpenFreqBench**; do not use OpenFrequency, OpenFreqeuen, OpenFreq, or OpenFrequencyBench.
- **frequency**, **benchmark**, **disturbance**, **estimator**, **presentation**.
- **low-inertia IBR grids**.
- **inverter-based resources (IBRs)** on first use if space allows; **IBR** afterward.
- **BESS**, **PV**, **PMU**, **DFR**, **HIL**, **ATLAS**.
- **RoCoF** in normal prose; **ROCOF** only when matching standards/source language.
- **Ttrip** in prose and `T_{\mathrm{trip}}` in math.
- **trip-risk exposure**, **false-trip exposure**, **settling time**, **CPU cost**, **structural latency**.
- **RA-EKF**, **EKF**, **UKF**, **SRF-PLL**, **SOGI-FLL**, **SOGI-PLL**, **IpDFT**, **TFT**, **VFF-RLS**, **Koopman (RK-DPMU)**, **PI-GRU**, **ZCD**, **ESPRIT**, **Prony**, **LKF**, **TKEO**.
- **phase jump**, **harmonic**, **interharmonic**, **ringdown**, **impulsive noise**, **multi-event**, **islanding**.
- Units: **Hz**, **Hz/s**, **ms**, **s**, **µs/sample**, **1 MHz**, **10 kHz**.
- Use **Monte Carlo** consistently, not MC in titles unless space is very tight.
- Use **hardware-in-the-loop**, **time-synchronized**, **non-time-synchronized**, **sub-millisecond**.

## Evidence scopes that must be clear

The deck currently mixes core, expanded, and ATLAS evidence. Keep all three, but label them clearly so the audience does not think the numbers conflict.

- **Core benchmark:** 9 primary estimators + 2 legacy baselines; 5 scenarios A–E; main RA-EKF/EKF/UKF/SRF-PLL/IpDFT/TFT/Koopman/SOGI-FLL evidence.
- **Expanded benchmark:** **18 estimators; 32 scenarios; pilot Monte Carlo `n = 5`, per-scenario tuned** (paper-grade `n ≥ 30` run still pending). The earlier "16 estimators / 34 scenarios / 60 runs" wording was incorrect and has been removed from the deck — verified against `benchmark_definition.py` and `full_mc_benchmark.py`.
- **ATLAS severity sweep:** pilot run, `n = 1`, default controller policy unless otherwise stated. Treat ATLAS as diagnostic evidence, not as final universal ranking. Exception: the `magnitude_step` backup sweep is paper-grade (`n = 30`, fixed policy, both signs) and carries its own green badge.

Important consistency check:

- The core CPU slide lists PI-GRU at approximately **136,000 µs/sample**, while the ATLAS dynamic ranking lists PI-GRU at about **3,509 µs/sample**. If both are correct, label the benchmark contexts clearly. If one is stale, correct the stale value and update the interpretation.

## 45-slide target blueprint (historical)

> **Historical:** this map describes the abandoned 45-slide cut. The shipped deck is the expanded ~59-page form. Use this table only to trace which claim/title each slide descends from, not as a target structure.

Delete the standalone section-divider slides and merge only where indicated. The final exported PDF should follow this 45-slide map.

| Target slide | Content / new claim title | Source slide(s) | Required action |
|---:|---|---:|---|
| 1 | Title + conference identity | 1 | Keep, simplify if crowded. |
| 2 | IBR field disturbances make estimator errors operational | 6 | Compress the event table; preserve all event examples and benchmark implications. |
| 3 | PMU streams can show consequences but miss subcycle causes | 7 | Keep three limitations; clean mini Nyquist diagram. |
| 4 | Frequency estimation is a multi-objective measurement problem | 2 | Keep operating claim; tighten bullets. |
| 5 | Experiment scale: core benchmark and observed pattern | 3 | Clarify “9 primary estimators + 2 legacy baselines”. Add separate reference to expanded benchmark only if needed. |
| 6 | Contributions and out-of-scope boundaries | 4 | Rename “Scope” to “Out of scope” or “Not claimed here”. |
| 7 | Estimator choice follows event, metric, cost, and evidence | 5 | Keep diagram; verify arrow alignment. |
| 8 | Research problem: event-aware frequency metrics | 9 | Remove preceding section divider. Make the research gap a strong right-side takeaway. |
| 9 | Prior work leaves an estimator-level benchmark gap | 11 + 12 | Merge. Preserve standards/metrology/platforms/datasets/algorithm-study categories and their gaps. |
| 10 | Benchmark axes: truth, stress, trip-risk, CPU, artifacts | 13 | Keep matrix; add one-sentence interpretation. |
| 11 | OpenFreqBench makes estimator claims repeatable | 14 | Fix diagram arrow collision; define OpenFreqBench clearly. |
| 12 | Experimental method: same truth, stream, tuning, metrics | 16 | Clean crossed arrows and clarify local relay-class setting. |
| 13 | RA-EKF adds explicit RoCoF state and event-aware gating | 17 | Keep equations but add plain-English explanation. |
| 14 | Metrics separate average error from protection exposure | 18 | Define `T_{\mathrm{trip}}` as a proxy, not a relay command. |
| 15 | Scenario suite moves from clean cases to composite IBR stress | 19 | Redesign: waveforms are too small; use five scenario cards plus readable thumbnails. |
| 16 | Estimator families occupy different latency/stress positions | 20 | Keep families; define legacy baselines as references. |
| 17 | Full benchmark workflow produces machine-readable evidence | 21 | Clean workflow; add step numbers; reduce tiny text. |
| 18 | Standard-style cases already select different winners | 23 | Keep key numeric result: RA-EKF ramp RMSE = 0.0113 Hz. |
| 19 | Scenario D: RA-EKF reduces EKF false-trip exposure | 24 | Keep 165 ms → 0.6 ms and note SRF-PLL is competitive in this case. |
| 20 | Scenario E: no single scalar ranking is defensible | 25 | Keep table; highlight RMSE/Ttrip/CPU trade-off. |
| 21 | Trip-risk is an event metric, not average error | 26 | Clean red annotation; improve bar-chart labels. |
| 22 | Winners depend on the question asked | 27 | Keep table; emphasize boundary column. |
| 23 | CPU cost changes the accuracy ranking | 28 | Clarify Python timing as software-cost proxy. Resolve PI-GRU context. |
| 24 | Event responses explain why metrics disagree | 29 | Move legends outside plots; remove leftover “(g)” label. |
| 25 | Deployment risk combines RMSE, trip exposure, settling, delay | 30 | Enlarge plots; fix label overlaps around PI-GRU/cost/risk. |
| 26 | Balance profile shows conflicting deployment axes | 31 | Enlarge radar; state normalization direction. |
| 27 | Compliance-style passes do not guarantee IBR readiness | 32 | Define pass/fail threshold and `F*`. |
| 28 | Empirical claims: protection, compliance, metrics, deployability | 33 | Keep as synthesis slide; remove repetition already covered. |
| 29 | Composite IBR sequence stacks multiple failure mechanisms | 34 | Keep stress components; link to ATLAS. |
| 30 | Expanded benchmark tests whether the pattern scales | 35 | Keep core vs expanded comparison; label evidence scope. |
| 31 | Expanded families confirm ranking depends on event family | 36 | Keep table; ensure “60 Monte Carlo runs” visible. |
| 32 | ATLAS maps severity, sign, policy, and validation gates | 37 | Keep pilot limitation badge: `n = 1`. |
| 33 | Dynamic accuracy is not the same as deployability | 38 | Make chart labels readable; keep pilot badge. |
| 34 | Severity curves reveal estimator-specific failure modes | 39 | Keep two curves; increase legend readability. |
| 35 | Metric choice changes the leaderboard | 40 | Keep RMSE vs RoCoF rank-shift chart. |
| 36 | CPU–accuracy Pareto separates plausible and costly methods | 41 | Keep Pareto; avoid overlapping labels. |
| 37 | Harmonic stress favors low baseline error and robustness | 42 | Keep harmonic plot; make log-axis readable. |
| 38 | Three findings summarize the benchmark evidence | 44 | Keep exactly three findings. |
| 39 | Validity limits define the next validation path | 45 | Keep limits and validation program. |
| 40 | Recommendation map is conditional, not universal | 46 | Keep four deployment targets. |
| 41 | Qualification profile turns rankings into test profiles | 48 | Fix text and arrow overlaps. |
| 42 | Standardization profile: what the benchmark adds | 49 | Preserve canonical events, locked metrics, and task profiles. |
| 43 | IBR Event Qualification Score states objective weights explicitly | 50 | Keep formula; define weight/profile examples. |
| 44 | Realtime IBR estimators require diagnosis, design, qualification | 51 | Fix curved arrow overlap and duplicated “atlas” text. |
| 45 | Validation agenda + conclusion | 52 + 53 | Merge. End with event + metric + compute envelope message. |

Removed as standalone slides: current slides **8, 10, 15, 22, 43, 47**. Their section names can be kept as small running labels or agenda breadcrumbs if desired.

## Visual and diagram QA tickets

### VIS-01 — Fix OpenFreqBench pipeline diagram

Current source slide 14 has a crowded pipeline. The arrow from “canonical scenario suite” into “locked metric profile” visually collides with the metric box border. Rebuild with equal spacing and connector shortening so arrowheads stop outside the boxes.

Acceptance criteria:

- No arrowhead touches or enters a box.
- Metric profile is readable in one line or two balanced lines.
- The three bottom cards align and have consistent widths.

### VIS-02 — Clean experimental-method flow

Current source slide 16 has a tuning arrow that crosses the main flow. Redesign the grid search as a side loop feeding estimator execution, not as a crossing connector.

Acceptance criteria:

- Main flow reads left-to-right: truth → stream → estimators → metrics → dashboard.
- Tuning policy is visibly a loop or side input.
- No diagonal arrow crosses through text.

### VIS-03 — Redesign scenario-suite slide

Current source slide 19 has many tiny waveform panels. Make the slide self-explaining by using five cards: A Step, B Ramp, C Modulation, D Islanding phase jump + harmonics, E Composite IBR multi-event. Use only readable thumbnail traces.

Acceptance criteria:

- Scenario names and stress mechanisms readable at normal presentation zoom.
- D and E visually emphasized as the IBR stress cases.
- Keep the message that RMSE, `T_{\mathrm{trip}}`, and CPU require joint reading.

### VIS-04 — Improve full benchmark workflow

Current source slide 21 is clean structurally but contains small text and a large dashed bottom loop. Add step numbers and reduce the visual weight of the loop.

Acceptance criteria:

- Seven main steps remain visible.
- Locked manifest, fixed I/O, aggregation audit, and qualification profile remain present.
- Bottom feedback loop does not dominate the slide.

### VIS-05 — Clean trip-risk bar charts

Current source slide 26 has a red arrow/label overlapping the Scenario D plot. Replace with an external callout or bracket.

Acceptance criteria:

- 165 ms → 0.6 ms and 275× reduction are readable.
- No callout overlaps bars or axis labels.
- Scenario D and Scenario E charts have consistent axis label sizes.

### VIS-06 — Clean event-response plots

Current source slide 29 has legends and annotations covering plotted content. Move legends outside the plot areas or into a shared legend. Remove the stray “(g)” label at the bottom left.

Acceptance criteria:

- Four panels remain readable.
- Legends do not cover key transients.
- Annotations explain phase jump, ramp lag, recovery, and unstable methods without clutter.

### VIS-07 — Clean deployment dashboard

Current source slide 30 has small plots and overlapping labels in the cost-risk chart. Enlarge the plots or simplify labels.

Acceptance criteria:

- Cost vs risk chart is readable.
- PI-GRU / costly-risky label does not overlap plot title or data.
- The right-side bullets are shortened to make room for larger visuals.

### VIS-08 — Improve balance-profile radar

Current source slide 31 should define radar-axis direction and normalization.

Acceptance criteria:

- Radar chart is larger.
- Add phrase: “larger is better after normalization” or equivalent.
- Legend is readable and not crowded.

### VIS-09 — Define compliance heatmap symbols

Current source slide 32 uses P, F, and F* without enough visible criteria.

Acceptance criteria:

- Define P, F, and F* on the slide.
- State pass/fail basis: declared thresholds for RMSE, `T_{\mathrm{trip}}`, settling, and validity.
- Keep the takeaway that Step/Ramp/Mod success does not imply islanding/multi-event readiness.

### VIS-10 — Fix qualification-profile diagram

Current source slide 48 has text labels overlapping arrows and boxes, especially near “ATLAS stress sweeps” and “reproducible reporting”. Rebuild or simplify the diagram.

Acceptance criteria:

- No label overlaps a connector or box.
- The two rows read as separate but related pipelines.
- Final concept “Dynamic estimator benchmark profile” is visually emphasized.

### VIS-11 — Fix realtime-estimator loop

Current source slide 51 has a curved arrow crossing the ATLAS box and text, and the word “atlas” appears redundantly inside the first box.

Acceptance criteria:

- Feedback arrow runs below the boxes or behind them with clear padding.
- Remove duplicated “atlas”.
- Keep field replay + HIL as a validation feedback path.

### VIS-12 — ATLAS plot hygiene

For source slides 38–42, ensure all ATLAS charts include a visible evidence-scope badge: “ATLAS pilot, n = 1, default policy”. Improve label placement on log-scale plots.

Acceptance criteria:

- Chart titles, axes, legends, and key labels are readable.
- No labels overlap densely.
- Pilot status is visible on every ATLAS plot slide.

## Content tickets

### CNT-01 — Make the deck self-exposing

For every slide except the title, add or preserve a one-sentence interpretation box. Good formats:

- **Takeaway:** ...
- **Result:** ...
- **Reading:** ...
- **Deployment implication:** ...

Acceptance criteria:

- A reader can identify the slide’s point in 5 seconds.
- No slide is only a table, only equations, or only a plot without interpretation.

### CNT-02 — Clarify RA-EKF claim boundaries

Do not present RA-EKF as a universal winner. Preserve the stronger, defensible claim:

> RA-EKF improves dynamic tracking and sharply reduces EKF-like false-trip exposure in the phase-jump stress case, while remaining computationally feasible. It is not the universal winner across all metrics and events.

Acceptance criteria:

- Scenario D slide notes SRF-PLL is also strong in that specific phase-jump case.
- Scenario E slide says the leaderboard breaks because RMSE, `T_{\mathrm{trip}}`, and CPU select different methods.

### CNT-03 — Clarify `T_{\mathrm{trip}}`

`T_{\mathrm{trip}}` must be described as a protection-oriented risk proxy, not as an actual relay trip command.

Acceptance criteria:

- Metrics slide includes this wording.
- At least one results slide reminds the reader that trip-risk is an event metric, not average error.

### CNT-04 — Clarify CPU/latency language

CPU timing must be described as a software-cost proxy, not a final hardware-latency claim.

Acceptance criteria:

- Computational feasibility slide includes “software-cost proxy”.
- Validity slide includes “hardware latency requires platform timing”.
- PI-GRU timing context is resolved or explicitly labeled.

### CNT-05 — Strengthen OpenFreqBench definition

OpenFreqBench should be introduced early as the reproducible benchmark layer.

Acceptance criteria:

- The first OpenFreqBench slide defines: estimator I/O contract, canonical scenario suite, locked metric profile, CSV/JSON artifacts, plots/rank tables, standardized comparison.
- Later slides should not redefine it inconsistently.

### CNT-06 — Keep core/expanded/ATLAS evidence separate

Acceptance criteria:

- Core result slides use a “CORE BENCHMARK” or equivalent cue when useful.
- Expanded slides use “EXPANDED BENCHMARK: 16 estimators, 34 scenarios, 60 Monte Carlo runs per scenario”.
- ATLAS slides use “ATLAS SEVERITY SWEEP: pilot run, n = 1, default controller policy”.

### CNT-07 — Improve closing message

Merge current slides 52 and 53 into one strong final slide.

Final slide must include:

- Near-term validation: fixed-policy ATLAS sweeps, Monte Carlo replication, rank stability, qualification profiles.
- Longer-term validation: three-phase/unbalance/control interactions, field replay, HIL.
- Final conclusion: estimator choice depends on event class, metric target, and compute envelope.

## Build and QA workflow

1. Inspect repository structure and locate the Beamer source file(s), figures, tables, and build script.
2. Save a backup branch or commit before editing.
3. Implement the 45-slide blueprint.
4. Compile with the existing project command. If absent, use `latexmk -pdf` or the local project convention.
5. Verify exact slide count:

```bash
pdfinfo <final_deck>.pdf | grep Pages
```

6. Extract text and run terminology checks:

```bash
pdftotext <final_deck>.pdf - | grep -niE "OpenFrequency|OpenFreqeuen|OpenFreqeu|ROCOF|µs|Ttrip|F\*"
```

Review matches manually. `ROCOF` may be valid in standards references; otherwise prefer `RoCoF`.

7. Render pages to PNG for visual QA:

```bash
mkdir -p build/slide_check
pdftoppm -png -r 160 <final_deck>.pdf build/slide_check/slide
```

8. Manually inspect all 45 rendered pages for overlaps, tiny text, legends covering data, clipped logos, and incorrect page numbers.
9. Check that page footer says `x/45`, not `x/53`.
10. Deliver final PDF plus a concise change report: slide count, merged slides, removed divider slides, visual issues fixed, terminology corrections, and any unresolved assumptions.

## Done definition

The task is done only when:

- Final PDF has exactly 45 slides.
- No substantive content from the 53-slide draft has been lost.
- The deck reads coherently without narration.
- Diagrams and plots have no overlapping arrows, labels, or legends.
- OpenFreqBench and all technical terms are spelled consistently.
- Core, expanded, and ATLAS evidence are visibly distinguished.
- RA-EKF is presented with defensible boundaries, not as a universal winner.
- A final build/test report is produced.
