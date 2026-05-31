# SGSMA Slide Visual Adjustment Plan

Review target: `slide.pdf`, 58 slides, 16:9 Beamer/SGSMA template.

Audience: power systems, PMU, protection, fault detection, and IBR experts. The deck should read as an evidence-led conference presentation: one claim per slide, one dominant proof object, short interpretation, and no paper-sized figures forced into slide format.

## Priority Scale

- P0: visual defect or credibility risk that must be fixed before presenting.
- P1: high-impact legibility or hierarchy issue; fix in the next slide pass.
- P2: professional polish; improves flow but does not block use.
- P3: optional refinement.

## Global Rules for the Next Pass

1. Keep the deck at 60 slides or fewer. Current count is 58, so only two net new slides are available unless another slide is merged or removed.
2. Every technical content slide should have one dominant object: diagram, table, heatmap, plot, or equation block. If the audience cannot identify the main object at thumbnail size, simplify the slide.
3. Do not place source lines inside the footer zone. The footer starts visually at about the lower 13 percent of the canvas.
4. Rebuild paper figures for slides when possible. A paper plot can appear in backup, but oral slides need larger labels, direct annotations, and fewer panels.
5. Keep the SGSMA chrome, but reduce internal boxes when a single visual can carry the slide.
6. Use consistent evidence tags: `CORE BENCHMARK`, `FULL MONTE-CARLO`, `ATLAS SEVERITY SWEEP`, `PAPER-GRADE SWEEP`. Do not mix long banners with long titles.
7. Prefer direct labels on plots over legends when there are fewer than six methods.
8. For heatmaps, add redundant text/shape cues, not color alone.
9. Section divider slides are visually good now; do not redesign them.
10. The strongest deck rhythm is: field motivation -> platform -> method -> core evidence -> ATLAS findings -> deployment.

## Highest-Impact Fixes

1. S04: source line is clipped by the footer. Move source text above the content block or remove it from the visible slide.
2. S16: workflow is accurate but dense. Split into a clean top-level flow plus a smaller "locked evidence contract" inset, or simplify in place.
3. S17-S18: scenario suites are still paper-grid figures. Split into four larger visual slides: standard voltage, standard frequency, composite voltage, composite frequency. This uses the two available slide slots and brings the deck to 60.
4. S27: four event-response plots are too small for oral explanation. Replace in place with two dominant plots and two small thumbnails, or merge S26/S27 into a two-slide evidence pair.
5. S43-S49: ATLAS evidence is valuable but several slides use print-sized plots. Redraw the most important plots with larger axes and direct labels; move full matrices to backup only if necessary.
6. S49: Pareto plot should be the deployment punchline. Increase point labels and annotate three zones directly: low-cost core, accurate-but-costly, unstable/outlier.
7. S56: qualification score is strong but equation-heavy. Add a small visual "score composition" bar or flow so the equation is not the only anchor.

## Slide-by-Slide Plan

| Slide | Visual status | Adjustment |
|---:|---|---|
| 1 | Strong title slide. | Keep. Optional: slightly enlarge author/affiliation line if presenting from a large room. |
| 2 | Section divider is clean. | Keep. |
| 3 | Timeline is useful; event cards are compact. | P2: enlarge event card body text by shortening each stress phrase. Keep chronological order. |
| 4 | Good concept, but source line is clipped by footer. | P0: remove visible source line or place it above the footer; reduce one text block by one line. |
| 5 | Clear research problem. | Keep. Optional: make the right "Research gap" box a little taller and less text-heavy. |
| 6 | Good operating claim structure. | Keep. Minor: tighten bullets to improve scan speed. |
| 7 | Useful but redundant with later contribution/method slides. | P2: keep if oral deck needs contribution framing; otherwise mark as optional skip. |
| 8 | Good scale summary. | Keep. Consider replacing "grid search" card with a small tuning icon/process marker. |
| 9 | Good decision logic. | P2: enlarge bottom recommendation box or remove the example sentence if too small in room projection. |
| 10 | Section divider is clean. | Keep. |
| 11 | Literature gap diagram works. | Keep. Minor: enlarge the right-side pipeline labels. |
| 12 | Comparison table is readable but compact. | P2: use stronger row emphasis for the benchmark row; reduce cell text in source categories. |
| 13 | Platform slide is strong and specific. | P1: bottom callout sits low; raise it slightly and reduce the three lower boxes by one line. |
| 14 | Section divider is clean. | Keep. |
| 15 | Method flow is readable. | Keep. Optional: make "measurement setting" and "experimental principle" equal width. |
| 16 | Workflow is accurate but visually crowded. | P1: simplify in place or split by replacing S16 with a cleaner flow and moving details into S15/S37. |
| 17 | Scenario grid has too many panels for oral use. | P1: split into two slides: standard voltage waveforms and standard frequency traces. |
| 18 | Scenario grid has too many panels for oral use. | P1: split into two slides: composite voltage waveforms and composite frequency traces. |
| 19 | Estimator taxonomy is readable. | Keep. Minor: align box widths exactly. |
| 20 | Metric slide is clean. | Keep. Optional: add small color coding that matches later RMSE/trip/CPU usage. |
| 21 | Section divider is clean. | Keep. |
| 22 | Heatmap-style table works. | Keep. Minor: enlarge table by trimming right-side prose one line. |
| 23 | Important, but mostly text/table. | P1: convert into a two-column contrast diagram: state-space/loop recovery vs window/spectral trip. |
| 24 | Good illustrative waveform. | Keep. This is a model for other mechanism slides. |
| 25 | Strong table and message. | P2: raise source line; table could be larger by reducing lower text blocks. |
| 26 | Useful plots, but not dominant enough. | P2: increase the two charts and shorten the result-reading block. |
| 27 | Four plots are paper-sized. | P1: redesign in place with two large plots plus two small thumbnails, or split only if another slide is merged. |
| 28 | Good comparison summary. | P2: convert the table into a "question -> answer -> boundary" decision ladder. |
| 29 | Good feasibility table. | Keep. Minor: make CPU class/family labels visually consistent with estimator families. |
| 30 | Pareto plot is important but labels are small. | P1: enlarge plot, direct-label key methods, reduce interpretation bullets. |
| 31 | Radar plot is useful but right text dominates. | P1: enlarge radar and reduce bullet block to three statements. |
| 32 | Compliance heatmap is good and recognizable. | P2: enlarge heatmap and move column explanation into a compact legend strip. |
| 33 | Claims slide is clear. | Keep. Optional: number the four claims to match oral transitions. |
| 34 | Section divider is clean. | Keep. |
| 35 | Good transition into expanded evidence. | Keep. Minor: make the two evidence layers visually asymmetric: MC as base, ATLAS as stress microscope. |
| 36 | Strong table; usable as a main results slide. | P2: add color highlights to RMSE leader and trip-risk leader columns. |
| 37 | Good ATLAS process slide. | Keep. Optional: turn the five boxes into a more explicit left-to-right maturity ladder. |
| 38 | Strong result slide. | P2: increase the small comparison table and use one sentence per right-side block. |
| 39 | Valuable table but text-heavy. | P1: redesign as a severity ladder by stress family, with methods-below-threshold as the visual variable. |
| 40 | Strong decomposition insight. | Keep. Minor: make AM/FM/interharmonic labels larger and color-coded. |
| 41 | Excellent diagnostic case. | Keep. Minor: table could be 10-15 percent larger; trim one sentence in the left block. |
| 42 | Good bar chart and claim. | Keep. Optional: annotate down-ramp region directly on chart. |
| 43 | Important ATLAS finding, but plot labels are small. | P1: rebuild plot with fewer visible rows or add a zoomed "worst zone" inset. |
| 44 | Strong result, but right heatmap is small. | P1: enlarge the heatmap or split table/heatmap into two stacked evidence bands. |
| 45 | Strong story; plot is small and bottom result line is long. | P1: rebuild as three stress cards plus one small heatmap; shorten result line. |
| 46 | Good theoretical/mechanism slide. | Keep. Minor: make the mechanism equation slightly more prominent. |
| 47 | Good explanatory illustration. | Keep. Optional: add one visual marker showing larger timing jitter after sag. |
| 48 | Valuable result; six subplots are too small. | P1: show only the three most diagnostic panels or rebuild as family-slope chart. |
| 49 | Core deployment plot but labels are small. | P1: enlarge labels and add zone callouts; reduce empty margins around plot. |
| 50 | Section divider is clean. | Keep. |
| 51 | Good findings slide. | P2: make the three findings visually parallel; avoid long lines in the result box. |
| 52 | Good limitations slide. | Keep. Minor: make validation limits and validation program visually balanced. |
| 53 | Recommendation map is clean but sparse. | P2: add one small "event class -> estimator family" legend or table below. |
| 54 | Qualification profile is clean. | Keep. Optional: use it as a bridge into the score slide. |
| 55 | Good standardization argument. | P2: emphasize "benchmark profile" as the new layer; reduce current-standard text. |
| 56 | Strong concept; equation dominates. | P1: add score-composition visual and reduce formula block height. |
| 57 | Good closing process diagram. | Keep. Minor: make bottom bullets shorter and more assertive. |
| 58 | Good conclusion but dense for final slide. | P2: make final result sentence larger; reduce agenda bullets or move to backup. |

## Recommended Implementation Waves

### Wave 1: Defect and projection fixes

Target: keep slide count at 58.

- Fix S04 source clipping.
- Raise or shorten low source/callout lines on S13, S25, and S45.
- Increase key labels on S30, S31, S43, S44, S48, and S49 without changing slide count.
- Apply a consistent evidence-tag style across S35-S49.

### Wave 2: Scenario-suite readability

Target: bring deck to 60 slides.

- Replace S17 with two slides: standard voltage waveforms, standard frequency traces.
- Replace S18 with two slides: composite voltage waveforms, composite frequency traces.
- Keep all four as visual explanation slides, not source-dense figures.

### Wave 3: Core evidence redesign

Target: stay at 60 by replacing in place.

- Redesign S23 as a mechanism contrast slide.
- Redesign S27 as two dominant response panels plus two thumbnails.
- Redesign S28 as a decision ladder instead of a dense comparison table.
- Redesign S30/S49 with direct labels and deployment zones.

### Wave 4: ATLAS polish

Target: keep only the highest-value ATLAS evidence in the oral deck.

- S39: convert table to severity ladder.
- S43: add zoomed inset for non-monotone phase-jump region.
- S44: make the "champions rotate" heatmap the dominant object.
- S45: rebuild ZCD paradox as three cards plus one heatmap.
- S48: replace six subplots with family-slope summary.

### Wave 5: Closing discipline

Target: sharper ending.

- Merge the messaging of S51 and S58 if a new evidence slide is needed elsewhere.
- Make S56 less formula-first: score composition, weights, and use cases should be visible before the equation.
- Use S53-S57 as deployment/standardization sequence; avoid adding new claims after S58.

## Suggested 30-Minute Oral Route

Use approximately 28-32 slides orally and keep the remaining slides available for Q&A.

- Open and motivation: S1-S6.
- Platform and gap: S10-S16.
- Method and scenarios: S17-S20 after split.
- Core paper-grade evidence: S21-S33.
- Expanded/ATLAS results: select S35, S36, S38, S39, S41, S42, S45, S46, S48, S49.
- Deployment and closing: S51-S58.

## Visual QA Checklist After Edits

1. Compile `slides/slide.tex` twice with `pdflatex`.
2. Render contact sheets at 120-140 dpi.
3. Inspect every slide at thumbnail size for dominant object and title clarity.
4. Inspect P1 slides at full resolution.
5. Confirm no source/callout text enters the footer zone.
6. Confirm slide count is 60 or fewer.
7. Copy `slides/slide.pdf` to root `slide.pdf`.
