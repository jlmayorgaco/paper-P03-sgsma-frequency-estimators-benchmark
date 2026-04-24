# Writing And AI-Pattern Audit For `paper/`

This file is an expanded rewrite of the earlier pattern log. It inventories the
material writing problems I found in the current manuscript, using multiple
writing-review lenses rather than a single "AI detector" lens.

This pass is meant to be exhaustive for the current draft as it exists now.

## Skill lenses used in this pass

Active in this session:

- `academic-paper`
- `academic-paper-reviewer`

Applied manually from local `SKILL.md` / reference files after installation
(these will become first-class session skills after you restart Codex):

- `research-paper-writing`
- `sciwrite`
- `avoid-ai-writing`
- `paper-write`

Primary reference files and checklists used:

- `academic-paper/references/writing_quality_check.md`
- `academic-paper/references/writing_judgment_framework.md`
- `research-paper-writing/references/paper-review.md`
- `research-paper-writing/references/does-my-writing-flow-source.md`
- `sciwrite/SKILL.md`
- `avoid-ai-writing/SKILL.md`
- `paper-write/SKILL.md`

## Scope reviewed

- `paper/Config/abstract.tex`
- `paper/Sections/C1_Introduction/main.tex`
- `paper/Sections/C2_Related_Work/main.tex`
- `paper/Sections/C3_Methods/main.tex`
- `paper/Sections/C4_Simulation_Results/main.tex`
- `paper/Sections/C5_Conclusions/main.tex`

## How to read this file

- `P0` = high-priority rewrite target. These hurt clarity, scientific tone, or
  reviewer confidence.
- `P1` = strong pattern issue. These make the prose feel templated, AI-shaped,
  or over-smoothed.
- `P2` = secondary polish issue. These are worth fixing after the high-impact
  problems.

Important boundary:

- "AI-sounding" does not mean false.
- Many individual patterns below also appear in human drafts.
- The problem is accumulation. In this manuscript, the same rhetorical moves
  repeat across sections until the prose starts to feel generated.

## Priority summary

If you only fix a few things first, fix these:

1. Front-load one concrete result in the Abstract and Introduction.
2. Rewrite Related Work so it builds a gap, not just a family catalog.
3. Reduce label-heavy, snippet-like Results prose and replace it with fuller
   interpretive paragraphs.
4. Remove benchmark-branding repetition (`stress-oriented benchmark`,
   `performance ceiling`, `selection guidance`, repeated standard-vs-IBR pitch).
5. Tighten terminology so the same disturbance classes are named consistently.
6. Cut nominalized, management-like prose and replace it with direct technical
   statements.
7. Reduce semicolon / dash / parenthetical overload.
8. Rewrite the Conclusion so it lands one strong field-specific takeaway,
   instead of closing on a polished three-point template.

## Findings

## 1. [P0] The paper front-loads benchmark framing instead of its strongest result

Why it matters:

- `paper-write` and `research-paper-writing` both push the same rule: the reader
  should see the paper's strongest result early.
- Here, the top-level sections spend more space branding the benchmark than
  showing what the benchmark actually found.

Evidence:

- `paper/Config/abstract.tex:2-19`
- `paper/Sections/C1_Introduction/main.tex:28-41`
- Strongest numeric result only arrives later in
  `paper/Sections/C4_Simulation_Results/main.tex:13-15`

Why it reads as AI-shaped:

- Models often produce a polished problem statement and contribution frame
  before they commit to one sharp empirical takeaway.
- The draft sounds "complete" early, but not yet "earned."

## 2. [P0] The Abstract uses synthesis language without one hard quantitative anchor

Why it matters:

- The Abstract says what happened, but not with one memorable number.
- Reviewer-facing abstracts benefit from one hard anchor: best result,
  dominant failure case, or largest contrast.

Evidence:

- `paper/Config/abstract.tex:11-19`

Examples:

- `Several estimators ... degrade substantially`
- `no single family dominates`
- `data-driven approaches incur higher computational cost`

What is missing:

- A single concrete result such as the RA-EKF advantage in one scenario or the
  trip-risk contrast in Scenario E.

## 3. [P0] Related Work is taxonomy-first and gap-last

Why it matters:

- The section acts as a compact survey of families, but it does not build
  pressure toward the exact gap this paper closes until the end.
- `research-paper-writing` explicitly warns against paper-by-paper or family-by-family
  catalog prose that never turns into a sharp delta.

Evidence:

- `paper/Sections/C2_Related_Work/main.tex:4-16`

Why it feels weak:

- Every sentence identifies a family, gives one strength or weakness, then moves on.
- The final sentence names the gap, but the paragraph never really builds to it.

## 4. [P0] Results prose is snippet-driven instead of paragraph-driven

Why it matters:

- Several Results blocks read like compressed figure notes rather than
  analytic prose.
- That makes the section skimmable, but also shallow and slightly machine-made.

Evidence:

- `paper/Sections/C4_Simulation_Results/main.tex:10-15`
- `paper/Sections/C4_Simulation_Results/main.tex:58-67`
- `paper/Sections/C4_Simulation_Results/main.tex:159-168`

Signals:

- bold inline labels
- two-sentence result snippets
- compressed winner sentences
- little interpretive transition between findings

## 5. [P0] Benchmark-branding sentences are repeated across Abstract, Introduction, Methods, and Conclusion

Why it matters:

- The same sales pitch is restated multiple times with light paraphrase.
- This is the strongest AI-writing signal in the draft because it creates a
  branded, rehearsed voice rather than a section-specific voice.

Evidence:

- `paper/Config/abstract.tex:2-19`
- `paper/Sections/C1_Introduction/main.tex:28-40`
- `paper/Sections/C3_Methods/main.tex:13-14`
- `paper/Sections/C3_Methods/main.tex:88-90`
- `paper/Sections/C5_Conclusions/main.tex:4-9`

Recurring payload:

- twelve estimators
- five families
- five scenarios
- standard-vs-IBR split
- scenario-wise tuning / performance ceiling

## 6. [P0] The Conclusion lands as a polished template, not as a field-specific ending

Why it matters:

- The ending announces `three principal observations`, delivers exactly three
  balanced bullets, then closes with a generic future-work and reproducibility
  bundle.
- That is readable, but it is also one of the most recognizable "LLM conference
  conclusion" shapes.

Evidence:

- `paper/Sections/C5_Conclusions/main.tex:7-39`
- `paper/Sections/C5_Conclusions/main.tex:41-53`

Why it reads as AI-shaped:

- exact three-part packaging
- equally weighted bullets
- clean cautionary finish
- no final sentence with discipline-specific sting

## 7. [P0] Terminology for the key disturbance classes drifts across sections

Why it matters:

- `sciwrite` pushes the Banana Rule: a concept should keep the same name unless
  there is a reason to rename it.
- Here, the draft keeps renaming closely related disturbance categories, which
  weakens reader confidence.

Evidence:

- `paper/Config/abstract.tex:10-19`
- `paper/Sections/C3_Methods/main.tex:63-80`
- `paper/Sections/C4_Simulation_Results/main.tex:56-67`
- `paper/Sections/C5_Conclusions/main.tex:33-51`

Drifting labels:

- `composite IBR-specific disturbances`
- `Composite Islanding`
- `IBR Multi-Event`
- `composite multi-disturbance scenarios`
- `Composite Phase-Step with Harmonic Distortion`

## 8. [P0] Major synthesis claims are often unanchored in the top-level prose

Why it matters:

- `research-paper-writing` treats claim-evidence alignment as a hard constraint,
  especially in Abstract and Introduction.
- This draft often states the synthesis before naming the evidence.

Evidence:

- `paper/Config/abstract.tex:11-18`
- `paper/Sections/C1_Introduction/main.tex:28-40`
- `paper/Sections/C5_Conclusions/main.tex:33-53`

Examples:

- `Several estimators ... degrade substantially`
- `no single family dominates`
- `data-driven approaches ... did not outperform scenario-optimized classical methods`

These may be true, but the prose often delivers them as polished verdicts first
and evidence second.

## 9. [P1] The standard-vs-IBR scenario split is repeated as a fixed formula

Why it matters:

- The `(A)-(C)` vs `(D)-(E)` split keeps returning in nearly the same sentence.
- Reusing a neat binary frame is a classic AI habit: once a model finds a clean
  explanation, it keeps replaying it.

Evidence:

- `paper/Config/abstract.tex:9-10`
- `paper/Sections/C1_Introduction/main.tex:30-35`
- `paper/Sections/C3_Methods/main.tex:79-80`
- `paper/Sections/C3_Methods/main.tex:151-158`

## 10. [P1] The "latency-robustness trade-off" slogan is overworked

Why it matters:

- The phrase is good.
- The manuscript leans on it so often that it starts to sound like branding
  instead of analysis.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:19`
- `paper/Sections/C2_Related_Work/main.tex:4`
- `paper/Sections/C5_Conclusions/main.tex:13-17`

## 11. [P1] Family roll-call prose is reused across sections

Why it matters:

- Multiple sections walk through estimator families in the same order with the
  same sentence template.
- This makes the paper sound assembled from reusable blocks.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:19-26`
- `paper/Sections/C2_Related_Work/main.tex:8-14`
- `paper/Sections/C3_Methods/main.tex:15-31`
- `paper/Sections/C4_Simulation_Results/main.tex:159-168`
- `paper/Sections/C5_Conclusions/main.tex:13-31`

Typical template:

- `Window-based methods ...`
- `Loop-based methods ...`
- `Model-based filters ...`
- `Data-driven approaches ...`

## 12. [P1] Contribution-list boilerplate is doing too much of the Introduction's work

Why it matters:

- `This paper presents ... The main contributions are ...` is standard.
- In this draft, the block is too polished and symmetric, and it substitutes for
  a more forceful story move.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:28-40`

Why it feels generic:

- three neat list items
- repeated benchmark metadata
- no strongest empirical result preview

## 13. [P1] The draft overuses contrastive symmetry

Why it matters:

- The prose repeatedly leans on `X but Y`, `while`, `whereas`, and balanced
  family-vs-family contrasts.
- This is readable once or twice. Repeated across sections, it becomes a tic.

Evidence:

- `paper/Config/abstract.tex:12-15`
- `paper/Sections/C1_Introduction/main.tex:20-26`
- `paper/Sections/C5_Conclusions/main.tex:14-31`

Repeated moves:

- `no single family dominates`
- `complementary strengths`
- `X ... but Y ...`
- `whereas ... at the cost of ...`

## 14. [P1] Semicolon-heavy claim compression makes the prose feel pre-packed

Why it matters:

- `academic-paper` and `avoid-ai-writing` both flag semicolon chaining as a
  common machine pattern.
- The manuscript often compresses claim, caveat, and consequence into one long
  sentence instead of allowing ideas to breathe.

Evidence:

- `paper/Config/abstract.tex:9-18`
- `paper/Sections/C1_Introduction/main.tex:30-39`
- `paper/Sections/C2_Related_Work/main.tex:5-7`
- `paper/Sections/C3_Methods/main.tex:21-24`
- `paper/Sections/C3_Methods/main.tex:90-103`
- `paper/Sections/C4_Simulation_Results/main.tex:58-67`
- `paper/Sections/C4_Simulation_Results/main.tex:74-85`
- `paper/Sections/C5_Conclusions/main.tex:41-52`

## 15. [P1] Dash-driven compression and parenthetical interruption are overused

Why it matters:

- The draft uses em-dash style interruptions to inject caveats, ratios, and
  clarifications.
- This is one of the most explicit `avoid-ai-writing` signals.

Evidence:

- `paper/Sections/C2_Related_Work/main.tex:6-7`
- `paper/Sections/C2_Related_Work/main.tex:44-47`
- `paper/Sections/C4_Simulation_Results/main.tex:14-15`
- `paper/Sections/C4_Simulation_Results/main.tex:60`
- `paper/Sections/C4_Simulation_Results/main.tex:67`
- `paper/Sections/C4_Simulation_Results/main.tex:187-190`
- `paper/Sections/C5_Conclusions/main.tex:49-50`

Effect:

- the voice sounds over-edited
- the sentence flow keeps getting interrupted by afterthoughts

## 16. [P1] Inline-header writing is overused in Methods, Results, and Conclusion

Why it matters:

- `avoid-ai-writing` flags inline-header lists and repeated bold headers because
  they often indicate generated scaffolding.
- This draft repeatedly writes in `Label:` mode.

Evidence:

- `paper/Sections/C3_Methods/main.tex:15-31`
- `paper/Sections/C3_Methods/main.tex:59-76`
- `paper/Sections/C4_Simulation_Results/main.tex:11-13`
- `paper/Sections/C4_Simulation_Results/main.tex:159-168`
- `paper/Sections/C5_Conclusions/main.tex:13-33`

Common shapes:

- `Loop-based:`
- `Window-based:`
- `Steady-state:`
- `Ramp:`
- `Latency-Robustness Trade-off:`

## 17. [P1] The Results section relies on winner-sentence templates

Why it matters:

- Multiple result sentences share the same optimized shape:
  method -> number -> multiplier -> short justification.
- This creates a very model-like cadence.

Evidence:

- `paper/Sections/C4_Simulation_Results/main.tex:14-15`
- `paper/Sections/C4_Simulation_Results/main.tex:58-60`
- `paper/Sections/C4_Simulation_Results/main.tex:67`
- `paper/Sections/C4_Simulation_Results/main.tex:176-182`
- `paper/Sections/C5_Conclusions/main.tex:23-28`

Typical skeleton:

- `The RA-EKF achieves ...`
- `... x lower than ...`
- `... via explicit ...`

## 18. [P1] Managerial or productized phrasing leaks into the scientific prose

Why it matters:

- Several phrases sound more like benchmark positioning copy than like
  engineering prose.
- This weakens the manuscript's scientific texture.

Evidence:

- `paper/Config/abstract.tex:6-7` -> `performance ceiling`
- `paper/Sections/C1_Introduction/main.tex:39-40` -> `selection guidance`
- `paper/Sections/C3_Methods/main.tex:88-90` -> `performance ceiling`
- `paper/Sections/C4_Simulation_Results/main.tex:110-118` -> `viable DSP/FPGA candidate`
- `paper/Sections/C4_Simulation_Results/main.tex:174-200` -> `Comprehensive Performance Dashboard`

## 19. [P1] Weak verbs and nominalized phrasing flatten otherwise technical sentences

Why it matters:

- `sciwrite` treats this as a core clarity problem.
- The draft often chooses `provide`, `offer`, `yield`, `show`, `achieve`,
  `providing`, and other weak-verb-plus-noun structures where a sharper verb
  would carry more force.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:22-25`
- `paper/Sections/C1_Introduction/main.tex:39-40`
- `paper/Sections/C2_Related_Work/main.tex:8-14`
- `paper/Sections/C4_Simulation_Results/main.tex:110-118`
- `paper/Sections/C5_Conclusions/main.tex:7-9`
- `paper/Sections/C5_Conclusions/main.tex:41-44`

Examples:

- `provide harmonic rejection`
- `offer improved tracking`
- `offer potential accuracy gains`
- `providing selection guidance`
- `yields three principal observations`

## 20. [P1] Caveat stacking has become a style, not just a necessity

Why it matters:

- Scientific caution is good.
- Repeated `visualization only`, `not certification`, `relative comparison only`,
  `Python baseline only`, `outside scope`, `future work` caveats start to sound
  machine-inserted when they appear this often.

Evidence:

- `paper/Sections/C3_Methods/main.tex:99-109`
- `paper/Sections/C4_Simulation_Results/main.tex:61-63`
- `paper/Sections/C4_Simulation_Results/main.tex:74-85`
- `paper/Sections/C4_Simulation_Results/main.tex:111-118`
- `paper/Sections/C4_Simulation_Results/main.tex:186-197`
- `paper/Sections/C5_Conclusions/main.tex:45-53`

Why it feels AI-shaped:

- the caveats are correct
- the cadence is repetitive
- the prose repeatedly stops itself to hedge and qualify

## 21. [P1] Acronym density creates friction, especially in the Abstract and captions

Why it matters:

- `sciwrite` explicitly warns about acronym overload and first-use discipline.
- The manuscript assumes a lot of shorthand very quickly.

Evidence:

- `paper/Config/abstract.tex:3-9`
- `paper/Sections/C2_Related_Work/main.tex:34-45`
- `paper/Sections/C4_Simulation_Results/main.tex:74-85`
- `paper/Sections/C4_Simulation_Results/main.tex:122-128`

Specific issue:

- `DSP` and `RMSE` appear in the Abstract without being expanded.
- Table captions behave like mini-glossaries, which is helpful, but also heavy.

## 22. [P1] Related Work has the rhythm of a generated coverage paragraph

Why it matters:

- `research-paper-writing` and `writing_quality_check` both warn against
  mirror-structured paragraphs where every sentence performs the same function.
- The opening Related Work paragraph is exactly that.

Evidence:

- `paper/Sections/C2_Related_Work/main.tex:4-16`

Pattern:

- family name
- one-line property
- one-line limitation
- next family

The paragraph is fluent, but too even.

## 23. [P1] Overpacked noun stacks and compound-heavy phrases make the prose airless

Why it matters:

- The draft frequently compresses multiple qualifiers into one noun phrase
  instead of distributing information across sentences.
- This is a common AI compression habit.

Evidence:

- `paper/Config/abstract.tex:2-5`
- `paper/Config/abstract.tex:17-19`
- `paper/Sections/C1_Introduction/main.tex:28-30`
- `paper/Sections/C1_Introduction/main.tex:39-41`
- `paper/Sections/C5_Conclusions/main.tex:49-52`

Examples:

- `stress-oriented benchmark`
- `local, non-time-synchronized relay-class configuration`
- `scenario-optimized classical methods`
- `probabilistic compliance bounds`

## 24. [P1] The manuscript keeps reasserting the same "the standard is not enough" claim

Why it matters:

- Repetition is not always emphasis. Sometimes it is just recurrence.
- Here, the standard-insufficiency claim appears so often that it starts to
  sound like a memorized refrain.

Evidence:

- `paper/Config/abstract.tex:16-19`
- `paper/Sections/C3_Methods/main.tex:79-80`
- `paper/Sections/C5_Conclusions/main.tex:4-9`
- `paper/Sections/C5_Conclusions/main.tex:33-37`
- `paper/Sections/C5_Conclusions/main.tex:47-51`

## 25. [P2] Some paragraphs fail the load-bearing test

Why it matters:

- `writing_judgment_framework` asks: if this paragraph disappears, does the
  paper still make sense?
- Several paragraphs feel equally polished even though they are not equally
  important.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:19-26`
- `paper/Sections/C2_Related_Work/main.tex:4-16`
- `paper/Sections/C5_Conclusions/main.tex:41-53`

Interpretation:

- the draft spends high polish on benchmark framing and family summary
- it spends less rhetorical force on the one or two takeaways that should
  carry the paper

## 26. [P2] The Introduction stops after the contribution list and never gives a roadmap

Why it matters:

- `paper-write` expects an Introduction to close with a short roadmap or at
  least a forward-looking transition.
- This is not an AI-pattern in itself, but it makes the section feel abruptly
  templated.

Evidence:

- `paper/Sections/C1_Introduction/main.tex:28-41`

Effect:

- the section ends on a polished list
- there is no final sentence that orients the reader toward the rest of the paper

## 27. [P2] Captions and tables are carrying argument that should live in the main prose

Why it matters:

- The paper's tables and figure captions are detailed and useful.
- But several arguments, caveats, and interpretation steps are living there
  instead of in the main paragraph flow.

Evidence:

- `paper/Sections/C2_Related_Work/main.tex:26-47`
- `paper/Sections/C3_Methods/main.tex:105-119`
- `paper/Sections/C4_Simulation_Results/main.tex:74-85`
- `paper/Sections/C4_Simulation_Results/main.tex:174-200`

Effect:

- the manuscript looks technically rich
- the main text can still feel under-interpreted

## 28. [P2] The draft sometimes sounds too polished and not authored enough

Why it matters:

- This is the broadest `avoid-ai-writing` signal.
- The English is mostly clean, but the sentence-level voice is often neutral,
  symmetric, and risk-averse in a way that feels generated.

Evidence:

- `paper/Config/abstract.tex:2-19`
- `paper/Sections/C1_Introduction/main.tex:19-41`
- `paper/Sections/C2_Related_Work/main.tex:4-16`
- `paper/Sections/C5_Conclusions/main.tex:4-53`

What creates that effect:

- repeated benchmark-branding language
- mirrored family-by-family exposition
- polished synthesis phrases
- compressed but not vivid sentence design

## 29. [P2] The final cautionary close is generic

Why it matters:

- The final sentences do what many generated conclusions do:
  admit scope, mention HIL future work, propose a next-step scenario, mention
  reproducibility.
- All of that is reasonable. The issue is that the sequence is generic.

Evidence:

- `paper/Sections/C5_Conclusions/main.tex:41-53`

## 30. [P2] The paper often states finished synthesis before the reader has felt the tension

Why it matters:

- `avoid-ai-writing` and `research-paper-writing` both point to this in
  different language: models like neat synthesis too early.
- The draft often arrives at the conclusion before it has dramatized the
  underlying evidence path.

Evidence:

- `paper/Config/abstract.tex:11-15`
- `paper/Sections/C1_Introduction/main.tex:39-41`
- `paper/Sections/C5_Conclusions/main.tex:7-31`

Examples:

- `complementary strengths`
- `no single family dominates`
- `selection guidance`
- `principal observations`

## Bottom line

The manuscript is readable, technically serious, and structurally competent.
The main writing problem is not grammar. It is pattern density.

The draft currently sounds AI-shaped because it repeatedly does the same things:

- re-sells the benchmark
- summarizes estimator families in mirrored templates
- compresses claims with semicolons, dashes, and caveats
- prefers polished synthesis over one sharp empirical punch
- closes with a generic, well-behaved three-part conclusion

The best rewrite strategy is not sentence polishing alone. It is structural:

1. rewrite the Abstract around one concrete result
2. rebuild Related Work around the gap
3. convert Results snippets into interpretive paragraphs
4. collapse repeated benchmark-branding sentences
5. rename disturbance classes consistently across the whole paper
