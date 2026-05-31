#!/usr/bin/env python3
# Generates slides_guion.tex: each page = rendered slide (left) + a full,
# word-for-word English speaker script (right) to memorize and deliver.
# Page HEIGHT is fitted per slide to the taller of (slide image, script), so
# there is no white gap and long scripts still fit. Frame images:
# build/guion/g-NN.png  (render: pdftoppm -png -r 130 slide_flat.pdf build/guion/g)
#
# Each entry: (title, tier, [say sentences -> verbatim], transition, key line)
#   tier: CORE = memorize verbatim | FIG = talk to the picture | QUICK = divider
# Time per slide is computed from word count (~145 wpm).
import os

S = [
 ("Title", "CORE", [
   "Good morning, everyone. Thank you for being here. My name is Jorge Mayorga, from Universidad de los Andes.",
   "I want to open with a question that sounds simple but really is not: in a power grid that is increasingly run by inverters, which frequency estimator should we actually trust?",
   "We tend to assume the estimator is a solved, interchangeable block. It is not.",
   "The honest answer to that question is, it depends, and the whole point of this talk is to make 'it depends' precise, measurable, and useful for engineers."],
   "So let me begin with why this is a problem worth solving.",
   "There is no universal best estimator; the right choice depends on what you measure."),
 ("Motivation (section)", "QUICK", [], "",
   "Why this matters."),
 ("Field events define the stress model", "CORE", [
   "Let me ground this in reality. Real inverter-driven disturbances look nothing like the clean sine wave we use in textbooks.",
   "If you look at recorded events, Blue Cut, Odessa, the Kauai oscillation, you see phase jumps, fast rate-of-change of frequency, harmonics, and noise all arriving at the same time.",
   "So instead of inventing convenient test signals, we let these real field events define the stress cases that our benchmark has to cover.",
   "That keeps the evaluation honest, because the grid does not care about our assumptions."],
   "And here is the catch: the very instrument we use to watch these events also shapes what we conclude about them.",
   "Field events, not textbook signals, set the bar."),
 ("PMU streams show consequences", "CORE", [
   "A standard PMU hands us a frequency value and a rate-of-change of frequency.",
   "But during a fast inverter event, those numbers show the consequence, not always the cause.",
   "And crucially, it is the estimator buried inside the PMU that decides what we actually see on the screen.",
   "Two different estimators, looking at the very same event, can report two different stories."],
   "So before we can trust any analysis of an event, we first have to ask which estimator produced those numbers.",
   "Choosing the estimator is itself a measurement decision."),
 ("Research problem", "CORE", [
   "And that points straight at the gap we are addressing. Today there is no agreed, systematic way to compare these estimators under inverter-specific stress.",
   "The compliance tests we do have run on clean, well-behaved signals.",
   "So they tell us a method passes a standard, but they never tell us how it behaves during a real, messy, composite event.",
   "We are essentially certifying estimators on the easy cases and hoping they survive the hard ones."],
   "Our response to that gap is one clear claim.",
   "We need a fair, inverter-relevant way to compare estimators."),
 ("Operating claim", "CORE", [
   "The central claim of this work is that estimator selection has to be conditional.",
   "It is not a single ranking. It depends on three things at once: the type of disturbance, the metric you actually care about, and your compute budget.",
   "Change any one of those, and the best estimator can change with it.",
   "So our job is not to crown a winner; it is to map out which method wins under which conditions."],
   "And to back that claim with evidence, we built three things.",
   "There is no single best estimator, only a best one for a given job."),
 ("Technical contributions", "CORE", [
   "We make three contributions. First, an open benchmark platform built around known-truth scenarios, so the error is never in doubt.",
   "Second, a common execution contract, so that all eighteen estimators are tested under exactly identical conditions, with no method getting an unfair advantage.",
   "And third, a metric vector that keeps accuracy, trip-risk, latency, and computational cost as separate, explicit outputs, instead of collapsing them into one number."],
   "Let me give you a feel for the scale of all this.",
   "Open platform, common contract, and multi-metric evidence."),
 ("Experiment scale", "CORE", [
   "The scale is deliberately large: eighteen estimators, thirty-three scenarios, across five families of stress, all with Monte-Carlo repetitions.",
   "On top of that, we built a severity-sweep engine that we call ATLAS, which pushes each stress from mild to extreme.",
   "And every single result is reproducible from saved artifacts, so anyone can regenerate the figures you will see today."],
   "And the logic for choosing among all of these methods actually fits into a single picture.",
   "Large, factorial, and fully reproducible."),
 ("Estimator selection logic (build-up)", "FIG", [
   "This diagram is, in a sense, the entire talk in miniature.",
   "Selection comes down to three questions: which event class are you facing, which metric matters for your application, and what compute budget do you have.",
   "Watch as each question appears, because each one, on its own, can send you to a different estimator."],
   "With that logic in hand, let me now show you the platform itself.",
   "This one diagram is the whole talk in miniature."),
 ("Benchmark and platform (section)", "QUICK", [], "",
   "First, the benchmark and the platform."),
 ("Benchmark literature gap", "CORE", [
   "If you look at the existing literature, most studies compare a handful of estimators on just one or two cases.",
   "What has been missing is a benchmark that covers the full estimator space, against composite inverter stress, with metrics you can actually reproduce.",
   "That combination, breadth, realism, and reproducibility, simply did not exist."],
   "So that is exactly what we set out to build.",
   "That blank space on the map is the gap we fill."),
 ("Benchmark comparison axes", "CORE", [
   "We compare the methods on four axes at the same time: accuracy, robustness, computational cost, and protection trip exposure.",
   "And the whole reason we keep them separate is that a method can win brilliantly on one axis and quietly lose on another.",
   "A single leaderboard would hide exactly the trade-off that an engineer needs to see."],
   "Here is the platform that measures all four of them.",
   "One number hides the trade-off; four axes reveal it."),
 ("OpenFreqBench platform", "FIG", [
   "This is OpenFreqBench. It has four parts.",
   "A scenario factory that generates waveforms with known true frequency. A uniform interface so every estimator plugs in the same way. A locked metric engine so the scoring never changes between methods. And fully traceable artifacts, every CSV and every figure.",
   "The design philosophy is simple: known truth in, locked metrics out, and everything traceable."],
   "Let me show you concretely how we keep the comparison fair.",
   "Known truth in, locked metrics out, everything traced."),
 ("Experimental method (1)", "CORE", [
   "Fairness is really the heart of this whole project.",
   "Every estimator sees the exact same input trace, driven by the same random seeds, with the same warm-up handling.",
   "Nobody gets a cleaner signal or a luckier start. The comparison is fair by construction, not because we trust ourselves to be fair."],
   "And we are just as strict about tuning and about timing.",
   "Fairness is enforced by construction, not by trust."),
 ("Experimental method (2)", "FIG", [
   "We tune every method under a single fixed policy, we discard the warm-up region before scoring, and we record CPU cost as a genuine output, not as an afterthought.",
   "So the computational numbers you will see are measured, not assumed."],
   "Put all of that together, and this is the pipeline.",
   "Nothing is hand-tuned to flatter one method."),
 ("Full benchmark workflow", "FIG", [
   "The workflow runs in a fixed order: scenario, then Monte-Carlo, then the estimator, then the locked metrics, and finally the artifacts.",
   "And I want to stress this: every single figure in this talk traces straight back to those artifacts. Nothing here is hand-drawn."],
   "Now let me introduce the scenarios themselves.",
   "One reproducible pipeline, end to end."),
 ("Standard stresses: controlled waveform", "CORE", [
   "We start with the standard stresses: a magnitude step, a frequency ramp, and amplitude modulation.",
   "Because we generate the waveform ourselves, we know the true frequency at every single sample.",
   "That means the error is not estimated, it is exact, and that is what makes a fair score possible."],
   "Here is what those standard cases actually look like.",
   "Known truth is what lets us score the error with no ambiguity."),
 ("Scenario suite I (dashboard)", "FIG", [
   "This dashboard shows the standard suite. Voltage is on the top row, the true frequency on the bottom, one column per scenario.",
   "And here is a subtle but important point: for the magnitude step and the AM case, the frequency stays perfectly flat at sixty hertz.",
   "The stress is not a frequency event at all, and yet, as we will see, the estimators still react to it."],
   "Now let us make the scenarios realistic.",
   "The stress is not always a frequency event, but estimators react anyway."),
 ("Composite IBR stresses", "CORE", [
   "Now we stack the disturbances on top of one another: harmonics, phase jumps, a rate-of-change segment, and a ringdown, all inside one waveform.",
   "This is far closer to what a real, inverter-dominated fault actually produces in the field.",
   "It is messy on purpose, because the grid is messy."],
   "Here is that composite suite.",
   "Composite stress is the realistic, and the revealing, case."),
 ("Scenario suite II (dashboard)", "FIG", [
   "Same layout as before, voltage on top and frequency below, but now these cases are deliberately harder than anything you would find in a compliance test.",
   "If a method can stay accurate here, it has genuinely earned our trust."],
   "Before we look at methods, let me name the three mechanisms that actually break estimators.",
   "If a method survives here, it has earned our trust."),
 ("Scenario families: distortion, leakage, and low SNR", "CORE", [
   "It helps to group the hard scenarios into three families, because each one breaks an estimator in a different way.",
   "Harmonic distortion attacks any method that assumes a pure sinusoid. Spectral leakage hurts the window-based methods when the frequency does not sit neatly inside a bin.",
   "And a low signal-to-noise ratio simply drowns the zero-crossing and timing-based methods.",
   "Keeping these three failure mechanisms separate is exactly what lets us explain, later, why a given method wins or collapses."],
   "With those mechanisms named, here are the estimators we put up against them.",
   "Distortion, leakage, and low SNR are three distinct failure mechanisms, not one."),
 ("Estimator families", "CORE", [
   "The eighteen estimators sort naturally into families.",
   "Loop-based methods, the PLLs. Window and spectral methods. Model-based Kalman filters. Adaptive methods. And data-driven methods.",
   "We deliberately analyze by family, rather than method by method, because the family is what really explains the behavior we see, the strengths and the failure modes."],
   "And these are the numbers we score them on.",
   "Eighteen estimators, five families, one fair contract."),
 ("Metrics and decision variables", "CORE", [
   "Here are the metrics. RMSE captures the average error. Peak error captures the worst moment.",
   "And then a trip-risk time, which counts how long the error stays above half a hertz, because that is where protection relays start to act.",
   "Plus settling time, and CPU cost. Five dimensions, kept separate on purpose."],
   "Now let me show you what the data actually says.",
   "Half a hertz is our protection-oriented risk proxy."),
 ("Core evidence (section)", "QUICK", [], "",
   "The core evidence: five scenarios, from standard to composite."),
 ("Standard scenario results (build-up)", "CORE", [
   "Let us start with the standard cases. On the step and the modulation, almost everything looks excellent, all of them below fifty millihertz.",
   "If we stopped here, we would conclude that the estimator choice barely matters.",
   "CLICK. But now watch the ramp column. The textbook EKF and the IpDFT intermittently blow up, to thirty-seven and fourteen hertz.",
   "And these are not edge methods; these are textbook, widely-used estimators."],
   "You would expect a phase jump to be even worse, so let me show you that next.",
   "Robustness, not point accuracy, is what separates the families."),
 ("Scenario D: phase jump separates families", "CORE", [
   "A clean sixty-degree phase jump cleanly separates the families.",
   "The state-space filters and the PLLs absorb it gracefully, with effectively zero trip exposure.",
   "The windowed and spectral methods, on the other hand, carry over a full second of trip exposure, because their analysis window straddles the discontinuity.",
   "But here is the part that surprised us."],
   "Because a single, clean jump is not actually the hard problem.",
   "The phase jump does not break the EKF; divergence on the ramp is the real danger."),
 ("Composite IBR sequence: hardest case", "CORE", [
   "Scenario E stacks everything together: harmonics, two phase jumps, a rate-of-change segment, a ringdown, and impulsive noise.",
   "So spectral leakage, loop re-locking, and state-model mismatch all hit the estimator at the very same instant.",
   "There is nowhere for a method to hide."],
   "And this is exactly where the leaderboard falls apart.",
   "This is the hardest, most realistic case in the whole suite."),
 ("Scenario E: the leaderboard breaks", "CORE", [
   "Under that stacked sequence, every usable method collapses to roughly one-point-one hertz of error, and ZCD diverges all the way to four hundred hertz.",
   "There is no accurate winner here; high accuracy is simply not on the table.",
   "And worse, the rankings by RMSE, by trip-risk, and by CPU now disagree with one another. The method with the best RMSE is not the cheapest, and not the safest."],
   "Let me make the trip-risk side of that visual.",
   "Under stacked stress, there is simply no accurate winner."),
 ("Trip-risk evidence (figure)", "FIG", [
   "This is the trip-risk picture. In Scenario D, the state-space filters hold zero exposure, while the windowed methods sail past one-point-two seconds.",
   "But in Scenario E, look, everyone carries exposure. Nobody is safe.",
   "And notice the ranking here is completely different from the RMSE ranking."],
   "And you can actually watch the divergence happen, sample by sample.",
   "Trip-risk is a completely different ranking from RMSE."),
 ("Event responses (figure)", "FIG", [
   "Here it is, directly. The true frequency is the black line, and each colored line is an estimator trying to follow it.",
   "You can literally watch EKF and IpDFT run away on the ramp, climbing to tens of hertz, while RA-EKF just stays locked on the truth.",
   "This is what intermittent divergence looks like in practice."],
   "So what does all of this mean when you actually have to choose a method?",
   "This is the intermittent divergence, seen with your own eyes."),
 ("Result comparison: winners by question", "CORE", [
   "The honest summary is that the winner depends on the question you ask.",
   "Ask for clean steady-state accuracy, and the state-space methods and SOGI-FLL win.",
   "Ask for ramp robustness, and it is RA-EKF. Ask about the multi-event case, and there is no winner at all. Ask about feasibility, and the expensive methods simply drop out of contention."],
   "And that last point, feasibility, brings us straight to cost.",
   "Different question, different winner, every single time."),
 ("Computational feasibility", "CORE", [
   "Cost matters, because these run on relays and embedded hardware.",
   "The loop and state-space methods run in nine to seventeen microseconds per sample. That is comfortably embedded-feasible.",
   "The spectral and data-driven methods, by contrast, are a hundred to a thousand times slower. For a real-time protection device, that is often a deal-breaker, regardless of accuracy."],
   "Put cost and risk on the same picture, and it becomes even clearer.",
   "Cost alone already rules several methods out of a relay."),
 ("Deployment limits (figure)", "FIG", [
   "Deployment is really the combination of all of it: accuracy, trip exposure, cost, and divergence-robustness, together.",
   "And the encouraging thing in this picture is that the low-cost cluster, the state-space and loop methods, is also the low-trip cluster.",
   "The cheap methods are often also the safe ones."],
   "We can capture each method's overall balance in a single shape.",
   "The practical winners sit together, at low cost and low trip."),
 ("Balance profile (radar)", "FIG", [
   "On this radar, each axis is one scenario, and a larger area means more robust.",
   "RA-EKF and SOGI-FLL stay robust on every single axis. EKF and IpDFT, in contrast, collapse on the ramp axis.",
   "The shape tells you the whole story at a glance."],
   "Now, this next slide is the most important one about compliance testing.",
   "No method is strong on every axis."),
 ("Compliance heatmap (build-up)", "CORE", [
   "This heatmap is the heart of the argument. Look at the standard columns first, step, ramp, and modulation. Almost every method passes.",
   "On a standard compliance test, these all look qualified.",
   "CLICK. But now the event columns appear, phase jump and multi-event, and watch: the same methods that passed now fail.",
   "Passing the standard tests told us almost nothing about event readiness."],
   "Let me distill all of this into our empirical claims.",
   "Passing the standard tests does not predict event readiness."),
 ("Empirical claims", "CORE", [
   "Four empirical claims come out of the core study.",
   "First, divergence, not phase jumps, is the dominant failure mode.",
   "Second, simple pass-fail testing leaves event readiness unresolved. Third, the three metrics, accuracy, trip-risk, and cost, pick genuinely different winners. And fourth, feasibility changes the ranking again."],
   "Now the natural question is: does this pattern survive when we scale the study up?",
   "Four claims, one theme: always report the conditions."),
 ("Expanded and ATLAS (section)", "QUICK", [], "",
   "Does the pattern hold across families and severity? That is ATLAS."),
 ("Expanded stress analysis", "CORE", [
   "To test that, we scale up to a full Monte-Carlo benchmark: eighteen estimators, thirty-three scenarios, five seeds each.",
   "And then we add the ATLAS severity sweeps on top, which push each disturbance from mild all the way to extreme."],
   "The first question is whether the family winners stay fixed as we scale.",
   "We scale the test, we do not just repeat it."),
 ("SGSMA benchmark: family winners", "CORE", [
   "And they do not stay fixed. The winner rotates from family to family.",
   "One method leads on the step, a different one on the ramp, another on harmonics, another on the multi-event case.",
   "There is no method that quietly wins everywhere when you are not looking."],
   "ATLAS lets us understand why, by sweeping the severity continuously.",
   "The leaderboard is regime-dependent, never absolute."),
 ("ATLAS severity analysis", "CORE", [
   "ATLAS sweeps each stress by severity and, importantly, by sign, under a fixed policy, at thirty Monte-Carlo runs per point.",
   "And I want to be clear about evidence quality here: this is now paper-grade across all nine of the required sweeps, not a quick pilot."],
   "And the headline result from those sweeps is genuinely striking.",
   "This is paper-grade severity evidence, not a pilot."),
 ("ATLAS: metric choice changes the answer", "CORE", [
   "Across a hundred and thirteen severity levels, the preferred estimator changes the moment you change the objective.",
   "In eighty-three of those levels, optimizing for RMSE and optimizing for trip-risk point you to two different methods.",
   "So the deliverable from ATLAS is not a ranking. It is a selection map: tell me your event and your metric, and it tells you the method."],
   "ATLAS also tells us something equally useful: where each method gives up.",
   "The output is a selection map, not a leaderboard."),
 ("ATLAS: where estimators stop being valid", "FIG", [
   "For every method, ATLAS pinpoints the exact severity at which it crosses over from working to failing.",
   "That boundary, the point of no return, is different for every family, and knowing it is what lets you deploy a method safely."],
   "Let me now show you a few of the most surprising individual sweeps.",
   "That validity boundary is different for every family."),
 ("AM, FM and interharmonics", "CORE", [
   "Even within modulation, the story splits in two.",
   "Amplitude modulation is relatively easy, and there the model-based methods win.",
   "But frequency modulation and interharmonics are much harder, and there the spectral methods take over. Same broad category, opposite winners."],
   "Now the handful of findings I most want you to walk away with.",
   "Even inside modulation, the winner switches."),
 ("When the textbook filter diverges", "CORE", [
   "This is the divergence mechanism up close. The textbook EKF tracks most seeds perfectly well, and then, on certain seeds, it intermittently runs away to tens of hertz.",
   "And here is the unsettling part: it is not driven by severity. The worst seed in our set was actually the one with the gentlest ramp.",
   "So you cannot predict it from how hard the event looks."],
   "The next finding is about something we usually ignore: direction.",
   "The robustified RA-EKF removes the divergence on every single seed."),
 ("RoCoF has a sign", "CORE", [
   "Rate-of-change of frequency has a sign, and it turns out that down-ramps are harder than up-ramps.",
   "The linear Kalman filter is five-and-a-half times worse when the frequency is falling. TKEO is almost four times worse.",
   "This matters enormously, because a falling frequency is the protection-critical direction, the one that sheds load."],
   "And severity can surprise us in the opposite direction too.",
   "A rising-only test can certify a method that fails when frequency drops."),
 ("Phase-jump severity is non-monotone", "CORE", [
   "Phase-jump severity is non-monotone, which is genuinely counterintuitive. The worst case is not the largest jump.",
   "Most methods actually peak in error around ninety to a hundred and twenty degrees, and then recover toward a hundred and eighty, because a half-cycle jump partly cancels itself out.",
   "So if you test only at one nominal angle, you can completely miss the real worst case."],
   "Severity also reshuffles the winners across the whole sweep.",
   "Bigger is not always worse, so you have to sweep the range."),
 ("No universal winner (map)", "FIG", [
   "This map really captures the central message. The champion rotates by rate-of-change regime.",
   "Koopman wins at low rates. ESPRIT wins in the middle. SOGI-PLL takes over once it gets severe.",
   "No single column owns the whole map."],
   "And remarkably, one estimator manages to be both the best and the worst.",
   "No universal winner; the deliverable is the map itself."),
 ("The ZCD paradox (build-up)", "CORE", [
   "This is my favorite result, the ZCD paradox.",
   "Below fifteen percent harmonic distortion, zero-crossing detection is the single most accurate method we tested. The best of all eighteen.",
   "CLICK. But add broadband noise, and that very same method explodes to three thousand hertz.",
   "CLICK. And push past twenty percent distortion, and it collapses again, to tens of hertz.",
   "Best and worst, the same estimator, depending only on the stress."],
   "And the very same asymmetry shows up in voltage events.",
   "Aggregate rankings would completely hide this."),
 ("Sags and swells are not symmetric", "CORE", [
   "Voltage magnitude is not symmetric either. A deep sag is far harder to handle than an equal swell.",
   "Zero-crossing detection goes from millihertz of error on a swell to hundreds of hertz on a deep sag.",
   "The direction of the voltage event, down versus up, changes the result by orders of magnitude."],
   "And there is a clean, physical reason for that asymmetry.",
   "Direction matters, so you have to sweep both signs."),
 ("Why a sag breaks zero-crossing", "FIG", [
   "The mechanism is simple physics. The timing error of a zero-crossing scales with the noise divided by the amplitude times the frequency.",
   "A deep sag shrinks the amplitude, which flattens the slope of the waveform right at the crossing.",
   "And once the slope is flat, even small noise creates false crossings. It is not a coding bug; it is the geometry of the signal."],
   "Harmonics tell a similar, family-by-family story.",
   "This is physics, not a software bug."),
 ("Harmonic degradation by family", "CORE", [
   "Harmonic degradation also depends strongly on the family.",
   "Loop methods like ZCD win at low distortion, but they collapse once you go above roughly fifteen percent.",
   "Window methods degrade much more gently, and they actually end up leading at extreme distortion. So low-distortion accuracy does not predict high-distortion survival."],
   "And cost is the final dimension to bring in.",
   "Low-distortion accuracy does not predict high-distortion survival."),
 ("CPU-accuracy Pareto", "FIG", [
   "On the cost-accuracy Pareto front, the spectral and data-driven methods do buy you a little extra accuracy.",
   "But they pay for it with a hundred to a thousand times more computation.",
   "For relay-class, real-time timing, that trade is almost never worth it."],
   "So, with all of that on the table, what do we actually deploy?",
   "For relay-class timing, that trade is usually not worth it."),
 ("Findings and deployment (section)", "QUICK", [], "",
   "So what do we actually deploy?"),
 ("Three findings", "CORE", [
   "Three findings sum up the work.",
   "First, there is a clear latency-versus-stress trade-off; the most accurate methods are often the slowest.",
   "Second, robustness beats point accuracy. RA-EKF, with its explicit rate-of-change state, stays stable exactly where the textbook EKF and IpDFT diverge.",
   "And third, compliance tests, on their own, are simply not enough for inverter-dominated events."],
   "Let me be honest about the limits of what we have shown.",
   "Robustness, not accuracy, is the deployment discriminator."),
 ("Validity limits and next steps", "CORE", [
   "On validity, I want to be transparent. The core scenario means are at five seeds, and the ATLAS severity sweeps are paper-grade at thirty.",
   "The next step is journal-grade, at a hundred runs, and adding the subspace and machine-learning methods to the full sweep.",
   "We are deliberately explicit about what is diagnostic and what is paper-grade evidence."],
   "From there, the work becomes a concrete recommendation.",
   "We are explicit about what is diagnostic and what is paper-grade."),
 ("Recommendation map", "CORE", [
   "So here is our recommendation, and notice it is a map, not a single answer.",
   "For fast frequency response and anti-islanding, RA-EKF is the strongest, best-supported choice in our benchmark.",
   "PLL and EKF make solid low-cost baselines. And IpDFT or TFT are good where some observation delay is acceptable."],
   "We then wrap that into a qualification idea.",
   "We deliver a recommendation map, not a single ranking."),
 ("Qualification profile", "FIG", [
   "A real qualification profile, in our view, should test transient recovery, trip exposure, compute feasibility, and standard-case RMSE, all together, not one at a time.",
   "Because, as we have seen, a method can pass any one of these and fail the others."],
   "And it should be reported in a standard, comparable form.",
   "Qualify on all dimensions at once, not one at a time."),
 ("Standardization profile", "FIG", [
   "So we propose reporting these dimensions explicitly and consistently, so that any two estimators can be compared on truly equal terms.",
   "That is what turns a one-off study into something the community can build on."],
   "And when you genuinely need just one number, here is how we do it.",
   "Make the comparison reproducible and fair."),
 ("IBR Event Qualification Score", "FIG", [
   "When a single number is unavoidable, for instance in a procurement spec, we define a composite, weighted, event-class qualification score.",
   "It rolls the dimensions into one figure, but it does so transparently, so you can always see what went into it."],
   "Finally, let me point to where this work is heading.",
   "One number when you need it, without hiding the dimensions."),
 ("Toward realtime IBR estimators", "CORE", [
   "Looking ahead, there is a clear roadmap.",
   "Real-time estimators on hardware. Integrating recorded Simulink microgrid fault data. Adding modern and machine-learning methods. C++ acceleration for the heavy methods. And co-simulation with ANDES for closed-loop studies."],
   "And with that, let me close.",
   "This is where the benchmark goes next."),
 ("Validation agenda and conclusion", "CORE", [
   "So, to conclude. The single most important message is that there is no universal frequency estimator.",
   "The defensible choice always depends on the event you expect, the metric you care about, and the compute budget you have.",
   "And what OpenFreqBench provides is a reproducible, open way for the community to make that choice on evidence, rather than on habit."],
   "",
   "Thank you very much for your attention. I would be very glad to take your questions."),
]

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = "build/guion/g-{:02d}.png"
TIER_COLOR = {"CORE": "ofbTeal", "FIG": "ofbBlue", "QUICK": "ofbMuted"}

def esc(t):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        t = t.replace(a, b)
    return t

def secs(say, trans):
    words = sum(len(s.split()) for s in say) + len(trans.split())
    return max(5, round(words / 2.42))  # ~145 wpm

total = 0
for _, _, say, trans, _ in [(a,b,c,d,e) for (a,b,c,d,e) in S]:
    total += secs(say, trans)
mm, ss = divmod(total, 60)

out = []
# Variable page height: each page is fitted to the taller of (slide, script)
# plus margins, so no white gap and long scripts always fit. Width fixed 30 cm.
out.append(r"""\documentclass[11pt]{article}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{tcolorbox}
\usepackage{calc}
\definecolor{ofbTeal}{HTML}{0E6E6E}
\definecolor{ofbBlue}{HTML}{1F5C8B}
\definecolor{ofbInk}{HTML}{1A1A1A}
\definecolor{ofbMuted}{HTML}{5B5B5B}
\setlength{\pdfpagewidth}{31cm}\setlength{\paperwidth}{31cm}
\setlength{\hoffset}{-1in}\setlength{\voffset}{-1in}
\setlength{\oddsidemargin}{0.8cm}\setlength{\evensidemargin}{0.8cm}
\setlength{\topmargin}{0.8cm}\setlength{\headheight}{0pt}\setlength{\headsep}{0pt}\setlength{\topskip}{0pt}
\setlength{\textwidth}{29.4cm}\setlength{\textheight}{60cm}
\setlength{\parindent}{0pt}\pagestyle{empty}
\newsavebox{\slidebox}
\begin{document}
""")
run = 0
for i, (title, tier, say, trans, key) in enumerate(S, start=1):
    sc = secs(say, trans)
    run += sc
    rmm, rss = divmod(run, 60)
    col = TIER_COLOR.get(tier, "ofbInk")
    body = []
    body.append(r"\savebox{\slidebox}{%")
    body.append(r"\begin{minipage}[t]{0.55\textwidth}\vspace{0pt}")
    body.append(r"{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.4pt}\fbox{\includegraphics[width=\linewidth]{%s}}}" % IMG.format(i))
    body.append(r"\end{minipage}\hspace{1.0cm}%")
    body.append(r"\begin{minipage}[t]{0.40\textwidth}\vspace{0pt}")
    body.append(r"{\footnotesize\color{ofbMuted}Slide %d / %d \quad $\sim$%ds \quad cum %d:%02d}\hfill{\footnotesize\bfseries\color{%s}%s}\\[2pt]"
               % (i, len(S), sc, rmm, rss, col, tier))
    body.append(r"{\Large\bfseries\color{%s} %s}\\[6pt]" % (col, esc(title)))
    if say:
        body.append(r"{\small\itshape\color{ofbMuted}Say (verbatim):}\\[2pt]")
        body.append(r"\begin{itemize}[leftmargin=1.05em,itemsep=4pt,topsep=2pt]")
        for b in say:
            body.append(r"\item %s" % esc(b))
        body.append(r"\end{itemize}")
    if trans:
        body.append(r"\vspace{3pt}{\color{ofbBlue}$\rightarrow$ \itshape %s}\\[2pt]" % esc(trans))
    body.append(r"\vspace{4pt}")
    body.append(r"\begin{tcolorbox}[colback=%s!8,colframe=%s,boxrule=0.7pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt]" % (col, col))
    body.append(r"{\bfseries Key line: }%s" % esc(key))
    body.append(r"\end{tcolorbox}")
    body.append(r"\end{minipage}}")
    out.append("\n".join(body))
    # fit page height to the box
    out.append(r"\setlength{\pdfpageheight}{\dimexpr\ht\slidebox+\dp\slidebox+1.6cm\relax}\setlength{\paperheight}{\pdfpageheight}")
    out.append(r"\noindent\usebox{\slidebox}\clearpage")

out.append(r"\end{document}")

with open(os.path.join(ROOT, "slides_guion.tex"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("wrote slides_guion.tex:", len(S), "slides, est. total %d:%02d (%ds)" % (mm, ss, total))
