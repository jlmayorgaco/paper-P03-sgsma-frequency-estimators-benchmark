# Contributing Guide

## Development setup
1. Use Python 3.13.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -e .`

## Quality gates
- Canonical gate (required for PR merge):
  - `cd src && python -m pipelines.run_quality_gate --profile canonical`
- Legacy compatibility gate (recommended):
  - `cd src && python -m pipelines.run_quality_gate --profile legacy`
- Manual/nightly full benchmark gate:
  - `cd src && python -m pipelines.run_quality_gate --profile manual-nightly`

## Test policy
- Do not delete `tests/montecarlo`; it is a legacy compatibility harness.
- `tests/montecarlo/temp` is preserved for local experiments and is not part of canonical PR gate.
- New benchmark-science claims must be traceable to canonical artifacts in `artifacts/full_mc_benchmark/`.

## Coding policy
- Keep pipeline logic under `src/pipelines/`.
- Keep benchmark plotting under `src/plotting/benchmark/`.
- Preserve scientific contracts:
  - `step()` and `step_vectorized()` equivalence
  - deterministic ESPRIT behavior
  - no silent numerical masking in Prony
  - CPU timing via repeated `time.process_time()`

## Pull request checklist
- [ ] Canonical gate passes.
- [ ] Behavior/doc updates included.
- [ ] No heavy generated artifacts added.
- [ ] If benchmark behavior changed, artifact validator command/result provided.
