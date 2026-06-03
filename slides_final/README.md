# SGSMA 2026 Final Compact Deck

This folder contains the compact conference version of the SGSMA presentation.

- Main source: `slide_final.tex`
- Paper claim audit: `PAPER_CLAIM_AUDIT.md`
- Figure assets are referenced from `../slides/figures`
- The long working deck in `../slides` is not modified or required to select frames.

Build from this folder:

```bash
pdflatex slide_final.tex
pdflatex slide_final.tex
```

The compact deck integrates the SGSMA paper diagrams and claims first, then adds the OpenFreqBench/ATLAS extended stress analysis with explicit caveats.
