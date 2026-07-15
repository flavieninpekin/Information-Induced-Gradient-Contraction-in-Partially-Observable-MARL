# 510K Paper — AAAI-27 Submission

## Compile

Requires a LaTeX distribution (MiKTeX or TeX Live).

```bash
cd paper
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

On Windows with MiKTeX:
```pwsh
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Files

- `paper.tex`        — 7-page anonymous submission for AAAI-27
- `references.bib`   — Bibliography
- `aaai2027.sty`     — AAAI-27 style file (from Author Kit)
- `aaai2027.bst`     — AAAI-27 bibliography style
- `figures/`         — Figures (3 PDF + 3 PNG)

## Template Notes

- Uses `\usepackage[submission]{aaai2027}` for anonymous submission
- Change to `[camera]` for camera-ready version
- No hyperlinks, no \vspace with negative values
- Page limit: 7 pages + references

## Story

Self-play homogenizes strategic differentiation across rule changes in a 
multi-agent card game. SINGLE and STATIC converge (L2=0.03). 
DYNAMIC shows bimodal behavior. No seeds develop cooperative signatures.

## Targets

1. AAAI-27 (deadline July 28, 2026) — abstract July 21
2. AAMAS-27 (deadline Oct 2026) — fallback
