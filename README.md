# Sparse-View 3D Reconstruction Proposal Slides

This repository contains a LaTeX Beamer slide deck created from
`Sparse_View_3D_Reconstruction_Proposal.docx`.

## Files

- `Sparse_View_3D_Reconstruction_Proposal.docx`: source proposal context
- `pdf/main.tex`: Beamer entry point
- `pdf/sections/`: slide content split by topic
- `pdf/main.pdf`: generated presentation

## Build

From `pdf/`, use the required repo workflow:

```bash
latexmk -pdf main.tex
latexmk -c main.tex
find . -maxdepth 1 -type f \( \
  -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
  -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
  -name "*.bbl" -o -name "*.blg" \
\) -delete
```
