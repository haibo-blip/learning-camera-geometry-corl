# Overleaf Project

This directory is an Overleaf-ready CoRL 2026 LaTeX project for the paper draft.

## Main files

- `main.tex`: paper entry point
- `sections/*.tex`: section drafts
- `figures/`: current draft figures copied from the local project
- `corl_2026.sty`, `corlabbrvnat.bst`: official CoRL 2026 template files downloaded from the CoRL author instructions

## Notes

- The paper is still a draft. MimicGen results are preliminary and should be replaced once the full 4--5 task evaluation is available.
- References are intentionally empty until sources are verified.
- For an initial anonymous CoRL submission, keep `\usepackage{corl_2026}` in `main.tex`.
- For camera-ready, switch to `\usepackage[final]{corl_2026}`.

## Overleaf

Upload this directory as a zip to Overleaf, or connect it through Overleaf's Git integration using the repository remote.
