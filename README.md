# Simple Stats

Code and article for the paper *"Keep it simple: a case for minimal statistical models in psychology"*.

## Argument

The "keep it maximal" prescription (Barr et al., 2013) is theoretically defensible but practically harmful:
- Maximal random-effects models frequently fail to converge, making pre-registration nearly impossible.
- They impose an implicit variance-homogeneity assumption that is no more justified than any other.
- Most practitioners cannot fully explain what their model is doing.

Separately, logistic models express effects in log-odds — a scale that is not psychologically interpretable.

**Recommendation:** use the simplest test compatible with the research question. Binomial tests, paired *t*-tests, and χ² on contingency tables are not a compromise; they are the most transparent and reproducible tools available.

## Structure

```
simple_stats/
├── code/           # Python simulations (numpy/scipy/matplotlib, no R)
│   ├── fig1_forking_paths.py      # 676 RE structures for a 2×2 design
│   ├── fig2_convergence.py        # Convergence rate vs. model complexity
│   ├── fig3_variance_assumption.py# Hidden variance-pooling assumption
│   ├── fig4_logodds.py            # Log-odds are not interpretable
│   ├── fig5_simple_stats.py       # Type I error & power comparison
│   └── run_all.py                 # Run all figures in sequence
└── article/
    ├── simple_stats.tex           # LaTeX source (5 sections, 5 figures)
    └── references.bib
```

## Usage

**Generate figures** (from `code/`):
```bash
pip install numpy scipy matplotlib
python run_all.py
```

**Compile the paper** (from `article/`):
```bash
pdflatex simple_stats && bibtex simple_stats && pdflatex simple_stats && pdflatex simple_stats
```
