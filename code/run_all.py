"""
run_all.py  –  Generate all figures for the paper.

Run from the code/ directory:
    python run_all.py

Each script saves its figure as both PDF and PNG in the current
working directory (code/).  The LaTeX article references the PDFs.

Dependencies: numpy, scipy, matplotlib  (all standard; no R or lme4)
"""

import subprocess, sys, pathlib

scripts = [
    "fig1_forking_paths.py",
    "fig2_convergence.py",
    "fig3_variance_assumption.py",
    "fig4_logodds.py",
    "fig5_simple_stats.py",
]

here = pathlib.Path(__file__).parent

for s in scripts:
    print(f"\n{'='*60}\nRunning {s} …")
    result = subprocess.run(
        [sys.executable, str(here / s)],
        cwd=str(here),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR in {s} (exit code {result.returncode})")
    else:
        print(f"  Done.")

print("\nAll figures generated.")
