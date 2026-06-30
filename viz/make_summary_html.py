"""
Build a single static HTML summary (viz/summary.html) that gathers the headline
figures and the verified headline metrics in one page.  Every number is read
from a real artifact under viz/artifacts/; every figure is one produced by the
plot scripts.  Nothing is hardcoded.

Usage:
    python3 viz/make_summary_html.py
"""

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = _SCRIPT_DIR / "artifacts"
FIGURES = _SCRIPT_DIR / "figures"


def _load(path):
    p = ARTIFACTS / path
    return json.load(open(p)) if p.exists() else None


def main():
    cpp = _load("cpp_validation.json")
    bench = _load("surrogate_benchmark.json")
    ab = _load("a1_betaStar/summary.json")
    nw = _load("near_wall4/summary.json")

    rows = []
    if cpp:
        rows.append(("Solver validation, channel",
                     f"{cpp['channel']['rel_error_pct']:.1f}% vs Dean 1978",
                     "build/rans_sst --validate-channel"))
        rows.append(("Solver validation, flat plate",
                     f"{cpp['plate']['rel_error_pct']:.1f}% vs Schoenherr",
                     "build/rans_sst --validate-plate"))
    if bench:
        rows.append(("Surrogate speedup",
                     f"{bench['per_eval_speedup']:,.0f}x per evaluation "
                     f"({bench['surrogate_eval_per_s']:,.0f} eval/s vs "
                     f"{bench['cfd_s_per_solve']:.1f} s/solve)",
                     "surrogate_benchmark.json"))
    if ab:
        a1 = ab["posterior"]["a1"]; bs = ab["posterior"]["betaStar"]
        dg = ab["diagnostics"]
        rows.append(("Posterior (a1, beta*)",
                     f"a1 = {a1['mean']:.3f} +/- {a1['std']:.3f}, "
                     f"beta* = {bs['mean']:.3f} +/- {bs['std']:.3f}",
                     "a1_betaStar/summary.json"))
        rmax = max(dg["split_rhat"].values())
        rows.append(("MCMC convergence",
                     f"split-R-hat max {rmax:.3f}, acceptance {dg['acceptance_mean']:.2f}, "
                     f"{ab['budget']['n_posterior_samples']:,} samples",
                     "a1_betaStar/summary.json"))
    if nw:
        h = nw["surrogate"]
        rows.append(("Surrogate fidelity (sampling region)",
                     f"holdout RMSE {h['holdout_rmse_loglik']:.1f} log-lik units "
                     f"(n={h['n_holdout']})",
                     "near_wall4/summary.json"))
        C = nw["posterior_correlation"]["matrix"]
        rows.append(("Identifiability",
                     f"a1-beta* posterior correlation {C[0][1]:.2f} (ridge)",
                     "near_wall4/summary.json"))
        ls = h["ard_lengthscales"]
        inert = max(ls, key=ls.get)
        rows.append(("Sensitivity",
                     f"{inert} inert (ARD lengthscale {ls[inert]:,.0f})",
                     "near_wall4/summary.json"))

    figs = sorted(p.name for p in FIGURES.glob("*.png")) if FIGURES.exists() else []

    metric_rows = "\n".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td><td class='src'>{s}</td></tr>"
        for k, v, s in rows)
    fig_blocks = "\n".join(
        f"<figure><img src='figures/{f}' alt='{f}'><figcaption>{f}</figcaption></figure>"
        for f in figs)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>QBTM - calibration summary</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 1100px; color: #1b2430; line-height: 1.5; }}
 h1 {{ margin-bottom: 0.2rem; }} .sub {{ color: #5a6675; margin-top: 0; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
 td, th {{ border-bottom: 1px solid #e2e6ea; padding: 8px 10px; text-align: left;
          vertical-align: top; }}
 .src {{ color: #7a8694; font-family: ui-monospace, monospace; font-size: 0.85em; }}
 figure {{ margin: 0 0 2rem; }} img {{ max-width: 100%; border: 1px solid #e2e6ea;
          border-radius: 6px; }}
 figcaption {{ color: #5a6675; font-size: 0.9em; margin-top: 0.4rem;
              font-family: ui-monospace, monospace; }}
 .note {{ background: #f4f7fa; border-left: 4px solid #3b6ea5; padding: 0.6rem 1rem;
         border-radius: 4px; }}
</style></head><body>
<h1>QBTM: Bayesian calibration for RANS turbulence models</h1>
<p class="sub">Headline metrics and figures, all generated from real solver and
inference output on this machine.</p>
<p class="note">Every value below is read from an artifact under
<code>viz/artifacts/</code>; the source file or command is in the right column.</p>
<h2>Verified metrics</h2>
<table>
<tr><th>Quantity</th><th>Value</th><th>Source</th></tr>
{metric_rows}
</table>
<h2>Figures</h2>
{fig_blocks}
</body></html>
"""
    out = _SCRIPT_DIR / "summary.html"
    out.write_text(html)
    print(f"  wrote {out}  ({len(rows)} metrics, {len(figs)} figures)")


if __name__ == "__main__":
    main()
