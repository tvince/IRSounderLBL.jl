"""
Diff per-line Y values: ARTS (after AdaptHitranLineMixing order=1) vs
Julia _calc_W_and_Y, for CO2 lines in 664.5–665.5 cm⁻¹ at T=288.20 K,
P=1.0132 atm (AFGL US Std surface).

Inputs:
  data/arts_Y_near_665.csv   (from scripts/dump_arts_Y_near_665.py)
  data/julia_Y_near_665.csv  (from scripts/dump_julia_Y_near_665.jl)

Match key: (iso, ν±tol).  We report Y_per_atm side by side, the absolute
difference ΔY, the ratio, and an "LM impact" weight S·|ΔY|.  Lines flagged
where |ΔY·p| > 0.01 are the candidates for closing the 33 K ARTS-vs-Julia
gap.

Run with any Python that has pandas:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/diff_arts_julia_Y.py
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

ARTS  = os.path.join(DATA, "arts_Y_near_665.csv")
JULIA = os.path.join(DATA, "julia_Y_near_665.csv")
OUT   = os.path.join(DATA, "diff_arts_julia_Y_near_665.csv")

NU_TOL    = 1.0e-3   # cm⁻¹ — both come from HITRAN
S_FLOOR   = 1.0e-26  # ignore essentially-zero-strength lines for "important" view
P_atm     = 1.0132

# ── Load ────────────────────────────────────────────────────────────────────
a = pd.read_csv(ARTS)
j = pd.read_csv(JULIA)
print(f"ARTS rows:  {len(a)}")
print(f"Julia rows: {len(j)}")

# ── Match by (iso, ν) within NU_TOL ─────────────────────────────────────────
# Strategy: for each iso, sort both by ν, then merge_asof.
matched = []
for iso, ja in j.groupby("iso"):
    aa = a[a["iso_num"] == iso].copy()
    if aa.empty:
        print(f"  iso {iso}: 0 ARTS lines, {len(ja)} Julia lines (skip)")
        continue
    ja_s = ja.sort_values("nu_cm").reset_index(drop=True)
    aa_s = aa.sort_values("nu_cm").reset_index(drop=True)
    m = pd.merge_asof(
        ja_s, aa_s,
        on="nu_cm", direction="nearest", tolerance=NU_TOL,
        suffixes=("_jl", "_arts"),
    )
    n_match = m["Y_per_atm_arts"].notna().sum()
    print(f"  iso {iso}: matched {n_match}/{len(ja_s)} Julia rows against "
          f"{len(aa_s)} ARTS rows")
    matched.append(m)

mt = pd.concat(matched, ignore_index=True)

# Unmatched Julia rows
mt["matched"] = mt["Y_per_atm_arts"].notna()
n_unmatched = (~mt["matched"]).sum()
print(f"\nTotal matched: {mt['matched'].sum()}  unmatched (Julia w/o ARTS): {n_unmatched}")

# ── Stats on matched rows ───────────────────────────────────────────────────
m = mt[mt["matched"]].copy()
m["dY_per_atm"] = m["Y_per_atm_arts"] - m["Y_per_atm_jl"]
m["dY_dim"]     = m["dY_per_atm"] * P_atm
m["ratio"]      = m["Y_per_atm_arts"] / m["Y_per_atm_jl"]
m["abs_dY_dim"] = m["dY_dim"].abs()
m["S_x_abs_dY"] = m["S_T"] * m["abs_dY_dim"]  # "matters here" weight

print("\n── Summary of |Y_per_atm| (matched lines) ──")
print(f"  Julia: min={m['Y_per_atm_jl'].min(): .4f}  max={m['Y_per_atm_jl'].max(): .4f}  "
      f"|mean|={m['Y_per_atm_jl'].abs().mean(): .4e}")
print(f"  ARTS : min={m['Y_per_atm_arts'].min(): .4f}  max={m['Y_per_atm_arts'].max(): .4f}  "
      f"|mean|={m['Y_per_atm_arts'].abs().mean(): .4e}")

# ── Top divergences weighted by line strength ───────────────────────────────
top = m.sort_values("S_x_abs_dY", ascending=False).head(40)
print("\n── Top 40 lines by S·|ΔY·p|  (lines where Y disagreement most matters) ──")
cols = ["iso", "nu_cm", "Ji", "branch", "S_T",
        "Y_per_atm_jl", "Y_per_atm_arts", "dY_per_atm", "ratio"]
with pd.option_context("display.float_format", "{:+.4e}".format,
                       "display.width", 200,
                       "display.max_columns", None):
    print(top[cols].to_string(index=False))

# ── Top divergences by |ΔY·p| alone ─────────────────────────────────────────
top2 = m.sort_values("abs_dY_dim", ascending=False).head(20)
print("\n── Top 20 lines by |ΔY·p|  (largest absolute Y disagreement, any S) ──")
print(top2[cols + ["abs_dY_dim"]].to_string(index=False))

# ── Iso-1 (CO2-626, the dominant absorber) breakdown ────────────────────────
m1 = m[m["iso"] == 1].sort_values("nu_cm")
print(f"\n── CO2-626 (iso 1) matched lines: {len(m1)} ──")
print(f"  |ΔY_per_atm| mean = {m1['dY_per_atm'].abs().mean():.4e}")
print(f"  |ΔY_per_atm| max  = {m1['dY_per_atm'].abs().max():.4e}")
print(f"  Total S·|ΔY·p| iso 1 = {m1['S_x_abs_dY'].sum():.3e}")
print(f"  Total S·|ΔY·p| all   = {m['S_x_abs_dY'].sum():.3e}")
print(f"  Iso 1 fraction of S·|ΔY·p| total: {m1['S_x_abs_dY'].sum()/m['S_x_abs_dY'].sum()*100:.1f}%")

print(f"\n── Iso-1 lines with biggest |ΔY·p| ──")
print(m1.sort_values("abs_dY_dim", ascending=False).head(20)[
    cols + ["abs_dY_dim"]].to_string(index=False))

# ── Save full matched table ─────────────────────────────────────────────────
m.to_csv(OUT, index=False)
print(f"\nWrote {len(m)} matched rows → {OUT}")
