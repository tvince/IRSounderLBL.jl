"""
Compare Q(T) from ARTS built-in tables vs TIPS-2024 (Julia) for H2O and CO2.

ARTS tables were written by WriteBuiltinPartitionFunctionsXML to data/arts_pf/.
Julia values are read from data/tips2024_qt.csv produced by scripts/dump_tips2024.jl.

Run order:
  1. julia --project scripts/dump_tips2024.jl   (writes data/tips2024_qt.csv)
  2. python scripts/compare_partition_functions.py
"""

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF_DIR  = os.path.join(ROOT, "data", "arts_pf")
JULIA_CSV = os.path.join(ROOT, "data", "tips2024_qt.csv")
OUT_PNG = os.path.join(ROOT, "data", "partition_function_comparison.png")

# ── Mapping: (species_label, hitran_mol, hitran_iso) -> ARTS filename stem ──────
SPECIES = [
    ("H2O iso1", 1, 1, "H2O-161"),
    ("H2O iso2", 1, 2, "H2O-181"),
    ("H2O iso3", 1, 3, "H2O-171"),
    ("CO2 iso1", 2, 1, "CO2-626"),
    ("CO2 iso2", 2, 2, "CO2-636"),
    ("CO2 iso3", 2, 3, "CO2-628"),
]


def read_arts_pf(stem):
    """Parse ARTS partition function XML (two-column: T, Q)."""
    path = os.path.join(PF_DIR, f"{stem}.xml")
    T_vals, Q_vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) == 2:
                try:
                    T_vals.append(float(parts[0]))
                    Q_vals.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(T_vals), np.array(Q_vals)


def interp_arts(T_vals, Q_vals, T_query):
    return np.interp(T_query, T_vals, Q_vals)


# ── Load Julia TIPS-2024 CSV ───────────────────────────────────────────────────
julia_df = pd.read_csv(JULIA_CSV)
# columns: T, mol_id, iso_id, Q

T_QUERY = np.arange(150, 401, 1, dtype=float)

# ── Build comparison ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

print(f"{'Species':10s}  {'T_ref':>6s}  {'ARTS Q296':>10s}  {'TIPS Q296':>10s}  "
      f"{'Max |ratio-1|%':>16s}  {'RMS |ratio-1|%':>16s}")

T_REF = 296.0

for ax, (label, mol, iso, arts_stem) in zip(axes, SPECIES):
    # ARTS
    T_a, Q_a = read_arts_pf(arts_stem)
    Q_arts = interp_arts(T_a, Q_a, T_QUERY)

    # Julia TIPS-2024
    mask = (julia_df["mol_id"] == mol) & (julia_df["iso_id"] == iso)
    sub  = julia_df[mask].sort_values("T")
    Q_julia = np.interp(T_QUERY, sub["T"].values, sub["Q"].values)

    ratio = Q_arts / Q_julia   # 1.0 => perfect agreement
    pct   = (ratio - 1.0) * 100.0

    Q296_arts  = float(interp_arts(T_a, Q_a, T_REF))
    Q296_julia = float(np.interp(T_REF, sub["T"].values, sub["Q"].values))

    print(f"{label:10s}  {T_REF:6.0f}  {Q296_arts:10.4f}  {Q296_julia:10.4f}  "
          f"{np.max(np.abs(pct)):16.4f}  {np.sqrt(np.mean(pct**2)):16.4f}")

    ax.plot(T_QUERY, pct, color="steelblue", lw=1.5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"{label} ({arts_stem})")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Q_ARTS/Q_TIPS2024 − 1  (%)")
    ax.set_xlim(150, 400)
    ax.grid(True, alpha=0.3)

plt.suptitle("Partition function comparison: ARTS built-in vs TIPS-2024\n(positive = ARTS larger)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nSaved plot → {OUT_PNG}")
