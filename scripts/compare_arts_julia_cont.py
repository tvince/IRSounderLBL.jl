"""
Compare ARTS and Julia BT spectra: continuum vs no-continuum.

Loads four CSV files:
  data/arts_bt_iasi.csv        — ARTS, no continuum (baseline)
  data/arts_bt_iasi_cont.csv   — ARTS, with H2O MT-CKD continuum
  data/julia_bt_645_800.csv    — Julia, no continuum (baseline)
  data/julia_bt_cont.csv       — Julia, with H2O MT-CKD continuum

Panels:
  1. BT spectra (ARTS-cont and Julia-cont)
  2. Julia-cont − ARTS-cont  (forward model comparison with continuum)
  3. Continuum effect: (cont − no-cont) for ARTS and Julia

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/compare_arts_julia_cont.py
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

def load_csv(fname):
    nu, bt = [], []
    with open(os.path.join(DATA_DIR, fname)) as f:
        for row in csv.DictReader(f):
            nu.append(float(row["nu_cm1"]))
            bt.append(float(row["BT_K"]))
    return np.array(nu), np.array(bt)

print("Loading CSV files...")
nu_arts,      bt_arts      = load_csv("arts_bt_iasi.csv")
nu_arts_cont, bt_arts_cont = load_csv("arts_bt_iasi_cont.csv")
nu_jul,       bt_jul       = load_csv("julia_bt_645_800.csv")
nu_jul_cont,  bt_jul_cont  = load_csv("julia_bt_cont.csv")

# Interpolate everything onto the ARTS no-continuum grid
ref_nu = nu_arts
bt_ac  = np.interp(ref_nu, nu_arts_cont, bt_arts_cont)   # ARTS cont
bt_jc  = np.interp(ref_nu, nu_jul_cont,  bt_jul_cont)    # Julia cont
bt_j0  = np.interp(ref_nu, nu_jul,       bt_jul)         # Julia no-cont

# Panel 2: Julia-cont minus ARTS-cont
diff_cont = bt_jc - bt_ac
bias_c = diff_cont.mean()
rms_c  = np.sqrt((diff_cont**2).mean())
amax_c = np.max(np.abs(diff_cont))
imax_c = np.argmax(np.abs(diff_cont))
pct1_c = 100.0 * np.sum(np.abs(diff_cont) <= 1.0) / len(diff_cont)

# Continuum effects
eff_arts  = bt_ac - bt_arts    # ARTS: cont − no-cont
eff_julia = bt_jc - bt_j0     # Julia: cont − no-cont

print(f"\n--- Julia-cont vs ARTS-cont ---")
print(f"Bias: {bias_c:+.3f} K   RMS: {rms_c:.3f} K   Max |ΔBT|: {amax_c:.3f} K @ {ref_nu[imax_c]:.2f} cm⁻¹")
print(f"Fraction within 1.0 K: {pct1_c:.1f}%")

print(f"\n--- Continuum effect (cont − no-cont) ---")
print(f"ARTS  : mean {eff_arts.mean():+.3f} K   max cooling {eff_arts.min():+.3f} K @ {ref_nu[np.argmin(eff_arts)]:.2f} cm⁻¹")
print(f"Julia : mean {eff_julia.mean():+.3f} K   max cooling {eff_julia.min():+.3f} K @ {ref_nu[np.argmin(eff_julia)]:.2f} cm⁻¹")
print(f"ARTS−Julia continuum effect diff: mean {(eff_arts - eff_julia).mean():+.3f} K   RMS {np.sqrt(((eff_arts - eff_julia)**2).mean()):.3f} K")

BANDS = [
    (667,  "CO₂\n15µm"),
    (1040, "O₃\n9.6µm"),
    (1300, "N₂O/\nCH₄"),
    (1600, "H₂O\n6.3µm"),
    (2140, "CO\n4.7µm"),
    (2350, "CO₂\n4.3µm"),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                         gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]})
fig.suptitle(
    f"Julia vs ARTS — with H2O MT-CKD 3.50 continuum\n"
    f"Bias = {bias_c:+.3f} K   RMS = {rms_c:.3f} K   "
    f"Max |ΔBT| = {amax_c:.3f} K   {pct1_c:.1f}% within 1 K",
    fontsize=11,
)

# Panel 1: BT spectra (continuum runs)
ax = axes[0]
ax.plot(ref_nu, bt_ac,  lw=0.5, color="steelblue", label="ARTS 2.6 + continuum", alpha=0.8)
ax.plot(ref_nu, bt_jc,  lw=0.5, color="tomato",    label="Julia + continuum",    alpha=0.8)
ax.set_ylabel("BT (K)")
ax.set_xlim(ref_nu[0], ref_nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

# Panel 2: Julia-cont − ARTS-cont
ax = axes[1]
ax.axhline(0, color="k", lw=0.6)
ax.plot(ref_nu, diff_cont, lw=0.4, color="darkgreen")
ax.axhline( 1.0, color="gray", lw=0.6, ls="--")
ax.axhline(-1.0, color="gray", lw=0.6, ls="--")
ax.set_ylabel("Julia − ARTS (K)")
ax.set_xlim(ref_nu[0], ref_nu[-1])
ax.grid(True, lw=0.3, alpha=0.4)

# Panel 3: continuum effect on each model
ax = axes[2]
ax.axhline(0, color="k", lw=0.6)
ax.plot(ref_nu, eff_arts,  lw=0.5, color="steelblue", alpha=0.8, label="ARTS: cont − no-cont")
ax.plot(ref_nu, eff_julia, lw=0.5, color="tomato",    alpha=0.8, label="Julia: cont − no-cont")
ax.set_ylabel("Continuum effect (K)")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_xlim(ref_nu[0], ref_nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

# Band labels on panels 1 and 2
for ax in axes[:2]:
    for wn, label in BANDS:
        if ref_nu[0] <= wn <= ref_nu[-1]:
            ax.axvline(wn, color="k", lw=0.4, ls=":", alpha=0.5)
            ax.text(wn, ax.get_ylim()[1] * 0.98, label,
                    ha="center", va="top", fontsize=6, alpha=0.6)

plt.tight_layout()
out = os.path.join(DATA_DIR, "arts_julia_cont_diff.png")
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
