"""
Characterise the CO2 line mixing effect and the Julia VP_Y gap (645–800 cm⁻¹).

Loads:
  data/arts_bt_iasi_cont.csv      — ARTS, continuum, no LM  (full spectrum)
  data/arts_bt_co2_15um_lm.csv   — ARTS, continuum + VP_Y  (645–800 cm⁻¹)
  data/julia_bt_cont.csv          — Julia, continuum, no LM (full spectrum)
  data/julia_bt_co2_15um_lm.csv  — Julia, continuum + VP_Y (645–800 cm⁻¹)

Panels:
  1. BT spectra (ARTS VP_Y and Julia VP_Y)
  2. LM effect in ARTS: (cont+LM) − cont
  3. Julia VP_Y gap vs ARTS VP_Y: Julia-LM − ARTS-LM
  4. Residual improvement: (Julia-cont − ARTS-LM) vs (Julia-LM − ARTS-LM)

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/compare_lm_effect.py
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
nu_lm,  bt_arts_lm    = load_csv("arts_bt_co2_15um_lm.csv")    # ARTS VP_Y (reference grid)
nu_ref, bt_arts_cont  = load_csv("arts_bt_iasi_cont.csv")       # ARTS no-LM (full spectrum)
_,      bt_julia_cont = load_csv("julia_bt_cont.csv")           # Julia no-LM (full spectrum)
nu_jlm, bt_julia_lm   = load_csv("julia_bt_co2_15um_lm.csv")   # Julia VP_Y

# Use ARTS VP_Y grid as reference
nu = nu_lm

# Interpolate full-spectrum CSVs onto 645–800 grid
bt_arts_cont_r  = np.interp(nu, nu_ref, bt_arts_cont)
bt_julia_cont_r = np.interp(nu, nu_ref, bt_julia_cont)
bt_julia_lm_r   = np.interp(nu, nu_jlm, bt_julia_lm)

# LM effect in ARTS: (cont+LM) − cont
lm_effect = bt_arts_lm - bt_arts_cont_r

# Julia gap WITHOUT LM vs ARTS with LM
gap_no_lm = bt_julia_cont_r - bt_arts_lm

# Julia gap WITH LM vs ARTS with LM
gap_lm    = bt_julia_lm_r - bt_arts_lm

print(f"\n--- Line mixing effect in ARTS (cont+LM − cont) ---")
print(f"Mean: {lm_effect.mean():+.3f} K   RMS: {np.sqrt((lm_effect**2).mean()):.3f} K")
print(f"Max warming: {lm_effect.max():+.3f} K @ {nu[np.argmax(lm_effect)]:.2f} cm⁻¹")
print(f"Max cooling: {lm_effect.min():+.3f} K @ {nu[np.argmin(lm_effect)]:.2f} cm⁻¹")

def stats(arr, label):
    bias = arr.mean()
    rms  = np.sqrt((arr**2).mean())
    amax = np.max(np.abs(arr))
    imax = np.argmax(np.abs(arr))
    pct1 = 100.0 * np.sum(np.abs(arr) <= 1.0) / len(arr)
    print(f"  Bias: {bias:+.3f} K   RMS: {rms:.3f} K   Max|Δ|: {amax:.3f} K @ {nu[imax]:.2f} cm⁻¹   {pct1:.1f}% within 1 K")
    return bias, rms

print(f"\n--- Julia-cont vs ARTS+LM (before adding Julia LM) ---")
b1, r1 = stats(gap_no_lm, "no_lm")
print(f"\n--- Julia-LM  vs ARTS+LM  (Julia VP_Y implementation) ---")
b2, r2 = stats(gap_lm, "lm")
print(f"\nRMS improvement: {r1:.3f} → {r2:.3f} K ({100*(r1-r2)/r1:.1f}% reduction)")

# Sub-region breakdown
REGIONS = [
    ("CO₂ P-branch", 645, 655),
    ("CO₂ Q-branch", 655, 675),
    ("CO₂ R-branch", 675, 720),
    ("hot bands",    720, 800),
]
print(f"\n--- Sub-region breakdown ---")
for label, w1, w2 in REGIONS:
    mask = (nu >= w1) & (nu <= w2)
    if mask.sum() == 0:
        continue
    gap_r_before = gap_no_lm[mask]
    gap_r_after  = gap_lm[mask]
    print(f"  {label:16s} ({w1:.0f}–{w2:.0f}): "
          f"before={gap_r_before.mean():+.3f} K  after={gap_r_after.mean():+.3f} K  "
          f"RMS before={np.sqrt((gap_r_before**2).mean()):.3f}  after={np.sqrt((gap_r_after**2).mean()):.3f}")

BANDS = [(655, "P"), (667, "Q"), (675, "R"), (720, "hot")]

fig, axes = plt.subplots(4, 1, figsize=(12, 13),
                         gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1.2]})
fig.suptitle(
    f"CO₂ 15µm line mixing — Julia VP_Y vs ARTS VP_Y, 645–800 cm⁻¹\n"
    f"Julia-LM vs ARTS-LM: bias {b2:+.3f} K   RMS {r2:.3f} K"
    f"  (before LM: {r1:.3f} K RMS)",
    fontsize=11,
)

ax = axes[0]
ax.plot(nu, bt_arts_lm,    lw=0.6, color="steelblue", label="ARTS VP_Y",  alpha=0.9)
ax.plot(nu, bt_julia_lm_r, lw=0.6, color="tomato",    label="Julia VP_Y", alpha=0.8)
ax.set_ylabel("BT (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[1]
ax.axhline(0, color="k", lw=0.6)
ax.plot(nu, lm_effect, lw=0.6, color="steelblue", label="ARTS LM effect (cont+LM − cont)")
ax.set_ylabel("LM effect (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[2]
ax.axhline(0, color="k", lw=0.6)
ax.plot(nu, gap_no_lm, lw=0.5, color="gray",      alpha=0.7, label="Julia-cont − ARTS-LM")
ax.plot(nu, gap_lm,    lw=0.6, color="darkgreen",             label="Julia-LM  − ARTS-LM")
ax.axhline( 1.0, color="gray", lw=0.5, ls="--")
ax.axhline(-1.0, color="gray", lw=0.5, ls="--")
ax.set_ylabel("Julia − ARTS+LM (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[3]
improvement = np.abs(gap_no_lm) - np.abs(gap_lm)
ax.axhline(0, color="k", lw=0.6)
ax.fill_between(nu, 0, improvement,
                where=improvement > 0, color="green", alpha=0.4, label="Improvement")
ax.fill_between(nu, 0, improvement,
                where=improvement < 0, color="red",   alpha=0.4, label="Regression")
ax.set_ylabel("|gap| reduction (K)")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

for ax in axes:
    for wn, label in BANDS:
        ax.axvline(wn, color="k", lw=0.4, ls=":", alpha=0.5)
        ylim = ax.get_ylim()
        ax.text(wn, ylim[0] + 0.96*(ylim[1]-ylim[0]), label,
                ha="center", va="top", fontsize=7, alpha=0.6)

plt.tight_layout()
out = os.path.join(DATA_DIR, "arts_julia_lm_comparison.png")
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
