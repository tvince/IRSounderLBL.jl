"""
Compare ARTS and Julia BT spectra.

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/compare_arts_julia.py
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

print("Loading...")
nu_arts, bt_arts = load_csv("arts_bt_iasi.csv")
nu_jul,  bt_jul  = load_csv("julia_bt_645_800.csv")

# Interpolate Julia onto ARTS grid (both are 0.25 cm⁻¹ but may not be bit-identical)
bt_jul_i = np.interp(nu_arts, nu_jul, bt_jul)
diff     = bt_jul_i - bt_arts

bias = diff.mean()
rms  = np.sqrt((diff**2).mean())
amax = np.max(np.abs(diff))
imax = np.argmax(np.abs(diff))
pct1 = 100.0 * np.sum(np.abs(diff) <= 1.0) / len(diff)

print(f"Bias: {bias:+.3f} K   RMS: {rms:.3f} K   Max |ΔBT|: {amax:.3f} K @ {nu_arts[imax]:.2f} cm⁻¹")
print(f"Fraction within 1.0 K: {pct1:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                         gridspec_kw={"height_ratios": [2.5, 1.5, 1.5]})
fig.suptitle(
    f"Julia vs ARTS — pure LBL, no continuum, no ILS\n"
    f"Bias = {bias:+.3f} K   RMS = {rms:.3f} K   Max |ΔBT| = {amax:.3f} K   {pct1:.1f}% within 1 K",
    fontsize=11,
)

# Panel 1: BT spectra
ax = axes[0]
ax.plot(nu_arts, bt_arts, lw=0.5, color="steelblue", label="ARTS 2.6", alpha=0.8)
ax.plot(nu_arts, bt_jul_i, lw=0.5, color="tomato",    label="Julia",    alpha=0.8)
ax.set_ylabel("BT (K)")
ax.set_xlim(nu_arts[0], nu_arts[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

# Panel 2: difference (Julia − ARTS)
ax = axes[1]
ax.axhline(0, color="k", lw=0.6)
ax.plot(nu_arts, diff, lw=0.4, color="darkgreen")
ax.axhline( 1.0, color="gray", lw=0.6, ls="--")
ax.axhline(-1.0, color="gray", lw=0.6, ls="--")
ax.set_ylabel("Julia − ARTS (K)")
ax.set_xlim(nu_arts[0], nu_arts[-1])
ax.grid(True, lw=0.3, alpha=0.4)

# Panel 3: |difference| on log scale
ax = axes[2]
ax.semilogy(nu_arts, np.abs(diff) + 1e-5, lw=0.4, color="purple")
ax.axhline(1.0, color="gray", lw=0.6, ls="--")
ax.set_ylabel("|Julia − ARTS| (K)")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_xlim(nu_arts[0], nu_arts[-1])
ax.grid(True, lw=0.3, alpha=0.4)

# Band labels on all panels
BANDS = [
    (667,  "CO₂\n15µm"),
    (1040, "O₃\n9.6µm"),
    (1300, "N₂O/\nCH₄"),
    (1600, "H₂O\n6.3µm"),
    (2140, "CO\n4.7µm"),
    (2350, "CO₂\n4.3µm"),
]
for ax in axes[:2]:
    for wn, label in BANDS:
        if nu_arts[0] <= wn <= nu_arts[-1]:
            ax.axvline(wn, color="k", lw=0.4, ls=":", alpha=0.5)
            ax.text(wn, ax.get_ylim()[1] * 0.98, label,
                    ha="center", va="top", fontsize=6, alpha=0.6)

plt.tight_layout()
out = os.path.join(DATA_DIR, "arts_julia_full_iasi_diff.png")
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
