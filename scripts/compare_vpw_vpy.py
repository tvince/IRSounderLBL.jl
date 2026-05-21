"""
Compare Julia VP_W (full-matrix LM) and VP_Y (Rosenkranz first-order) over the
CO₂ 15µm region (645–800 cm⁻¹), with ARTS+LM as a reference.

Inputs:
  data/arts_bt_co2_15um_lm.csv  — ARTS, continuum + LM
  data/julia_bt_co2_15um_lm.csv — Julia, continuum + VP_Y
  data/julia_bt_co2_15um_vpw.csv — Julia, continuum + VP_W (top-5 bands)

Panels:
  1. BT spectra (VP_Y vs VP_W overlay)
  2. VP_W − VP_Y (LM-method difference)
  3. Residuals: VP_Y − ARTS and VP_W − ARTS
  4. |VP_W − ARTS| − |VP_Y − ARTS|  (negative = VP_W closer to ARTS)
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

print("Loading CSVs …")
nu,        bt_arts = load_csv("arts_bt_co2_15um_lm.csv")
_,         bt_vpy  = load_csv("julia_bt_co2_15um_lm.csv")
_,         bt_vpw  = load_csv("julia_bt_co2_15um_vpw.csv")

d_method  = bt_vpw - bt_vpy
res_vpy   = bt_vpy - bt_arts
res_vpw   = bt_vpw - bt_arts
delta_abs = np.abs(res_vpw) - np.abs(res_vpy)   # negative = VP_W closer to ARTS

def rms(x):
    return float(np.sqrt(np.mean(x**2)))

print(f"\nVP_W − VP_Y over 645-800 cm⁻¹:")
print(f"  RMS={rms(d_method):.3f} K   max|Δ|={np.max(np.abs(d_method)):.3f} K @ {nu[np.argmax(np.abs(d_method))]:.2f}")

print(f"\nResiduals vs ARTS+LM:")
print(f"  VP_Y: bias={res_vpy.mean():+.3f}  RMS={rms(res_vpy):.3f}")
print(f"  VP_W: bias={res_vpw.mean():+.3f}  RMS={rms(res_vpw):.3f}")

REGIONS = [
    ("Q-branch wings 660-675", 660, 675),
    ("R-branch 675-720",       675, 720),
    ("Hot bands 720-800",      720, 800),
]
print(f"\nPer-region RMS vs ARTS:")
for label, lo, hi in REGIONS:
    m = (nu >= lo) & (nu <= hi)
    print(f"  {label:25s}  VP_Y {rms(res_vpy[m]):.3f}   VP_W {rms(res_vpw[m]):.3f}")

BANDS = [(655, "P"), (667, "Q"), (675, "R"), (720, "hot")]

fig, axes = plt.subplots(4, 1, figsize=(12, 13),
                         gridspec_kw={"height_ratios": [2, 1.2, 1.4, 1.2]})
fig.suptitle(
    f"Julia VP_W vs VP_Y vs ARTS, 645–800 cm⁻¹\n"
    f"VP_Y RMS {rms(res_vpy):.3f} K   VP_W RMS {rms(res_vpw):.3f} K   "
    f"VP_W−VP_Y RMS {rms(d_method):.3f} K",
    fontsize=11,
)

ax = axes[0]
ax.plot(nu, bt_arts, lw=0.6, color="steelblue", alpha=0.85, label="ARTS+LM")
ax.plot(nu, bt_vpy,  lw=0.6, color="tomato",    alpha=0.8,  label="Julia VP_Y")
ax.plot(nu, bt_vpw,  lw=0.6, color="darkgreen", alpha=0.7,  label="Julia VP_W (top-5)")
ax.set_ylabel("BT (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[1]
ax.axhline(0, color="k", lw=0.6)
ax.plot(nu, d_method, lw=0.6, color="purple", label="VP_W − VP_Y")
ax.set_ylabel("ΔBT (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[2]
ax.axhline(0, color="k", lw=0.6)
ax.axhline( 1.0, color="gray", lw=0.5, ls="--")
ax.axhline(-1.0, color="gray", lw=0.5, ls="--")
ax.plot(nu, res_vpy, lw=0.5, color="tomato",    alpha=0.8, label=f"VP_Y − ARTS   RMS={rms(res_vpy):.3f} K")
ax.plot(nu, res_vpw, lw=0.5, color="darkgreen", alpha=0.7, label=f"VP_W − ARTS   RMS={rms(res_vpw):.3f} K")
ax.set_ylabel("Julia − ARTS (K)")
ax.set_xlim(nu[0], nu[-1])
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, lw=0.3, alpha=0.4)

ax = axes[3]
ax.axhline(0, color="k", lw=0.6)
ax.fill_between(nu, 0, -delta_abs,
                where=-delta_abs > 0, color="green", alpha=0.4, label="VP_W closer to ARTS")
ax.fill_between(nu, 0, -delta_abs,
                where=-delta_abs < 0, color="red",   alpha=0.4, label="VP_W farther from ARTS")
ax.set_ylabel("|VP_Y − ARTS| − |VP_W − ARTS| (K)")
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
out = os.path.join(DATA_DIR, "vpw_vs_vpy_comparison.png")
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
