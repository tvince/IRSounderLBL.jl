"""
Compare Julia VP_W iso-1-only vs Julia VP_W iso-1+iso-2 over 645-800 cm⁻¹,
with ARTS+LM as reference.

Inputs:
  data/arts_bt_co2_15um_lm.csv          — ARTS, continuum + LM
  data/julia_bt_co2_15um_vpw.csv        — Julia VP_W, top-5 iso-1 bands
  data/julia_bt_co2_15um_vpw_iso12.csv  — Julia VP_W, top-5 iso-1 + top-5 iso-2 bands

Panels:
  1. BT overlay (ARTS, iso1, iso12)
  2. iso12 − iso1   (the iso-2 contribution to VP_W)
  3. Residuals vs ARTS: iso1 − ARTS and iso12 − ARTS
  4. |iso12 − ARTS| − |iso1 − ARTS|   (negative ⇒ iso12 closer to ARTS)
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

nu_arts,  bt_arts  = load_csv("arts_bt_co2_15um_lm.csv")
nu_iso1,  bt_iso1  = load_csv("julia_bt_co2_15um_vpw.csv")
nu_iso12, bt_iso12 = load_csv("julia_bt_co2_15um_vpw_iso12.csv")
assert np.allclose(nu_arts, nu_iso1) and np.allclose(nu_arts, nu_iso12)
nu = nu_arts

d_iso  = bt_iso12 - bt_iso1
d1_ar  = bt_iso1  - bt_arts
d12_ar = bt_iso12 - bt_arts
closer = np.abs(d12_ar) - np.abs(d1_ar)   # negative => iso12 closer to ARTS

rms_iso  = np.sqrt((d_iso**2).mean())
rms_1ar  = np.sqrt((d1_ar**2).mean())
rms_12ar = np.sqrt((d12_ar**2).mean())

fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

ax = axes[0]
ax.plot(nu, bt_arts,  lw=0.6, color="#444", label="ARTS+LM")
ax.plot(nu, bt_iso1,  lw=0.6, color="#1f77b4", alpha=0.85, label="Julia VP_W (iso-1 top-5)")
ax.plot(nu, bt_iso12, lw=0.6, color="#d62728", alpha=0.7,  label="Julia VP_W (iso-1+iso-2 top-5)")
ax.set_ylabel("BT (K)")
ax.set_title("Julia VP_W iso-1 vs iso-1+iso-2 vs ARTS, 645-800 cm⁻¹\n"
             f"iso1 vs ARTS RMS {rms_1ar:.3f} K   "
             f"iso12 vs ARTS RMS {rms_12ar:.3f} K   "
             f"iso12 − iso1 RMS {rms_iso:.4f} K")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(0, color="k", lw=0.5)
ax.plot(nu, d_iso, lw=0.6, color="#d62728", label="iso12 − iso1")
ax.set_ylabel("ΔBT (K)")
ax.set_ylim(-0.15, 0.15)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.axhline(0, color="k", lw=0.5)
ax.plot(nu, d1_ar,  lw=0.6, color="#1f77b4", alpha=0.85, label=f"iso1 − ARTS   RMS={rms_1ar:.3f}")
ax.plot(nu, d12_ar, lw=0.6, color="#d62728", alpha=0.7,  label=f"iso12 − ARTS  RMS={rms_12ar:.3f}")
ax.set_ylabel("Julia − ARTS (K)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[3]
ax.axhline(0, color="k", lw=0.5)
pos = closer > 0
ax.fill_between(nu, 0, closer, where=~pos, color="green", alpha=0.5, label="iso12 closer to ARTS")
ax.fill_between(nu, 0, closer, where= pos, color="red",   alpha=0.5, label="iso12 farther from ARTS")
ax.set_ylabel("|iso12 − ARTS| − |iso1 − ARTS| (K)")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

out = os.path.join(DATA_DIR, "vpw_iso12_vs_iso1.png")
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"saved → {out}")
print(f"iso12 − iso1: bias {d_iso.mean():+.5f} K, RMS {rms_iso:.5f} K, max|Δ| {np.abs(d_iso).max():.4f} K at ν={nu[np.abs(d_iso).argmax()]:.2f}")
