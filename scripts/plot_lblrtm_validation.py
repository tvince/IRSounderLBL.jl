#!/usr/bin/env python3
"""
Plot the single-session LBLRTM validation: Julia (:cim default) vs LBLRTM for the
15 um and 4.3 um CO2 bands, with difference panels. Highlights the 2386-2398
band-head region that dominates the 4.3 um residual.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "data/lblrtm"

def load_j(p):
    a = np.loadtxt(f"{D}/{p}", delimiter=",", skiprows=1); return a[:, 0], a[:, 1]
def load_l(p):
    a = np.loadtxt(f"{D}/{p}", delimiter=",", skiprows=1, usecols=(0, 3)); return a[:, 0], a[:, 1]

# data
nuj15, bt15 = load_j("julia_bt_contOFF.csv");      nul15, btl15 = load_l("lblrtm_bt_contOFF_g12.csv")
nuj43, bt43 = load_j("julia_bt_43um_contOFF.csv"); nul43, btl43 = load_l("lblrtm_bt_43um_contOFF.csv")
d15 = bt15 - np.interp(nuj15, nul15, btl15)
d43 = bt43 - np.interp(nuj43, nul43, btl43)

fig, ax = plt.subplots(3, 1, figsize=(13, 10))

# 4.3 um overlay
ax[0].plot(nul43, btl43, lw=0.4, color="#d62728", label="LBLRTM", alpha=0.8)
ax[0].plot(nuj43, bt43, lw=0.4, color="#1f77b4", label="Julia (:cim)", alpha=0.8)
ax[0].axvspan(2386, 2398, color="orange", alpha=0.15, label="band head 2386-2398")
ax[0].set_title("4.3 um CO2 band (cont-OFF): Julia :cim vs LBLRTM")
ax[0].set_ylabel("BT (K)"); ax[0].legend(loc="lower left", fontsize=8)
ax[0].set_xlim(2000, 2500)

# 4.3 um diff
ax[1].axhline(0, color="k", lw=0.5)
ax[1].axvspan(2386, 2398, color="orange", alpha=0.15)
ax[1].plot(nuj43, d43, lw=0.4, color="#2ca02c")
bh = (nuj43 >= 2386) & (nuj43 <= 2398)
ax[1].set_title(f"4.3 um difference (Julia - LBLRTM)  |  full RMS {np.sqrt(np.mean(d43**2)):.3f} K, "
                f"band head 2386-2398 RMS {np.sqrt(np.mean(d43[bh]**2)):.3f} K")
ax[1].set_ylabel("dBT (K)"); ax[1].set_xlim(2000, 2500)
ax[1].set_ylim(-1.0, 1.0)

# 15 um diff
ax[2].axhline(0, color="k", lw=0.5)
ax[2].plot(nuj15, d15, lw=0.4, color="#9467bd")
ax[2].set_title(f"15 um difference (Julia - LBLRTM, cont-OFF)  |  RMS {np.sqrt(np.mean(d15**2)):.3f} K "
                f"(no regression from :cim)")
ax[2].set_xlabel("wavenumber (cm-1)"); ax[2].set_ylabel("dBT (K)"); ax[2].set_xlim(645, 800)

fig.tight_layout()
out = f"{D}/lblrtm_validation.png"
fig.savefig(out, dpi=130)
print("wrote", out)
