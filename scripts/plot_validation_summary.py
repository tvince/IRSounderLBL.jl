#!/usr/bin/env python3
"""
Summary plot of the post-pedestal-fix LBLRTM validation (branch
fix-voigt-parabolic-pedestal). Julia (:cim default, AER parabolic pedestal) vs
LBLRTM for the 15 um and 4.3 um CO2 bands.

Layout:
  row 1: spectral overlays (15 um | 4.3 um), Julia vs LBLRTM
  row 2: difference panels (Julia - LBLRTM), band head highlighted
  row 3: before/after RMS bars showing what the parabolic pedestal fix bought

Run:  python scripts/plot_validation_summary.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D = "data/lblrtm"
EDGE = 0.5          # cm-1 trimmed from each band edge before stats
BH = (2386, 2398)   # 4.3 um band head


def load_j(p):
    a = np.loadtxt(f"{D}/{p}", delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def load_l(p):
    a = np.loadtxt(f"{D}/{p}", delimiter=",", skiprows=1, usecols=(0, 3))
    return a[:, 0], a[:, 1]


def diff_on_julia(nuj, btj, nul, btl):
    """LBLRTM interpolated onto Julia grid; return trimmed nu, dBT."""
    lo = max(nuj.min(), nul.min()) + EDGE
    hi = min(nuj.max(), nul.max()) - EDGE
    m = (nuj >= lo) & (nuj <= hi)
    nu = nuj[m]
    d = btj[m] - np.interp(nu, nul, btl)
    return nu, d


# ---- data (post-fix Julia, regenerated 2026-06-06 evening) ----
nuj15, bt15 = load_j("julia_bt_contOFF.csv")
nul15, btl15 = load_l("lblrtm_bt_contOFF_g12.csv")
nuj43, bt43 = load_j("julia_bt_43um_contOFF.csv")
nul43, btl43 = load_l("lblrtm_bt_43um_contOFF.csv")

nu15, d15 = diff_on_julia(nuj15, bt15, nul15, btl15)
nu43, d43 = diff_on_julia(nuj43, bt43, nul43, btl43)

rms15 = np.sqrt(np.mean(d15 ** 2))
rms43 = np.sqrt(np.mean(d43 ** 2))
mbh = (nu43 >= BH[0]) & (nu43 <= BH[1])
rms_bh = np.sqrt(np.mean(d43[mbh] ** 2))

# ---- figure ----
fig = plt.figure(figsize=(14, 11))
gs = GridSpec(3, 2, height_ratios=[1.0, 0.85, 0.9], hspace=0.32, wspace=0.18,
              top=0.93, bottom=0.07, left=0.07, right=0.97)

C_J, C_L, C_D = "#1f77b4", "#d62728", "#2ca02c"

fig.suptitle("LBLRTM validation — Julia :cim + AER parabolic pedestal (2026-06-06)",
             fontsize=15, fontweight="bold")

# --- row 1: overlays ---
ax = fig.add_subplot(gs[0, 0])
ax.plot(nul15, btl15, lw=0.5, color=C_L, alpha=0.85, label="LBLRTM")
ax.plot(nuj15, bt15, lw=0.5, color=C_J, alpha=0.85, label="Julia (:cim)")
ax.set_xlim(645, 800)
ax.set_title("15 um CO2 band (cont-OFF)")
ax.set_ylabel("BT (K)")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[0, 1])
ax.plot(nul43, btl43, lw=0.4, color=C_L, alpha=0.85, label="LBLRTM")
ax.plot(nuj43, bt43, lw=0.4, color=C_J, alpha=0.85, label="Julia (:cim)")
ax.axvspan(*BH, color="orange", alpha=0.18, label="band head")
ax.set_xlim(2000, 2500)
ax.set_title("4.3 um CO2 band (cont-OFF)")
ax.set_ylabel("BT (K)")
ax.legend(loc="lower left", fontsize=8)

# --- row 2: differences ---
ax = fig.add_subplot(gs[1, 0])
ax.axhline(0, color="k", lw=0.5)
ax.plot(nu15, d15, lw=0.4, color="#9467bd")
ax.set_xlim(645, 800)
ax.set_ylim(-1, 1)
ax.set_title(f"15 um  Julia - LBLRTM   |   RMS {rms15:.4f} K")
ax.set_xlabel("wavenumber (cm$^{-1}$)")
ax.set_ylabel("$\\Delta$BT (K)")

ax = fig.add_subplot(gs[1, 1])
ax.axhline(0, color="k", lw=0.5)
ax.axvspan(*BH, color="orange", alpha=0.18)
ax.plot(nu43, d43, lw=0.4, color=C_D)
ax.set_xlim(2000, 2500)
ax.set_ylim(-1, 1)
ax.set_title(f"4.3 um  Julia - LBLRTM   |   RMS {rms43:.4f} K   "
             f"(band head {rms_bh:.4f} K)")
ax.set_xlabel("wavenumber (cm$^{-1}$)")
ax.set_ylabel("$\\Delta$BT (K)")

# --- row 3: before/after RMS bars (the win from the parabolic pedestal fix) ---
ax = fig.add_subplot(gs[2, :])
labels = ["15 um band", "4.3 um full", "4.3 um band head"]
before = [0.075, 0.67, 4.36]   # pre-fix (from project notes)
after = [rms15, rms43, rms_bh]  # measured this run
x = np.arange(len(labels))
w = 0.36
b1 = ax.bar(x - w / 2, before, w, color="#bbbbbb", label="before fix")
b2 = ax.bar(x + w / 2, after, w, color=C_J, label="after fix (measured)")
ax.set_yscale("log")
ax.set_ylim(0.02, 8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("RMS $\\Delta$BT vs LBLRTM (K, log)")
ax.set_title("AER parabolic pedestal fix: before vs after")
ax.legend(loc="upper left", fontsize=9)
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8)

out = f"{D}/lblrtm_validation_summary.png"
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"15um RMS {rms15:.4f} K | 4.3um RMS {rms43:.4f} K | band head {rms_bh:.4f} K")
