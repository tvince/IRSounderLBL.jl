#!/usr/bin/env python3
"""
Spectral-grid convergence of the ILS-convolved BT in the dense 4.3 µm window
(2326-2346 cm-1, around the 2335.9 worst point), CO2-only cont-OFF, :cim.

Each internal monochromatic grid is compared to the finest (0.0005 cm-1)
reference after ILS convolution + resampling to the sensor channel grid, for
IASI (ILS FWHM 0.5, Δν 0.25) and IASI-NG (ILS FWHM 0.25, Δν 0.125).

Data from scripts/grid_convergence_iasing.jl.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# internal grid (cm-1); 0.0005 is the reference (ΔBT == 0, omitted from log axis)
dnu = np.array([0.005, 0.0025, 0.001])

# (RMS, max|ΔBT|) vs the 0.0005 reference, 4.3 µm window
iasi    = {"rms": [0.23657, 0.01466, 0.00004], "max": [0.58897, 0.03645, 0.00011]}
iasing  = {"rms": [0.29934, 0.02027, 0.00006], "max": [0.87012, 0.05984, 0.00020]}

DOPPLER_FWHM = 0.0038   # CO2 4.3 µm Doppler-floor FWHM (~220 K)

fig, ax = plt.subplots(figsize=(9, 6.5))

ax.loglog(dnu, iasi["rms"],   "o-",  color="#1f77b4", lw=2, ms=8, label="IASI  RMS")
ax.loglog(dnu, iasi["max"],   "o--", color="#1f77b4", lw=1.3, ms=6, alpha=0.7, label="IASI  max|ΔBT|")
ax.loglog(dnu, iasing["rms"], "s-",  color="#d62728", lw=2, ms=8, label="IASI-NG  RMS")
ax.loglog(dnu, iasing["max"], "s--", color="#d62728", lw=1.3, ms=6, alpha=0.7, label="IASI-NG  max|ΔBT|")

# 0.01 K "sub-noise" target
ax.axhline(0.01, color="green", ls=":", lw=1.5)
ax.text(0.00052, 0.0115, "0.01 K target", color="green", fontsize=9)

# current production validation grid
ax.axvline(0.005, color="gray", ls="-", lw=1, alpha=0.6)
ax.text(0.00505, 1.1, "current 0.005", color="gray", fontsize=9, rotation=90, va="top")

# Doppler-resolution threshold band: ~1 pt/FWHM at Δν≈FWHM
ax.axvspan(DOPPLER_FWHM / 4, DOPPLER_FWHM, color="orange", alpha=0.12)
ax.text(DOPPLER_FWHM * 0.42, 2.5e-5, "Doppler core\nFWHM≈0.0038\n(≈1–4 pts)",
        color="darkorange", fontsize=8, ha="center")

ax.set_xlabel("internal monochromatic grid  Δν  (cm$^{-1}$)")
ax.set_ylabel("ILS-convolved BT error vs 0.0005 cm$^{-1}$ reference  (K)")
ax.set_title("4.3 µm grid convergence (2326–2346 cm$^{-1}$, CO$_2$ cont-OFF)\n"
             "0.005 cm$^{-1}$ is NOT converged for the convolved product; IASI-NG worse than IASI")
ax.set_xlim(0.0006, 0.007)
ax.set_ylim(1e-5, 2)
ax.invert_xaxis()                      # finer grid to the right
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc="lower left", fontsize=9)

# annotate the 0.005 points
ax.annotate(f"{iasing['rms'][0]:.2f} K", (0.005, iasing["rms"][0]),
            textcoords="offset points", xytext=(8, 6), fontsize=9, color="#d62728")
ax.annotate(f"{iasi['rms'][0]:.2f} K", (0.005, iasi["rms"][0]),
            textcoords="offset points", xytext=(8, -14), fontsize=9, color="#1f77b4")

fig.tight_layout()
out = "data/lblrtm/grid_convergence_43um.png"
fig.savefig(out, dpi=130)
print("wrote", out)
