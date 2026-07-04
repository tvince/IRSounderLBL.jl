#!/usr/bin/env python3
"""Plot observed vs modelled IASI brightness temperature + residual.

Reads data/iasi_retrieval_fov.csv (written by scripts/retrieve_iasi_fov.jl) and
makes a two-panel figure: top = observed BBT and the final modelled BBT, bottom =
residual (model - observed). Saves data/iasi_retrieval_fov.png.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CSV = "data/iasi_retrieval_fov.csv"

nu, obs, model, resid = [], [], [], []
with open(CSV) as f:
    for row in csv.DictReader(f):
        nu.append(float(row["wavenumber_cm-1"]))
        obs.append(float(row["bt_obs_K"]))
        model.append(float(row["bt_model_K"]))
        resid.append(float(row["residual_K"]))
nu = np.array(nu); obs = np.array(obs); model = np.array(model); resid = np.array(resid)
rms = float(np.sqrt(np.mean(resid**2)))

fig = plt.figure(figsize=(9, 6.5))
gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)

ax0 = fig.add_subplot(gs[0])
ax0.plot(nu, obs, color="k", lw=0.9, label="IASI observed")
ax0.plot(nu, model, color="tab:red", lw=0.9, alpha=0.8, label="modelled (retrieved T)")
ax0.set_ylabel("Brightness temperature [K]")
ax0.set_title(f"IASI L1C T$_{{sfc}}$ retrieval — 12 µm window  (residual RMS = {rms:.3f} K)")
ax0.legend(loc="upper right", frameon=False)
ax0.grid(alpha=0.25)
ax0.tick_params(labelbottom=False)

ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax1.axhline(0.0, color="0.6", lw=0.8)
ax1.plot(nu, resid, color="tab:blue", lw=0.8)
ax1.fill_between(nu, resid, 0.0, color="tab:blue", alpha=0.25)
ax1.set_xlabel("Wavenumber [cm$^{-1}$]")
ax1.set_ylabel("model − obs [K]")
ax1.grid(alpha=0.25)
rmax = 1.05 * np.max(np.abs(resid))
ax1.set_ylim(-rmax, rmax)

fig.savefig("data/iasi_retrieval_fov.png", dpi=140, bbox_inches="tight")
print(f"wrote data/iasi_retrieval_fov.png   (n={len(nu)}, residual RMS={rms:.3f} K, "
      f"max|resid|={np.max(np.abs(resid)):.2f} K)")
