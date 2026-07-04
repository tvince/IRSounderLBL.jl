#!/usr/bin/env python3
"""Plot the VP_Y line-mixing experiment written by
scripts/retrieve_iasi_profile_lm_experiment.jl.

Reads data/iasi_profile_lm_experiment_{Tp,fit}.csv and makes a 4-panel figure
comparing the LM-off (Voigt) and VP_Y retrievals on the same IASI FOV #1:
  (a) T(p): Voigt vs VP_Y retrieved profiles with ±σ_post error bars (log-p);
  (b) retrieval shift dT = T_vpy - T_voigt vs pressure;
  (c) observed BT over the CO2 nu2 band, with the 667 & 721 Q-branches shaded;
  (d) spectral residual (F-y) for Voigt vs VP_Y, Q-branches shaded, RMS annotated,
      with the scene ±NEDT (1sigma sqrt(diag Se), thesis Eq 2.3) noise band.

Matches the style of scripts/plot_iasi_profile_retrieval.py.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tp  = np.genfromtxt("data/iasi_profile_lm_experiment_Tp.csv",  delimiter=",", names=True)
fit = np.genfromtxt("data/iasi_profile_lm_experiment_fit.csv", delimiter=",", names=True)

p       = tp["p_hPa"]
T_voigt = tp["T_voigt_K"]
T_vpy   = tp["T_vpy_K"]
dT      = tp["dT_K"]
sig_v   = tp["sig_voigt_K"]     # posterior 1sigma per level (sqrt diag S_hat)
sig_y   = tp["sig_vpy_K"]
m = p > 1e-3          # drop the ~0 hPa top rows that crowd a log axis (AK~0 there)

nu    = fit["wavenumber_cm1"]
bto   = fit["bt_obs_K"]
r_v   = fit["res_voigt_K"]
r_y   = fit["res_vpy_K"]
nedt  = fit["nedt_K"]           # scene NEDT (sqrt diag Se), thesis Eq 2.3

# CO2 Q-branch windows (must match QBRANCHES in the Julia experiment script).
QWIN = [(665.0, 669.0, "667 Q"), (719.0, 723.0, "721 Q")]
def inwin(nu, lo, hi): return (nu >= lo) & (nu <= hi)
qmask = np.zeros_like(nu, dtype=bool)
for lo, hi, _ in QWIN: qmask |= inwin(nu, lo, hi)

def rms(v): return np.sqrt(np.mean(v**2))
rms_v, rms_y = rms(r_v), rms(r_y)
qrms_v, qrms_y = rms(r_v[qmask]), rms(r_y[qmask])

fig = plt.figure(figsize=(15, 6))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.9], height_ratios=[3, 1.2],
                       hspace=0.30, wspace=0.32)

# (a) T(p): Voigt vs VP_Y ---------------------------------------------------------
axT = fig.add_subplot(gs[:, 0])
axT.errorbar(T_voigt[m], p[m], xerr=sig_v[m], fmt="o-", color="0.5", ms=3, lw=1.2,
             ecolor="0.5", elinewidth=0.7, capsize=2, label="Voigt (LM off) ±σ")
axT.errorbar(T_vpy[m],   p[m], xerr=sig_y[m], fmt="o-", color="C3",  ms=3, lw=1.4,
             ecolor="C3",  elinewidth=0.7, capsize=2, label="VP_Y ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("T(p): Voigt vs VP_Y")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

# (b) retrieval shift dT ----------------------------------------------------------
axD = fig.add_subplot(gs[:, 1], sharey=axT)
axD.axvline(0, color="0.6", lw=0.8)
axD.plot(dT[m], p[m], "s-", color="C0", ms=3, lw=1.2)
axD.set_xlabel("$\\Delta T$ = VP_Y $-$ Voigt [K]")
axD.set_title("retrieval shift from LM")
axD.grid(True, which="both", alpha=0.3)
plt.setp(axD.get_yticklabels(), visible=False)

# (c) observed BT with Q-branches shaded ------------------------------------------
axO = fig.add_subplot(gs[0, 2])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi, name in QWIN:
    axO.axvspan(lo, hi, color="C1", alpha=0.15)
    axO.text((lo+hi)/2, axO.get_ylim()[0], name, ha="center", va="bottom",
             fontsize=7, color="C1")
axO.set_ylabel("obs BT [K]"); axO.set_title("spectral fit over CO$_2$ $\\nu_2$")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (d) residual: Voigt vs VP_Y -----------------------------------------------------
axR = fig.add_subplot(gs[1, 2], sharex=axO)
# scene ±NEDT (1σ) noise band — residuals inside it are at the noise floor.
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
frac_v = np.mean(np.abs(r_v) > nedt); frac_y = np.mean(np.abs(r_y) > nedt)
for lo, hi, _ in QWIN:
    axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_v, color="0.5", lw=0.7,
         label=f"Voigt (rms {rms_v:.2f} K; Q {qrms_v:.2f}; {100*frac_v:.0f}% >NEΔT)")
axR.plot(nu, r_y, color="C3",  lw=0.8,
         label=f"VP_Y (rms {rms_y:.2f} K; Q {qrms_y:.2f}; {100*frac_y:.0f}% >NEΔT)")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual  (grey band = scene ±NEΔT; orange = Q-branch windows)", fontsize=9)
axR.legend(loc="lower right", fontsize=7)
axR.grid(True, alpha=0.3)

fig.suptitle("IASI FOV #1 T(p) retrieval — VP_Y line-mixing vs plain Voigt  "
             "(CO$_2$ $\\nu_2$ 645–800 cm$^{-1}$)", fontsize=12)
out = "data/iasi_profile_lm_experiment.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  full-band residual RMS:  Voigt {rms_v:.3f} K -> VP_Y {rms_y:.3f} K")
print(f"  Q-branch residual RMS :  Voigt {qrms_v:.3f} K -> VP_Y {qrms_y:.3f} K")
