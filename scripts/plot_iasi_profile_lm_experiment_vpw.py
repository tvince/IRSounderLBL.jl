#!/usr/bin/env python3
"""Plot the VP_W line-mixing experiment written by
scripts/retrieve_iasi_profile_lm_experiment_vpw.jl.

Three-way comparison (Voigt / VP_Y / VP_W) on IASI FOV #1:
  (a) T(p): the three retrieved profiles with ±σ_post error bars (log-p);
  (b) retrieval shift dT = T_vpw - T_vpy vs pressure (does full-matrix move it?);
  (c) observed BT over CO2 nu2, with the 667 & 721 Q-branches shaded;
  (d) spectral residual (F-y) for all three, Q-branches shaded, RMS annotated,
      with the scene ±NEDT (1sigma sqrt(diag Se)) noise band.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tp  = np.genfromtxt("data/iasi_profile_lm_experiment_vpw_Tp.csv",  delimiter=",", names=True)
fit = np.genfromtxt("data/iasi_profile_lm_experiment_vpw_fit.csv", delimiter=",", names=True)

p       = tp["p_hPa"]
T_v     = tp["T_voigt_K"]; sig_v = tp["sig_voigt_K"]
T_y     = tp["T_vpy_K"];   sig_y = tp["sig_vpy_K"]
T_w     = tp["T_vpw_K"];   sig_w = tp["sig_vpw_K"]
dT      = T_w - T_y                     # VP_W - VP_Y
m = p > 1e-3

nu    = fit["wavenumber_cm1"]
bto   = fit["bt_obs_K"]
r_v   = fit["res_voigt_K"]
r_y   = fit["res_vpy_K"]
r_w   = fit["res_vpw_K"]
nedt  = fit["nedt_K"]

QWIN = [(665.0, 669.0, "667 Q"), (719.0, 723.0, "721 Q")]
def inwin(nu, lo, hi): return (nu >= lo) & (nu <= hi)
qmask = np.zeros_like(nu, dtype=bool)
for lo, hi, _ in QWIN: qmask |= inwin(nu, lo, hi)

def rms(v): return np.sqrt(np.mean(v**2))
rms_v, rms_y, rms_w = rms(r_v), rms(r_y), rms(r_w)
qr_v, qr_y, qr_w = rms(r_v[qmask]), rms(r_y[qmask]), rms(r_w[qmask])

fig = plt.figure(figsize=(15, 6))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.9], height_ratios=[3, 1.2],
                       hspace=0.30, wspace=0.32)

# (a) T(p): three profiles with error bars ---------------------------------------
axT = fig.add_subplot(gs[:, 0])
axT.errorbar(T_v[m], p[m], xerr=sig_v[m], fmt="o-", color="0.5", ms=2.5, lw=1.0,
             ecolor="0.5", elinewidth=0.6, capsize=1.5, label="Voigt (LM off) ±σ")
axT.errorbar(T_y[m], p[m], xerr=sig_y[m], fmt="o-", color="C0", ms=2.5, lw=1.0,
             ecolor="C0", elinewidth=0.6, capsize=1.5, label="VP_Y ±σ")
axT.errorbar(T_w[m], p[m], xerr=sig_w[m], fmt="o-", color="C3", ms=2.5, lw=1.3,
             ecolor="C3", elinewidth=0.6, capsize=1.5, label="VP_W ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("T(p): Voigt vs VP_Y vs VP_W")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

# (b) retrieval shift VP_W - VP_Y -------------------------------------------------
axD = fig.add_subplot(gs[:, 1], sharey=axT)
axD.axvline(0, color="0.6", lw=0.8)
axD.plot(dT[m], p[m], "s-", color="C3", ms=3, lw=1.2)
axD.set_xlabel("$\\Delta T$ = VP_W $-$ VP_Y [K]")
axD.set_title("full-matrix shift from first-order")
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

# (d) residual: Voigt vs VP_Y vs VP_W ---------------------------------------------
axR = fig.add_subplot(gs[1, 2], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi, _ in QWIN:
    axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_v, color="0.5", lw=0.6, label=f"Voigt (rms {rms_v:.2f}; Q {qr_v:.2f})")
axR.plot(nu, r_y, color="C0",  lw=0.6, alpha=0.8, label=f"VP_Y (rms {rms_y:.2f}; Q {qr_y:.2f})")
axR.plot(nu, r_w, color="C3",  lw=0.8, label=f"VP_W (rms {rms_w:.2f}; Q {qr_w:.2f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual  (grey band = scene ±NEΔT; orange = Q-branch windows)", fontsize=9)
axR.legend(loc="lower right", fontsize=7, ncol=2)
axR.grid(True, alpha=0.3)

fig.suptitle("IASI FOV #1 T(p) retrieval — VP_W full-matrix vs VP_Y vs Voigt  "
             "(CO$_2$ $\\nu_2$ 645–800 cm$^{-1}$)", fontsize=12)
out = "data/iasi_profile_lm_experiment_vpw.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  full-band residual RMS:  Voigt {rms_v:.3f} -> VP_Y {rms_y:.3f} -> VP_W {rms_w:.3f} K")
print(f"  Q-branch residual RMS :  Voigt {qr_v:.3f} -> VP_Y {qr_y:.3f} -> VP_W {qr_w:.3f} K")
