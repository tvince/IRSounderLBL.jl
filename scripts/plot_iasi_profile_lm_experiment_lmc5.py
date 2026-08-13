#!/usr/bin/env python3
"""Plot the lm_cutoff=5 VP_Y experiment (scripts/retrieve_iasi_profile_lm_experiment_lmc5.jl).

Three-way: Voigt / VP_Y(lm_cutoff=25) / VP_Y(lm_cutoff=5) on IASI FOV #1.
  (a) T(p) with ±sigma_post error bars (log-p);
  (b) retrieval shift dT = VP_Y(lm=5) - VP_Y(lm=25) vs pressure;
  (c) observed BT over CO2 nu2, 667 & 721 Q-branches shaded, 692.75 R-trough marked;
  (d) spectral residual for all three + scene +/-NEDT band; zoom inset on 688-696.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tp  = np.genfromtxt("data/iasi_profile_lm_experiment_lmc5_Tp.csv",  delimiter=",", names=True)
fit = np.genfromtxt("data/iasi_profile_lm_experiment_lmc5_fit.csv", delimiter=",", names=True)

p     = tp["p_hPa"]
T_v   = tp["T_voigt_K"];  s_v  = tp["sig_voigt_K"]
T_y25 = tp["T_vpy25_K"];  s_y25 = tp["sig_vpy25_K"]
T_y5  = tp["T_vpy5_K"];   s_y5  = tp["sig_vpy5_K"]
dT    = T_y5 - T_y25
m = p > 1e-3

nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
r_v  = fit["res_voigt_K"]
r25  = fit["res_vpy25_K"]
r5   = fit["res_vpy5_K"]
nedt = fit["nedt_K"]

QWIN = [(665.0, 669.0, "667 Q"), (719.0, 723.0, "721 Q")]
def inwin(nu, lo, hi): return (nu >= lo) & (nu <= hi)
def rms(v): return np.sqrt(np.mean(v**2))
rms_v, rms_25, rms_5 = rms(r_v), rms(r25), rms(r5)

fig = plt.figure(figsize=(15, 6))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.9], height_ratios=[3, 1.3],
                       hspace=0.32, wspace=0.34)

# (a) T(p)
axT = fig.add_subplot(gs[:, 0])
axT.errorbar(T_v[m], p[m], xerr=s_v[m], fmt="o-", color="0.5", ms=2.5, lw=1.0,
             ecolor="0.5", elinewidth=0.6, capsize=1.5, label="Voigt (LM off) ±σ")
axT.errorbar(T_y25[m], p[m], xerr=s_y25[m], fmt="o-", color="C0", ms=2.5, lw=1.0,
             ecolor="C0", elinewidth=0.6, capsize=1.5, label="VP_Y lm=25 ±σ")
axT.errorbar(T_y5[m], p[m], xerr=s_y5[m], fmt="o-", color="C3", ms=2.5, lw=1.3,
             ecolor="C3", elinewidth=0.6, capsize=1.5, label="VP_Y lm=5 (fix) ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("T(p): Voigt vs VP_Y lm=25 vs lm=5")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

# (b) shift lm=5 - lm=25
axD = fig.add_subplot(gs[:, 1], sharey=axT)
axD.axvline(0, color="0.6", lw=0.8)
axD.plot(dT[m], p[m], "s-", color="C3", ms=3, lw=1.2)
axD.set_xlabel("$\\Delta T$ = lm5 $-$ lm25 [K]")
axD.set_title("retrieval shift from the fix")
axD.grid(True, which="both", alpha=0.3)
plt.setp(axD.get_yticklabels(), visible=False)

# (c) observed BT
axO = fig.add_subplot(gs[0, 2])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi, name in QWIN:
    axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.axvline(692.75, color="C3", lw=0.8, ls="--")
axO.text(692.75, axO.get_ylim()[1], " 692.75 R-trough", color="C3", fontsize=7,
         va="top", ha="left")
axO.set_ylabel("obs BT [K]"); axO.set_title("spectral fit over CO$_2$ $\\nu_2$")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (d) residuals
axR = fig.add_subplot(gs[1, 2], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi, _ in QWIN:
    axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axvline(692.75, color="C3", lw=0.8, ls="--")
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_v,  color="0.5", lw=0.6, label=f"Voigt (rms {rms_v:.3f})")
axR.plot(nu, r25,  color="C0",  lw=0.6, alpha=0.8, label=f"VP_Y lm=25 (rms {rms_25:.3f})")
axR.plot(nu, r5,   color="C3",  lw=0.8, label=f"VP_Y lm=5 (rms {rms_5:.3f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual (dashed = 692.75 R-trough; grey = scene ±NEΔT)", fontsize=9)
axR.legend(loc="lower right", fontsize=7, ncol=2)
axR.grid(True, alpha=0.3)

# zoom inset on the spike — placed over the flat 740–800 residual region
# (upper-right), clear of the panel title and the lower-right legend.
axin = axR.inset_axes([0.66, 0.40, 0.32, 0.50])
zm = inwin(nu, 688, 697)
axin.axhline(0, color="0.5", lw=0.6)
axin.plot(nu[zm], r_v[zm],  color="0.5", lw=0.8)
axin.plot(nu[zm], r25[zm],  color="C0",  lw=0.8)
axin.plot(nu[zm], r5[zm],   color="C3",  lw=1.0)
axin.axvline(692.75, color="C3", lw=0.6, ls="--")
axin.set_title("688–697 zoom (692.75 spike)", fontsize=7)
axin.tick_params(labelsize=6)
axin.patch.set_alpha(0.92)

fig.suptitle("IASI FOV #1 T(p) — VP_Y line-mixing wing cutoff fix "
             "(lm_cutoff 25→5 cm$^{-1}$ clears the 692.75 R-trough spike)", fontsize=12)
out = "data/iasi_profile_lm_experiment_lmc5.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  full-band RMS: Voigt {rms_v:.3f} | lm=25 {rms_25:.3f} | lm=5 {rms_5:.3f} K")
js = np.argmin(np.abs(nu - 692.75))
print(f"  692.75 residual: Voigt {r_v[js]:+.2f} | lm=25 {r25[js]:+.2f} | lm=5 {r5[js]:+.2f} K")
