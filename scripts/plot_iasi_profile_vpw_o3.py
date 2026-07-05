#!/usr/bin/env python3
"""Plot the VP_W + O3 experiment (scripts/retrieve_iasi_profile_vpw_o3.jl).

Three residual traces on IASI FOV #1: VP_Y lm=5 no-O3 / VP_Y lm=5 +O3 /
VP_W lm=5 +O3. Shows that full-matrix line mixing removes the wing over-thinning
half of the 715-724 residual, so O3 no longer over-corrects (overshoot at 722-723
halved). Zoom on 710-740 where O3 nu2 R-branch overlaps the CO2 721 Q-branch.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_profile_vpw_o3_fit.csv", delimiter=",", names=True)
nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
r_yn = fit["res_vpy_noO3_K"]   # VP_Y, no O3
r_yo = fit["res_vpy_O3_K"]     # VP_Y, +O3
r_wo = fit["res_vpw_O3_K"]     # VP_W, +O3
nedt = fit["nedt_K"]

def rms(v): return np.sqrt(np.mean(v**2))
R = {"VP_Y no-O3": rms(r_yn), "VP_Y +O3": rms(r_yo), "VP_W +O3": rms(r_wo)}
QWIN = [(700.0, 710.0, "O$_3$ $\\nu_2$ Q"), (719.0, 723.0, "721 Q")]

fig = plt.figure(figsize=(15, 6.5))
gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1.6],
                       hspace=0.3, wspace=0.2)

# (a) obs BT
axO = fig.add_subplot(gs[0, 0])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi, _ in QWIN: axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.set_ylabel("obs BT [K]"); axO.set_title("IASI FOV #1 — CO$_2$ $\\nu_2$ fit (645–800)")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (b) residuals
axR = fig.add_subplot(gs[1, 0], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi, _ in QWIN: axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_yn, color="0.5", lw=0.6,            label=f"VP_Y no-O$_3$ (rms {R['VP_Y no-O3']:.3f})")
axR.plot(nu, r_yo, color="C0",  lw=0.6, alpha=0.8, label=f"VP_Y +O$_3$ (rms {R['VP_Y +O3']:.3f})")
axR.plot(nu, r_wo, color="C3",  lw=0.9,            label=f"VP_W +O$_3$ (rms {R['VP_W +O3']:.3f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual: VP_Y no-O$_3$ → VP_Y +O$_3$ → VP_W +O$_3$", fontsize=10)
axR.legend(loc="lower right", fontsize=8, ncol=2); axR.grid(True, alpha=0.3)

# (z) zoom 710-740
axZ = fig.add_subplot(gs[:, 1])
zm = (nu >= 710) & (nu <= 740)
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.plot(nu[zm], r_yn[zm], "o-", color="0.5", ms=2.5, lw=0.8, label="VP_Y no-O$_3$")
axZ.plot(nu[zm], r_yo[zm], "o-", color="C0",  ms=2.5, lw=0.8, label="VP_Y +O$_3$")
axZ.plot(nu[zm], r_wo[zm], "o-", color="C3",  ms=3.5, lw=1.2, label="VP_W +O$_3$")
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("710–740 zoom: VP_W trims the O$_3$\novershoot at 722–724", fontsize=10)
axZ.legend(loc="lower right", fontsize=8); axZ.grid(True, alpha=0.3)

fig.suptitle("VP_W + O$_3$: full-matrix line mixing removes the wing over-thinning, "
             "so O$_3$ stops over-correcting (722–723 overshoot halved)", fontsize=12)
out = "data/iasi_profile_vpw_o3.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
for k, v in R.items(): print(f"  {k}: rms {v:.3f} K")
