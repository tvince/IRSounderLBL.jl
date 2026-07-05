#!/usr/bin/env python3
"""Plot the O3 experiment (scripts/retrieve_iasi_profile_o3.jl).

Does modelling the O3 nu2 band change the 645-800 fit? Residuals for VP_Y lm=5
iso1-4 WITHOUT vs WITH O3, plus the O3-effect trace (res_O3 - res_noO3), with a
zoom on the 710-740 region where O3's nu2 R-branch overlaps the CO2 721 Q-branch.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_profile_o3_fit.csv", delimiter=",", names=True)
nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
r_n  = fit["res_noO3_K"]
r_o  = fit["res_O3_K"]
nedt = fit["nedt_K"]
dO3  = r_o - r_n

def rms(v): return np.sqrt(np.mean(v**2))
rms_n, rms_o = rms(r_n), rms(r_o)
QWIN = [(700.0, 710.0, "O$_3$ $\\nu_2$ Q"), (719.0, 723.0, "721 Q")]

fig = plt.figure(figsize=(15, 7))
gs  = fig.add_gridspec(3, 2, width_ratios=[2.2, 1], height_ratios=[1, 1, 1],
                       hspace=0.35, wspace=0.20)

# (a) obs BT
axO = fig.add_subplot(gs[0, 0])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi, _ in QWIN: axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.set_ylabel("obs BT [K]"); axO.set_title("IASI FOV #1 — CO$_2$ $\\nu_2$ fit (645–800)")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (b) residuals no-O3 vs O3
axR = fig.add_subplot(gs[1, 0], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi, _ in QWIN: axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_n, color="C0", lw=0.6, alpha=0.8, label=f"no O$_3$ (rms {rms_n:.3f})")
axR.plot(nu, r_o, color="C3", lw=0.8,            label=f"+ O$_3$ (rms {rms_o:.3f})")
axR.set_ylabel("F$-$y [K]"); axR.set_title("residual: VP_Y lm=5, iso 1–4 — without vs with O$_3$", fontsize=10)
axR.legend(loc="lower right", fontsize=8, ncol=3); axR.grid(True, alpha=0.3)
plt.setp(axR.get_xticklabels(), visible=False)

# (c) O3 effect
axD = fig.add_subplot(gs[2, 0], sharex=axO)
for lo, hi, _ in QWIN: axD.axvspan(lo, hi, color="C1", alpha=0.15)
axD.axhline(0, color="0.5", lw=0.8)
axD.plot(nu, dO3, color="C2", lw=0.7)
axD.set_xlabel("wavenumber [cm$^{-1}$]"); axD.set_ylabel("$\\Delta$BT O$_3$ [K]")
axD.set_title("O$_3$ effect (res$_{+O3}$ $-$ res$_{noO3}$) — negative = O$_3$ adds absorption", fontsize=10)
axD.grid(True, alpha=0.3)

# (z) zoom on 710-740
axZ = fig.add_subplot(gs[:, 1])
zm = (nu >= 710) & (nu <= 740)
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.plot(nu[zm], r_n[zm], "o-", color="C0", ms=3, lw=0.9, label="no O$_3$")
axZ.plot(nu[zm], r_o[zm], "o-", color="C3", ms=3.5, lw=1.2, label="+ O$_3$")
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("710–740 zoom: O$_3$ $\\nu_2$ R-branch\nover the CO$_2$ 721 Q-branch", fontsize=10)
axZ.legend(loc="lower right", fontsize=8); axZ.grid(True, alpha=0.3)

fig.suptitle("Modelling O$_3$ $\\nu_2$ — significant 2–4 K effect at 715–724 & 736–737 "
             "(overlaps the CO$_2$ Q-branch inter-line minima)", fontsize=12)
out = "data/iasi_profile_o3.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  full-band RMS: no O3 {rms_n:.3f} | + O3 {rms_o:.3f} K")
