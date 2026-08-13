#!/usr/bin/env python3
"""Plot the iso-4 experiment (scripts/retrieve_iasi_profile_lm_iso4.jl).

Does adding CO2 isotopologue 4 (627) collapse the 665 cm-1 residual spike?
Three residual traces on IASI FOV #1: Voigt(iso1-3) / VP_Y lm=5 (iso1-3) /
VP_Y lm=5 (iso1-4), with the observed BT on top and a zoom on the 662-670
nu2 Q-branch bandhead where the -8.7 K spike collapses to -0.14 K.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_profile_lm_iso4_fit.csv", delimiter=",", names=True)
nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
r_v  = fit["res_voigt_K"]        # Voigt, iso 1-3
r_3  = fit["res_vpy5_iso3_K"]    # VP_Y lm=5, iso 1-3
r_4  = fit["res_vpy5_iso4_K"]    # VP_Y lm=5, iso 1-4
nedt = fit["nedt_K"]

tp  = np.genfromtxt("data/iasi_profile_lm_iso4_Tp.csv", delimiter=",", names=True)
p    = tp["p_hPa"]
T_v  = tp["T_voigt_K"]; s_Tv = tp["sig_voigt_K"]
T_3  = tp["T_iso3_K"];  s_T3 = tp["sig_iso3_K"]
T_4  = tp["T_iso4_K"];  s_T4 = tp["sig_iso4_K"]
pm = p > 1e-3

def rms(v): return np.sqrt(np.mean(v**2))
rms_v, rms_3, rms_4 = rms(r_v), rms(r_3), rms(r_4)
QWIN = [(664.5, 669.0, "$\\nu_2$ Q bandhead"), (719.0, 723.0, "721 Q")]

fig = plt.figure(figsize=(17, 7))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 2.3, 1], height_ratios=[1, 1.6],
                       hspace=0.28, wspace=0.26)

# (P) retrieved T(p)
axP = fig.add_subplot(gs[:, 0])
axP.errorbar(T_v[pm], p[pm], xerr=s_Tv[pm], fmt="o-", color="0.5", ms=2.5, lw=1.0,
             ecolor="0.5", elinewidth=0.6, capsize=1.5, label="Voigt, iso 1–3 ±σ")
axP.errorbar(T_3[pm], p[pm], xerr=s_T3[pm], fmt="o-", color="C0", ms=2.5, lw=1.0,
             ecolor="C0", elinewidth=0.6, capsize=1.5, label="VP_Y lm=5, iso 1–3 ±σ")
axP.errorbar(T_4[pm], p[pm], xerr=s_T4[pm], fmt="o-", color="C3", ms=2.5, lw=1.3,
             ecolor="C3", elinewidth=0.6, capsize=1.5, label="VP_Y lm=5, iso 1–4 ±σ")
axP.set_yscale("log"); axP.invert_yaxis()
axP.set_xlabel("temperature [K]"); axP.set_ylabel("pressure [hPa]")
axP.set_title("retrieved T(p)")
axP.grid(True, which="both", alpha=0.3); axP.legend(loc="upper left", fontsize=7)

# (a) observed BT (full band)
axO = fig.add_subplot(gs[0, 1])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi, name in QWIN:
    axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.axvline(665.0, color="C3", lw=0.8, ls="--")
axO.set_ylabel("obs BT [K]")
axO.set_title("IASI FOV #1 — CO$_2$ $\\nu_2$ spectral fit (645–800 cm$^{-1}$)")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (b) residuals (full band)
axR = fig.add_subplot(gs[1, 1], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi, _ in QWIN:
    axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axvline(665.0, color="C3", lw=0.8, ls="--")
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_v, color="0.5", lw=0.6,           label=f"Voigt, iso 1–3 (rms {rms_v:.3f})")
axR.plot(nu, r_3, color="C0",  lw=0.6, alpha=0.8, label=f"VP_Y lm=5, iso 1–3 (rms {rms_3:.3f})")
axR.plot(nu, r_4, color="C3",  lw=0.9,           label=f"VP_Y lm=5, iso 1–4 (rms {rms_4:.3f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual — adding CO$_2$ iso-4 (627) collapses the 665 cm$^{-1}$ spike", fontsize=10)
axR.legend(loc="lower right", fontsize=8)
axR.grid(True, alpha=0.3)

# (c) zoom on the 662-670 bandhead
axZ = fig.add_subplot(gs[:, 2])
zm = (nu >= 661.5) & (nu <= 671.0)
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.axvline(665.0, color="C3", lw=0.7, ls="--")
axZ.plot(nu[zm], r_v[zm], "o-", color="0.5", ms=3, lw=0.9, label="Voigt, iso 1–3")
axZ.plot(nu[zm], r_3[zm], "o-", color="C0",  ms=3, lw=0.9, label="VP_Y lm=5, iso 1–3")
axZ.plot(nu[zm], r_4[zm], "o-", color="C3",  ms=3.5, lw=1.3, label="VP_Y lm=5, iso 1–4")
i665 = np.argmin(np.abs(nu - 665.0))
axZ.annotate(f"{r_3[i665]:+.2f} K", (665.0, r_3[i665]), color="C0",
             fontsize=8, xytext=(4, 0), textcoords="offset points", va="center")
axZ.annotate(f"{r_4[i665]:+.2f} K", (665.0, r_4[i665]), color="C3",
             fontsize=8, xytext=(4, 8), textcoords="offset points", va="center")
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("662–670 zoom: $\\nu_2$ Q-branch bandhead", fontsize=10)
axZ.legend(loc="lower right", fontsize=8)
axZ.grid(True, alpha=0.3)

out = "data/iasi_profile_lm_iso4.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  full-band RMS: Voigt {rms_v:.3f} | iso1–3 {rms_3:.3f} | iso1–4 {rms_4:.3f} K")
print(f"  665.0 residual: Voigt {r_v[i665]:+.2f} | iso1–3 {r_3[i665]:+.2f} | iso1–4 {r_4[i665]:+.2f} K")
