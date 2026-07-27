#!/usr/bin/env python3
"""T(p) retrieval on a real IASI FOV using the EUMETSAT NCM as Se
(retrieve_iasi_profile_ncm.jl). Left: spectral fit residual (F-y) against the
NCM per-channel BT noise envelope. Right: retrieved T(p) vs prior with posterior
sigma and averaging-kernel diagonal.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_profile_ncm_fit.csv", delimiter=",", names=True)
tp  = np.genfromtxt("data/iasi_profile_ncm_Tp.csv",  delimiter=",", names=True)
nu, yobs, ymod = fit["wavenumber_cm1"], fit["bt_obs_K"], fit["bt_model_K"]
resid, sncm = fit["residual_K"], fit["sigma_ncm_K"]
p, Tp, Tr = tp["p_hPa"], tp["T_prior_K"], tp["T_retr_K"]
spost, akd = tp["sigma_post_K"], tp["AK_diag"]
def rms(v): return np.sqrt(np.mean(v**2))

fig, (axR, axT) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios":[1.7,1]})

# residual vs NCM noise envelope
axR.fill_between(nu, -sncm, sncm, color="0.75", alpha=0.6, lw=0, label="NCM $\\pm\\sigma_e$ (BT)")
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, resid, color="C3", lw=0.7, label=f"F$-$y  (rms {rms(resid):.3f} K)")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("Spectral fit residual vs EUMETSAT NCM noise")
axR.legend(loc="lower right", fontsize=9); axR.grid(True, alpha=0.3)

# T(p): retrieved vs prior
axT.plot(Tp, p, "--", color="0.5", lw=1.3, label="prior")
axT.fill_betweenx(p, Tr-spost, Tr+spost, color="C0", alpha=0.20, lw=0, label="$\\pm\\sigma_{post}$")
axT.plot(Tr, p, "o-", color="C0", ms=3, lw=1.2, label="retrieved")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title(f"T(p)  ($\\Sigma$DOF {akd.sum():.2f})")
axT.grid(True, alpha=0.3, which="both"); axT.legend(loc="upper left", fontsize=9)
axd = axT.twiny()
axd.plot(akd, p, color="C2", lw=1.0, alpha=0.8)
axd.fill_betweenx(p, 0, akd, color="C2", alpha=0.12)
axd.set_xlim(0, max(0.05, akd.max()*1.15))
axd.set_xlabel("AK diagonal", color="C2", fontsize=8)
axd.tick_params(axis="x", colors="C2", labelsize=7)

fig.suptitle("Real IASI FOV T(p) retrieval with EUMETSAT-published noise covariance (NCM), "
             "BT space, 0.0005 cm$^{-1}$ grid", fontsize=12)
out = "data/iasi_profile_ncm.png"
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (resid rms {rms(resid):.4f} K; DOF {akd.sum():.2f})")
