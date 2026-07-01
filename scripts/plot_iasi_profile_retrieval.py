#!/usr/bin/env python3
"""Plot the T(p) profile-retrieval result written by retrieve_iasi_profile.jl.

Reads data/iasi_profile_retrieval_{Tp,fit}.csv and makes a 3-panel figure:
  (a) T(p): AFGL prior vs retrieved, with ±σ_post error bars (log-p, surface down);
  (b) averaging-kernel diagonal vs pressure (where the measurement adds information);
  (c) spectral fit residual (F−y) across the CO2 nu2 band, with obs BT for context.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tp  = np.genfromtxt("data/iasi_profile_retrieval_Tp.csv",  delimiter=",", names=True)
fit = np.genfromtxt("data/iasi_profile_retrieval_fit.csv", delimiter=",", names=True)

p      = tp["p_hPa"]
Tprior = tp["T_prior_K"]
Tretr  = tp["T_retr_K"]
sig    = tp["sigma_post_K"]
ak     = tp["AK_diag"]
dof    = np.nansum(ak)

# Keep the panels readable: plot only the levels with meaningful pressure (drop the
# ~0 hPa top rows that crowd a log axis); they carry no information anyway (AK≈0).
m = p > 1e-3

nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
res  = fit["residual_K"]
rms  = np.sqrt(np.mean(res**2))

# Scene NEDeltaT envelope (thesis Eq. 2.3), recomputed from the observed BT — the
# same closed form as the Julia scene_nedt: sigma(nu) = NEDT_280 * dB/dT(nu,T_ref)
# / dB/dT(nu, Tb_obs). The C1 prefactor cancels in the ratio, so this reproduces
# the retrieval's Se diagonal exactly (median ~0.32 K over this band).
C2 = 1.4387769  # cm.K  (second radiation constant)
NEDT_280, T_REF = 0.25, 280.0
def dB_dT(nu, T):
    x = C2 * nu / T
    ex = np.exp(x)
    return nu**4 * ex / (T**2 * (ex - 1.0)**2)   # up to the common C1*C2 factor
sig_scene = NEDT_280 * dB_dT(nu, T_REF) / dB_dT(nu, bto)
frac_out  = np.mean(np.abs(res) > sig_scene)     # fraction beyond the 1-sigma band

fig = plt.figure(figsize=(14, 6))
gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.6], height_ratios=[3, 1],
                       hspace=0.28, wspace=0.30)

# (a) T(p) prior vs retrieved -----------------------------------------------------
axT = fig.add_subplot(gs[:, 0])
axT.plot(Tprior[m], p[m], "o-", color="0.6", ms=3, lw=1.2, label="AFGL prior")
axT.errorbar(Tretr[m], p[m], xerr=sig[m], fmt="o-", color="C3", ms=3, lw=1.4,
             ecolor="C3", elinewidth=0.8, capsize=2, label="retrieved ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("T(p): prior vs retrieved")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

# (b) averaging-kernel diagonal ---------------------------------------------------
axA = fig.add_subplot(gs[:, 1], sharey=axT)
axA.plot(ak[m], p[m], "s-", color="C0", ms=3, lw=1.2)
axA.set_xlabel("averaging-kernel diagonal"); axA.set_title(f"vertical sensitivity\nDOF={dof:.2f}")
axA.grid(True, which="both", alpha=0.3)
plt.setp(axA.get_yticklabels(), visible=False)

# (c) observed BT + residual ------------------------------------------------------
axO = fig.add_subplot(gs[0, 2])
axO.plot(nu, bto, color="C0", lw=0.8)
axO.set_ylabel("obs BT [K]"); axO.set_title("spectral fit over CO$_2$ $\\nu_2$")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

axR = fig.add_subplot(gs[1, 2], sharex=axO)
axR.fill_between(nu, -sig_scene, sig_scene, color="0.75", alpha=0.5,
                 label="$\\pm$ scene NE$\\Delta$T")
axR.plot(nu,  sig_scene, color="0.5", lw=0.6)
axR.plot(nu, -sig_scene, color="0.5", lw=0.6)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, res, color="C3", lw=0.8, label="F−y")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F−y [K]")
axR.set_title(f"residual  (RMS {rms:.2f} K; {100*frac_out:.0f}% beyond $\\pm$NE$\\Delta$T)",
              fontsize=9)
axR.legend(loc="lower right", fontsize=7, ncol=2)
axR.grid(True, alpha=0.3)

fig.suptitle("IASI FOV #1 T(p) retrieval — CO$_2$ $\\nu_2$ 645–800 cm$^{-1}$, scene NE$\\Delta$T S$_e$",
             fontsize=12)
out = "data/iasi_profile_retrieval.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (DOF={dof:.2f}, residual RMS={rms:.3f} K)")
