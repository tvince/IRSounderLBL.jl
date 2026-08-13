#!/usr/bin/env python3
"""Full per-level joint retrieval (retrieve_iasi_joint_full.jl): T(50)+H2O(50)+O3(50),
Tropical prior. Shows the fit residual and the three retrieved profiles (retrieved vs
prior) each annotated with its per-level averaging-kernel DOF — documenting that the
full 50-level H2O/O3 retrieval extracts the same ~3.3 DOF the reduced basis already had
(O3 ~1.1, H2O ~2.3), and relaxes to prior where uninformed (the ill-conditioned block).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_joint_full_tropical_fit.csv", delimiter=",", names=True)
pr  = np.genfromtxt("data/iasi_joint_full_tropical_profiles.csv", delimiter=",", names=True)
nu, r_full, nedt = fit["wavenumber_cm1"], fit["res_full_K"], fit["nedt_K"]
p   = pr["pressure"]
def rms(v): return np.sqrt(np.mean(v**2))

fig = plt.figure(figsize=(15, 8))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.30, wspace=0.32)

# (top) full-band residual
axR = fig.add_subplot(gs[0, :])
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene $\\pm$NE$\\Delta$T")
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_full, color="C3", lw=0.7, label=f"full T+H$_2$O(50)+O$_3$(50)  (rms {rms(r_full):.3f} K)")
for x in (715.5, 725.5): axR.axvline(x, color="C1", lw=0.8, ls=":")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("Full per-level joint retrieval — IASI FOV #1, Tropical prior (645-800 cm$^{-1}$)")
axR.legend(loc="lower right", fontsize=9); axR.grid(True, alpha=0.3)

def profile_panel(ax, ret, prior, dof, xlabel, title, logx=False, color="C0"):
    ax.plot(prior, p, "--", color="0.5", lw=1.2, label="prior")
    ax.plot(ret,   p, "o-", color=color, ms=2.5, lw=1.1, label="retrieved")
    if logx: ax.set_xscale("log")
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_ylabel("pressure [hPa]")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, which="both"); ax.legend(loc="upper right", fontsize=8)
    # twin axis: per-level DOF
    axd = ax.twiny()
    axd.plot(dof, p, color="C2", lw=1.0, alpha=0.8)
    axd.fill_betweenx(p, 0, dof, color="C2", alpha=0.12)
    axd.set_xlim(0, max(0.02, dof.max()*1.15))
    axd.set_xlabel("per-level DOF", color="C2", fontsize=8)
    axd.tick_params(axis="x", colors="C2", labelsize=7)
    return axd

axT = fig.add_subplot(gs[1, 0])
profile_panel(axT, pr["T_K"], pr["T_prior"], pr["T_dof"],
              "temperature [K]", f"T(p)   ($\\Sigma$DOF {pr['T_dof'].sum():.2f})", color="C0")
axH = fig.add_subplot(gs[1, 1])
profile_panel(axH, pr["H2O_vmr"], pr["H2O_prior"], pr["H2O_dof"],
              "H$_2$O VMR [mol/mol]", f"H$_2$O   ($\\Sigma$DOF {pr['H2O_dof'].sum():.2f})",
              logx=True, color="C0")
axO = fig.add_subplot(gs[1, 2])
profile_panel(axO, pr["O3_vmr"], pr["O3_prior"], pr["O3_dof"],
              "O$_3$ VMR [mol/mol]", f"O$_3$   ($\\Sigma$DOF {pr['O3_dof'].sum():.2f})",
              logx=True, color="C0")

fig.suptitle("Full 50-level H$_2$O+O$_3$ retrieval extracts the same information as the reduced basis "
             "(O$_3$ ~1.1 DOF, H$_2$O ~2.3 DOF); uninformed levels relax to prior", fontsize=12)
out = "data/iasi_joint_full.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (rms {rms(r_full):.4f} K; DOF T {pr['T_dof'].sum():.2f} "
      f"H2O {pr['H2O_dof'].sum():.2f} O3 {pr['O3_dof'].sum():.2f})")
