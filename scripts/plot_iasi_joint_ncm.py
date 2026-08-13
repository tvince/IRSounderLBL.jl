#!/usr/bin/env python3
"""Joint T+H2O(3 layers)+O3-column retrieval on a real IASI FOV using the EUMETSAT
NCM as Se (retrieve_iasi_joint.jl, IASI_NCM=... JOINT_DNU=0.0005 JOINT_BASE=tropical).

Shows how the *measured* (tight) noise covariance, WITHOUT masking the ~2 K
unfittable CO2 forward-model spikes, breaks the joint gas retrieval: the spikes are
~14-18 sigma events under the NCM, so the inversion pulls O3/H2O to unphysical
values and the fit gets WORSE than the T-only prior. Contrast the residuals against
the grey NCM +-sigma envelope.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt("data/iasi_joint_tropical_ncm_fit.csv", delimiter=",", names=True)
nu   = d["wavenumber_cm1"]
rpri = d["res_prior_K"]; rjoint = d["res_joint_K"]; nedt = d["nedt_K"]
def rms(v): return np.sqrt(np.mean(v**2))

# retrieved state (from the run log)
o3_scale = 0.033
h2o = [("1013-715 hPa", 0.647), ("633-378 hPa", 0.006), ("329-0 hPa", 5.936)]
spikes = (715.75, 722.75, 723.25)

fig, (ax, axz) = plt.subplots(2, 1, figsize=(13, 8),
                              gridspec_kw={"height_ratios":[1.3,1], "hspace":0.28})

for a in (ax, axz):
    a.fill_between(nu, -nedt, nedt, color="0.72", alpha=0.6, lw=0,
                   label="NCM $\\pm\\sigma_e$ (measured noise)")
    a.axhline(0, color="0.5", lw=0.8)
    a.plot(nu, rpri, color="C0", lw=0.8, label=f"prior: fixed-O$_3$, T-only  (rms {rms(rpri):.3f} K)")
    a.plot(nu, rjoint, color="C3", lw=0.8,
           label=f"JOINT T+H$_2$O+O$_3$ w/ NCM  (rms {rms(rjoint):.3f} K)")
    for x in spikes: a.axvline(x, color="C1", lw=0.7, ls=":")
    a.grid(True, alpha=0.3); a.set_ylabel("F$-$y [K]")

ax.set_xlim(645, 800); ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Joint retrieval with the measured EUMETSAT NCM as S$_e$ — the tight noise "
             "breaks the gas fit (0.0005 cm$^{-1}$, Tropical)")
axz.set_xlim(700, 730); axz.set_xlabel("wavenumber [cm$^{-1}$]")
axz.set_title("zoom 700–730 cm$^{-1}$ (the CO$_2$ $\\nu_2$ Q / spike region)", fontsize=10)

txt = ("Retrieved (unphysical — driven by 14–18$\\sigma$ spikes):\n"
       f"  O$_3$ column  = {o3_scale:.3f}× climatology\n"
       + "\n".join(f"  H$_2$O {lbl} = {s:.3f}×" for lbl, s in h2o)
       + "\n\nrms  prior 0.565 K  →  joint 1.453 K  (WORSE)\n"
         "$\\chi^2$/n = 200.9,  did not converge (max_iter 15)")
axz.text(0.015, 0.03, txt, transform=axz.transAxes, fontsize=8.5, va="bottom", ha="left",
         family="monospace", bbox=dict(boxstyle="round", fc="#fff4f4", ec="C3", alpha=0.9))

out = "data/iasi_joint_ncm.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (prior rms {rms(rpri):.3f}, joint rms {rms(rjoint):.3f} K)")
