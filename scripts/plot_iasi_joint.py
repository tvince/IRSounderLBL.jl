#!/usr/bin/env python3
"""Plot the joint T + H2O + total-column O3 retrieval (retrieve_iasi_joint.jl).

Prior = VP_Y lm=5 with FIXED climatological O3, T-only. Joint = same forward model
but retrieving T(profile) + H2O as DFS-matched bulk layers + O3 as a single column
scale. Shows the full-band residual improvement (rms 0.565 -> 0.462 K) and, in the
710-740 zoom, that retrieving the O3 amount (0.584x climatology) clears the 722-723
overshoot. Retrieved reduced params annotated.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fit = np.genfromtxt("data/iasi_joint_fit.csv", delimiter=",", names=True)
nu   = fit["wavenumber_cm1"]
bto  = fit["bt_obs_K"]
r_p  = fit["res_prior_K"]
r_j  = fit["res_joint_K"]
nedt = fit["nedt_K"]

def rms(v): return np.sqrt(np.mean(v**2))
Rp, Rj = rms(r_p), rms(r_j)

# retrieved reduced params (from the driver output)
o3_scale = 0.584
h2o = [("1013-617 hPa", 1.670), ("540-194 hPa", 1.040), ("166-0 hPa", 1.138)]
QWIN = [(700.0, 710.0), (719.0, 724.0)]

fig = plt.figure(figsize=(15, 6.5))
gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1.6],
                       hspace=0.30, wspace=0.20)

# (a) observed BT
axO = fig.add_subplot(gs[0, 0])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi in QWIN: axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.set_ylabel("obs BT [K]")
axO.set_title("IASI FOV #1 — joint T + H$_2$O + total-column O$_3$ (645–800 cm$^{-1}$)")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (b) residuals prior vs joint
axR = fig.add_subplot(gs[1, 0], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi in QWIN: axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_p, color="0.5", lw=0.6,            label=f"prior: fixed O$_3$, T-only (rms {Rp:.3f})")
axR.plot(nu, r_j, color="C3",  lw=0.7,            label=f"joint: T+H$_2$O+O$_3$col (rms {Rj:.3f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("residual: retrieving the H$_2$O and O$_3$ amounts", fontsize=10)
axR.legend(loc="lower right", fontsize=8); axR.grid(True, alpha=0.3)

# (z) zoom 710-740
axZ = fig.add_subplot(gs[:, 1])
zm = (nu >= 710) & (nu <= 740)
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.plot(nu[zm], r_p[zm], "o-", color="0.5", ms=2.5, lw=0.8, label="prior (fixed O$_3$)")
axZ.plot(nu[zm], r_j[zm], "o-", color="C3",  ms=3.0, lw=1.1, label="joint (O$_3$ retrieved)")
for x in (722.75, 723.25):
    axZ.annotate("", xy=(x, r_j[np.argmin(abs(nu-x))]), xytext=(x, r_p[np.argmin(abs(nu-x))]),
                 arrowprops=dict(arrowstyle="->", color="C2", lw=1.2))
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("710–740 zoom: retrieving O$_3$ clears\nthe 722–723 overshoot", fontsize=10)
axZ.legend(loc="upper right", fontsize=8); axZ.grid(True, alpha=0.3)

# annotation of retrieved reduced params
txt = (f"Retrieved amounts (× climatology)\n"
       f"O$_3$ total column: {o3_scale:.3f}×\n"
       + "\n".join(f"H$_2$O {p}: {s:.2f}×" for p, s in h2o))
axZ.text(0.03, 0.03, txt, transform=axZ.transAxes, fontsize=8, va="bottom", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

fig.suptitle("Joint retrieval beats the fixed-O$_3$ prior: RMS 0.565 → 0.462 K, "
             "χ² 2297 → 1788 (O$_3$ 0.58×, moist boundary layer 1.67×)", fontsize=12)
out = "data/iasi_joint.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (prior rms {Rp:.4f}, joint rms {Rj:.4f})")
