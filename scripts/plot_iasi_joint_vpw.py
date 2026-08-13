#!/usr/bin/env python3
"""Three-way joint retrieval comparison: fixed-O3 prior vs VP_Y joint vs VP_W joint.

Reads data/iasi_joint_fit.csv (res_joint = VP_Y) and data/iasi_joint_vpw_fit.csv
(res_joint = VP_W); res_prior is identical in both. Headline: VP_W (0.486 K) does
NOT beat VP_Y (0.462 K); the 715.75 residual (wants MORE O3) and the 722-723 region
(want LESS O3) pull in opposite directions, so a single O3 column scale can't satisfy
both — a spectral-shape issue, not a line-mixing-order one.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

vy = np.genfromtxt("data/iasi_joint_fit.csv",     delimiter=",", names=True)
vw = np.genfromtxt("data/iasi_joint_vpw_fit.csv", delimiter=",", names=True)
nu   = vy["wavenumber_cm1"]
bto  = vy["bt_obs_K"]
nedt = vy["nedt_K"]
r_pr = vy["res_prior_K"]      # fixed-O3, T-only (same in both files)
r_vy = vy["res_joint_K"]      # VP_Y joint
r_vw = vw["res_joint_K"]      # VP_W joint

def rms(v): return np.sqrt(np.mean(v**2))
Rp, Ry, Rw = rms(r_pr), rms(r_vy), rms(r_vw)
QWIN = [(700.0, 710.0), (719.0, 724.0)]

fig = plt.figure(figsize=(15, 6.5))
gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1.6],
                       hspace=0.30, wspace=0.20)

# (a) observed BT
axO = fig.add_subplot(gs[0, 0])
axO.plot(nu, bto, color="C0", lw=0.8)
for lo, hi in QWIN: axO.axvspan(lo, hi, color="C1", alpha=0.15)
axO.set_ylabel("obs BT [K]")
axO.set_title("IASI FOV #1 — joint T + H$_2$O + O$_3$ column: VP_Y vs VP_W (645–800 cm$^{-1}$)")
axO.grid(True, alpha=0.3); plt.setp(axO.get_xticklabels(), visible=False)

# (b) residuals: prior vs VP_Y vs VP_W
axR = fig.add_subplot(gs[1, 0], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
for lo, hi in QWIN: axR.axvspan(lo, hi, color="C1", alpha=0.15)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_pr, color="0.6", lw=0.6,            label=f"prior fixed-O$_3$ (rms {Rp:.3f})")
axR.plot(nu, r_vy, color="C3",  lw=0.7,            label=f"VP_Y joint (rms {Ry:.3f})")
axR.plot(nu, r_vw, color="C0",  lw=0.7, alpha=0.85,label=f"VP_W joint (rms {Rw:.3f})")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("VP_W does not beat VP_Y (0.486 vs 0.462 K)", fontsize=10)
axR.legend(loc="lower right", fontsize=8, ncol=2); axR.grid(True, alpha=0.3)

# (z) zoom 710-740 — the opposite-sign O3 story
axZ = fig.add_subplot(gs[:, 1])
zm = (nu >= 710) & (nu <= 740)
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.plot(nu[zm], r_pr[zm], "o-", color="0.6", ms=2.5, lw=0.8, label="prior fixed-O$_3$")
axZ.plot(nu[zm], r_vy[zm], "o-", color="C3",  ms=3.0, lw=1.1, label="VP_Y joint")
axZ.plot(nu[zm], r_vw[zm], "s-", color="C0",  ms=2.8, lw=0.9, label="VP_W joint")
for x, tag in ((715.75, "wants MORE O$_3$"), (722.75, "wanted LESS O$_3$")):
    j = np.argmin(abs(nu - x))
    axZ.annotate(tag, xy=(x, r_vy[j]), xytext=(x-2, r_vy[j] + (1.2 if x < 720 else -1.4)),
                 fontsize=7.5, color="0.2",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("710–740: 715 & 722 pull O$_3$ opposite ways\n(shape issue, not LM order)", fontsize=10)
axZ.legend(loc="lower right", fontsize=8); axZ.grid(True, alpha=0.3)

fig.suptitle("Joint retrieval: VP_Y wins (RMS 0.462 K); full-matrix VP_W is marginally "
             "worse (0.486 K) — O$_3$ column 0.58× vs 0.60×, opposite-sign 715/722 residuals",
             fontsize=12)
out = "data/iasi_joint_vpw.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (prior {Rp:.4f}, VP_Y {Ry:.4f}, VP_W {Rw:.4f})")
