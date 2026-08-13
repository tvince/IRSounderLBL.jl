#!/usr/bin/env python3
"""O3-column joint vs O3-2-bulk-layer joint residual, 710-730 zoom. They overlie almost
exactly: resolving O3 vertically (the 2nd DOF opens in the upper stratosphere) does not
touch the 715/722 residual. The remaining feature is O3 strong-line spectroscopy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt("data/iasi_joint_o3layers_fit.csv", delimiter=",", names=True)
nu = d["wavenumber_cm1"]; nedt = d["nedt_K"]
r_pr = d["res_prior_K"]; r_col = d["res_o3col_K"]; r_lay = d["res_o3layers_K"]
def rms(v, m): return np.sqrt(np.mean(v[m]**2))
z = (nu >= 710) & (nu <= 730)

fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(nu[z], -nedt[z], nedt[z], color="0.75", alpha=0.5, lw=0, label="scene ±NEΔT")
ax.axhline(0, color="0.5", lw=0.8)
ax.plot(nu[z], r_pr[z],  "o-", color="0.6", ms=2.5, lw=0.7, alpha=0.7,
        label=f"fixed-O$_3$ prior (rms {rms(r_pr,z):.3f})")
ax.plot(nu[z], r_col[z], "o-", color="C0", ms=3.0, lw=1.0,
        label=f"O$_3$ column joint (rms {rms(r_col,z):.3f})")
ax.plot(nu[z], r_lay[z], "s--", color="C3", ms=3.0, lw=1.0,
        label=f"O$_3$ 2-bulk-layer joint (rms {rms(r_lay,z):.3f})")
for x, dy, tag in ((715.75, 0.8, "survives (+2.91 K)"),):
    j = np.argmin(abs(nu - x))
    ax.annotate(tag, xy=(x, r_lay[j]), xytext=(x - 3.5, r_lay[j] + dy), fontsize=8, color="0.2",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
ax.set_xlim(710, 730)
ax.set_xlabel("wavenumber [cm$^{-1}$]"); ax.set_ylabel("F$-$y [K]")
ax.set_title("Resolving O$_3$ vertically (2 bulk layers) does not move the 710–730 residual\n"
             "the 2nd O$_3$ DOF opens in the upper stratosphere → 715/722 unchanged "
             "→ O$_3$ strong-line spectroscopy", fontsize=10)
ax.legend(loc="lower left", fontsize=8); ax.grid(True, alpha=0.3)
out = "data/iasi_joint_o3layers.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
