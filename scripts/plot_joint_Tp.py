#!/usr/bin/env python3
"""Retrieved T(p) from the joint retrieval vs the fixed-O3 T-only prior and the
AFGL first guess. Left: profiles with ±sigma_post (log-p). Right: increment vs the
first guess. T_sfc shown as a marker below the grid."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt("data/iasi_joint_Tp.csv", delimiter=",", names=True)
p = d["p_hPa"]
m = p > 1e-2                                   # drop the ~0 hPa top level for log axis

fig, (axT, axD) = plt.subplots(1, 2, figsize=(11, 7.5), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.08))

axT.plot(d["T_fg_K"][m], p[m], "--", color="0.55", lw=1.2, label="AFGL first guess")
axT.errorbar(d["T_prior_K"][m], p[m], xerr=d["sig_prior_K"][m], fmt="o-", color="0.4",
             ms=2.5, lw=1.0, elinewidth=0.6, capsize=1.5, label="prior: fixed O$_3$, T-only ±σ")
axT.errorbar(d["T_vpy_K"][m], p[m], xerr=d["sig_vpy_K"][m], fmt="o-", color="C3",
             ms=2.7, lw=1.2, elinewidth=0.6, capsize=1.5, label="VP_Y joint ±σ")
axT.errorbar(d["T_vpw_K"][m], p[m], xerr=d["sig_vpw_K"][m], fmt="s-", color="C0",
             ms=2.4, lw=0.9, elinewidth=0.5, capsize=1.2, alpha=0.8, label="VP_W joint ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("retrieved T(p)")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

axD.axvline(0, color="0.6", lw=0.8)
axD.plot(d["T_prior_K"][m] - d["T_fg_K"][m], p[m], "o-", color="0.4", ms=2.5, lw=1.0,
         label="prior $-$ first guess")
axD.plot(d["T_vpy_K"][m] - d["T_fg_K"][m], p[m], "o-", color="C3", ms=2.7, lw=1.2,
         label="VP_Y joint $-$ first guess")
axD.set_xlabel("$\\Delta T$ [K]"); axD.set_title("increment vs first guess")
axD.grid(True, which="both", alpha=0.3); axD.legend(loc="upper left", fontsize=8)

fig.suptitle("IASI FOV #1 joint retrieval — T(p)  "
             "(T$_{sfc}$: fg 288.2 → prior 293.9 → VP_Y 294.5 ± 0.17 K)", fontsize=12)
out = "data/iasi_joint_Tp.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
