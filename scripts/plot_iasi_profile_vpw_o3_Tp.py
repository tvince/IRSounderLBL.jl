#!/usr/bin/env python3
"""Plot retrieved T(p) for the VP_W+O3 experiment: VP_Y no-O3 / VP_Y +O3 / VP_W +O3.
Left: T(p) with +/-sigma_post (log-p). Right: differences vs the VP_Y no-O3 profile,
showing where O3 and the full-matrix line mixing move the temperature solution.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tp = np.genfromtxt("data/iasi_profile_vpw_o3_Tp.csv", delimiter=",", names=True)
p    = tp["p_hPa"]
T_a  = tp["T_vpy_noO3_K"]; s_a = tp["sig_vpy_noO3_K"]
T_b  = tp["T_vpy_O3_K"];   s_b = tp["sig_vpy_O3_K"]
T_c  = tp["T_vpw_O3_K"];   s_c = tp["sig_vpw_O3_K"]
m = p > 1e-3

fig, (axT, axD) = plt.subplots(1, 2, figsize=(11, 7), sharey=True,
                               gridspec_kw=dict(width_ratios=[1.4, 1], wspace=0.08))

axT.errorbar(T_a[m], p[m], xerr=s_a[m], fmt="o-", color="0.5", ms=2.5, lw=1.0,
             ecolor="0.5", elinewidth=0.6, capsize=1.5, label="VP_Y no-O$_3$ ±σ")
axT.errorbar(T_b[m], p[m], xerr=s_b[m], fmt="o-", color="C0", ms=2.5, lw=1.0,
             ecolor="C0", elinewidth=0.6, capsize=1.5, label="VP_Y +O$_3$ ±σ")
axT.errorbar(T_c[m], p[m], xerr=s_c[m], fmt="o-", color="C3", ms=2.5, lw=1.3,
             ecolor="C3", elinewidth=0.6, capsize=1.5, label="VP_W +O$_3$ ±σ")
axT.set_yscale("log"); axT.invert_yaxis()
axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
axT.set_title("retrieved T(p)")
axT.grid(True, which="both", alpha=0.3); axT.legend(loc="upper left", fontsize=8)

axD.axvline(0, color="0.6", lw=0.8)
axD.plot(T_b[m] - T_a[m], p[m], "s-", color="C0", ms=3, lw=1.1, label="+O$_3$ $-$ no-O$_3$ (VP_Y)")
axD.plot(T_c[m] - T_a[m], p[m], "o-", color="C3", ms=3, lw=1.3, label="VP_W+O$_3$ $-$ VP_Y no-O$_3$")
axD.set_xlabel("$\\Delta T$ [K]"); axD.set_title("shift vs VP_Y no-O$_3$")
axD.grid(True, which="both", alpha=0.3); axD.legend(loc="upper left", fontsize=8)

fig.suptitle("IASI FOV #1 retrieved T(p): O$_3$ + full-matrix line mixing "
             "(T$_{sfc}$: 293.72 → 293.95 → 293.98 K)", fontsize=12)
out = "data/iasi_profile_vpw_o3_Tp.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
