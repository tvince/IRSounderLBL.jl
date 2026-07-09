#!/usr/bin/env python3
"""Line-culprit view of the 710-730 opposite-sign residual.

Top: per-species ΔBT (absorption depth) from the species-toggle run — O3 owns the
whole 710-730 band; H2O is flat (<0.1 K). Bottom: the VP_Y joint residual (F-y) with
the 715.75 (wants MORE O3) and 722-723 (want LESS O3) features marked. Both live on
O3 lines that differ in lower-state energy, so a single column scale can't reconcile them.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

inv = np.genfromtxt("data/inventory_715.csv", delimiter=",", names=True)
fit = np.genfromtxt("data/iasi_joint_fit.csv", delimiter=",", names=True)

nu   = inv["wavenumber_cm1"]
dO3  = inv["dO3_K"]
dH2O = inv["dH2O_K"]

fnu  = fit["wavenumber_cm1"]
rvy  = fit["res_joint_K"]
nedt = fit["nedt_K"]
zf   = (fnu >= 710) & (fnu <= 730)

fig, (axA, axR) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.12))

axA.plot(nu, dO3,  color="C2", lw=1.3, label="ΔBT from O$_3$  (absorption depth)")
axA.plot(nu, dH2O, color="C0", lw=1.1, label="ΔBT from H$_2$O")
axA.axhline(0, color="0.6", lw=0.8)
for x in (715.75, 722.75, 723.25):
    axA.axvline(x, color="0.7", lw=0.7, ls=":")
axA.set_ylabel("ΔBT [K]  (negative = absorbs)")
axA.set_title("710–730 cm$^{-1}$: O$_3$ owns the band, H$_2$O is negligible "
              "(species-toggle at the retrieved atmosphere)")
axA.legend(loc="lower left", fontsize=9); axA.grid(True, alpha=0.3)

axR.fill_between(fnu[zf], -nedt[zf], nedt[zf], color="0.75", alpha=0.5, lw=0,
                 label="scene ±NEΔT")
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(fnu[zf], rvy[zf], "o-", color="C3", ms=3, lw=1.0, label="VP_Y joint residual F$-$y")
for x, dy, tag in ((715.75, 1.0, "wants MORE O$_3$  (⟨E''⟩ 186)"),
                   (722.75, -1.4, "wants LESS O$_3$  (⟨E''⟩ 206)")):
    j = np.argmin(abs(fnu - x))
    axR.annotate(tag, xy=(x, rvy[j]), xytext=(x - 3, rvy[j] + dy), fontsize=8, color="0.2",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.9))
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.set_title("Opposite-sign residuals are both O$_3$ lines at different E'' "
              "→ a single column scale cannot fix them", fontsize=10)
axR.legend(loc="lower left", fontsize=8); axR.grid(True, alpha=0.3)
axR.set_xlim(710, 730)

out = "data/inventory_715.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
