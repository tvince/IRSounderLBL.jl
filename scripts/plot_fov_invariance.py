#!/usr/bin/env python3
"""Two-panel figure for the 715 spike investigation.
  (A) The FOV#1 residual spectrum over 710-730: the sharp +10σ 2-channel spike at the
      strongest O3 line (715.71), with the scene ±NEΔT band.
  (B) FOV-invariance: the high-passed (sharp) residual at 715.50 & 715.75 across the 8
      clearest FOVs. Tight clustering (std ≪ mean) ⇒ SYSTEMATIC, not atmospheric.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt("data/iasi_joint_fit.csv", delimiter=",", names=True)
nu, res, nedt = d["wavenumber_cm1"], d["res_joint_K"], d["nedt_K"]
f = np.genfromtxt("data/o3_fov_invariance.csv", delimiter=",", names=True)
fov = f["fov"].astype(int)
rs50, rs75 = f["resSharp_71550"], f["resSharp_71575"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})

# ── Panel A: residual spectrum with the spike ───────────────────────────────────────
z = (nu >= 710) & (nu <= 730)
axA.fill_between(nu[z], -nedt[z], nedt[z], color="0.8", alpha=0.6, lw=0, label="scene ±NEΔT")
axA.axhline(0, color="0.5", lw=0.8)
axA.plot(nu[z], res[z], "o-", color="C3", ms=3.2, lw=1.0, label="residual F$-$y (FOV #1)")
axA.annotate("+10σ spike at the\nstrongest O$_3$ line (715.71)",
             xy=(715.55, res[np.argmin(abs(nu-715.5))]), xytext=(718.5, 2.6), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))
axA.set_xlim(710, 730); axA.set_ylim(-1.4, 3.7)
axA.set_xlabel("wavenumber [cm$^{-1}$]"); axA.set_ylabel("F $-$ y  [K]")
axA.set_title("(A) The residual: a sharp 2-channel spike\nno smooth O$_3$/T/continuum mechanism reaches it", fontsize=10)
axA.legend(loc="lower right", fontsize=8); axA.grid(True, alpha=0.3)

# ── Panel B: FOV-invariance of the sharp component ──────────────────────────────────
for y, lab, c in ((rs50, "715.50", "C0"), (rs75, "715.75", "C1")):
    m, s = y.mean(), y.std()
    axB.axhspan(m - s, m + s, color=c, alpha=0.13, lw=0)
    axB.axhline(m, color=c, lw=1.2, ls="--")
    axB.plot(fov, y, "o", color=c, ms=7, label=f"ν={lab}:  {m:+.2f} ± {s:.2f} K")
axB.set_xlabel("FOV index (8 clearest, 0% cloud)")
axB.set_ylabel("high-passed residual  [K]")
axB.set_ylim(0, 3)
axB.set_title("(B) …but it is CONSTANT across 8 clear FOVs\nstd ≪ mean  ⇒  SYSTEMATIC, not atmospheric", fontsize=10)
axB.legend(loc="lower center", fontsize=8, title="sharp residual (mean ± std)")
axB.grid(True, alpha=0.3)

fig.suptitle("715 cm$^{-1}$ residual spike — reproducible model-vs-obs mismatch at the steepest O$_3$ edge",
             fontsize=11, y=1.01)
fig.tight_layout()
out = "data/o3_fov_invariance.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
