#!/usr/bin/env python3
"""Per-channel IASI measurement noise in brightness-temperature space, from the
EUMETSAT L1C Noise Covariance Matrix (NCM) linearized at this FOV's observed BT
scene, compared with the analytic apodized scene covariance used previously.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.genfromtxt("data/ncm_noise_bt.csv", delimiter=",", names=True)
nu, yobs, sn, sa = d["wavenumber_cm1"], d["bt_obs_K"], d["sigma_ncm_K"], d["sigma_analytic_K"]

fig, (ax, axb) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios":[2.2,1]})

ax.plot(nu, sn, color="C0", lw=0.9, label=f"EUMETSAT NCM  (median {np.median(sn):.3f} K)")
ax.plot(nu, sa, color="C3", lw=0.9, alpha=0.8,
        label=f"analytic scene cov  (median {np.median(sa):.3f} K)")
for x in (715.75, 722.75, 723.25):
    ax.axvline(x, color="0.6", lw=0.7, ls=":")
ax.set_ylabel("NE$\\Delta$T  [K]")
ax.set_title("IASI per-channel noise (BT) — EUMETSAT NCM vs analytic scene covariance, "
             "645–800 cm$^{-1}$")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)

# scene BT on a twin axis for context (cold Q-branch = higher BT noise)
axt = ax.twinx()
axt.plot(nu, yobs, color="0.7", lw=0.7, alpha=0.6)
axt.set_ylabel("observed BT [K]", color="0.6", fontsize=8)
axt.tick_params(axis="y", colors="0.6", labelsize=7)

axb.plot(nu, sn/sa, color="C2", lw=0.9)
axb.axhline(1.0, color="0.5", lw=0.8, ls="--")
axb.set_ylabel("NCM / analytic"); axb.set_xlabel("wavenumber [cm$^{-1}$]")
axb.set_title(f"ratio (median {np.median(sn/sa):.2f} — NCM is ~2× tighter)", fontsize=10)
axb.grid(True, alpha=0.3)

out = "data/ncm_noise_bt.png"
fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}  (NCM median {np.median(sn):.3f} K, analytic {np.median(sa):.3f} K, "
      f"ratio {np.median(sn/sa):.2f})")
