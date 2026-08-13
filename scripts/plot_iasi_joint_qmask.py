#!/usr/bin/env python3
"""Spike-region blacklist experiment: do the sharp 715 / 720-Q / 725 residual spikes
come from a T(p) bias, or are they LOCAL spectral features the retrieval can't fit?

Compares two joint retrievals that differ ONLY by the channel blacklist:
  baseline  data/iasi_joint_fit.csv       — all channels IN the fit  (res_joint_K)
  masked    data/iasi_joint_qmask_fit.csv — spike regions HELD OUT   (res_joint_K)

Both spectra stay full-grid (channel_mask keeps y_fit/nu full), so the held-out spikes
are still plotted — those blocks are what the retrieval was NOT allowed to fit. If a
spike is LOCAL (not a T bias fed by other channels), its residual barely moves when the
neighbouring spike channels are removed from the fit.

Excluded regions must match JOINT_QMASK in the driver run (HELD_RANGES below).
T(p) is read from data/iasi_joint_qmask_Tp.csv (dump_qmask_Tp.jl).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tropical prior (JOINT_BASE=tropical): baseline + spike-masked joint fits.
base = np.genfromtxt("data/iasi_joint_tropical_fit.csv",       delimiter=",", names=True)
msk  = np.genfromtxt("data/iasi_joint_tropical_qmask_fit.csv", delimiter=",", names=True)
nu   = base["wavenumber_cm1"]
bto  = base["bt_obs_K"]
nedt = base["nedt_K"]
r_base = base["res_joint_K"]     # all channels in the fit
r_msk  = msk["res_joint_K"]      # spike regions held out

def rms(v): return np.sqrt(np.mean(v**2))
# Channels excluded from the retrieval — MUST match JOINT_QMASK in the driver run.
HELD_RANGES = [(715.0, 716.0), (719.0, 723.0), (724.5, 726.0)]
held = np.zeros_like(nu, dtype=bool)
for lo, hi in HELD_RANGES: held |= (nu >= lo) & (nu <= hi)
kept = ~held
SPIKE = (715.50, 715.75, 720.75, 725.50)          # representative spike channels
sidx  = [int(np.argmin(np.abs(nu - s))) for s in SPIKE]

Rb, Rm     = rms(r_base), rms(r_msk)
Rb_k, Rm_k = rms(r_base[kept]), rms(r_msk[kept])          # kept-channel rms
Rb_h, Rm_h = rms(r_base[held]), rms(r_msk[held])          # held-out block rms

try:
    tp = np.genfromtxt("data/iasi_joint_qmask_Tp.csv", delimiter=",", names=True)
    have_tp = True
except Exception:
    have_tp = False

def mark_excluded(ax):
    """Mark every excluded region: hatched red band with a light fill."""
    for lo, hi in HELD_RANGES:
        ax.axvspan(lo, hi, facecolor="none", edgecolor="C3", hatch="xx", lw=0.0,
                   alpha=0.9, zorder=0)
        ax.axvspan(lo, hi, color="C3", alpha=0.08, lw=0, zorder=0)

fig = plt.figure(figsize=(15, 7.5))
gs  = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1.6],
                       hspace=0.34, wspace=0.24)

nexcl = int(held.sum())

# (a) observed BT with the excluded spike channels marked
axO = fig.add_subplot(gs[0, 0])
axO.plot(nu, bto, color="C0", lw=0.8, zorder=3)
mark_excluded(axO)
axO.plot(nu[held], bto[held], "x", color="C3", ms=4, mew=1.0, zorder=4,
         label=f"excluded from retrieval ({nexcl} ch: 715, 720-Q, 725)")
axO.set_ylabel("obs BT [K]")
axO.set_title("IASI FOV #1 — spike-region blacklist experiment (Tropical prior, 645-800 cm$^{-1}$)")
axO.grid(True, alpha=0.3); axO.legend(loc="lower left", fontsize=8)
plt.setp(axO.get_xticklabels(), visible=False)

# (b) full-band residual: all channels in vs spikes held out
axR = fig.add_subplot(gs[1, 0], sharex=axO)
axR.fill_between(nu, -nedt, nedt, color="0.75", alpha=0.5, lw=0, label="scene $\\pm$NE$\\Delta$T")
mark_excluded(axR)
axR.axhline(0, color="0.5", lw=0.8)
axR.plot(nu, r_base, color="0.5", lw=0.6, label=f"all channels IN fit (rms {Rb:.3f})")
axR.plot(nu, r_msk,  color="C3",  lw=0.7, label=f"spikes HELD OUT (rms {Rm:.3f})")
axR.plot(nu[held], r_msk[held], "x", color="k", ms=5, mew=1.1, zorder=5,
         label="excluded channels")
axR.set_xlabel("wavenumber [cm$^{-1}$]"); axR.set_ylabel("F$-$y [K]")
axR.legend(loc="lower right", fontsize=8); axR.grid(True, alpha=0.3)

# (z) zoom 710-730: all three spikes and the held-out bands
axZ = fig.add_subplot(gs[0, 1])
zm = (nu >= 710) & (nu <= 730)
zheld = zm & held
axZ.fill_between(nu[zm], -nedt[zm], nedt[zm], color="0.75", alpha=0.5, lw=0)
mark_excluded(axZ)
axZ.axhline(0, color="0.5", lw=0.8)
axZ.plot(nu[zm], r_base[zm], "o-", color="0.5", ms=2.5, lw=0.8, label="in fit")
axZ.plot(nu[zm], r_msk[zm],  "o-", color="C3",  ms=3.0, lw=1.1, label="held out")
axZ.plot(nu[zheld], r_msk[zheld], "x", color="k", ms=6, mew=1.3, zorder=5,
         label="excluded")
axZ.set_xlabel("wavenumber [cm$^{-1}$]"); axZ.set_ylabel("F$-$y [K]")
axZ.set_title("710-730 zoom: the three spikes", fontsize=10)
axZ.legend(loc="upper right", fontsize=7, ncol=3, columnspacing=0.8, handletextpad=0.3)
axZ.grid(True, alpha=0.3)

# (t) retrieved temperature profiles: baseline vs masked, with the difference inset
axT = fig.add_subplot(gs[1, 1])
if have_tp:
    p = tp["pressure"]; TB = tp["T_baseline"]; TM = tp["T_masked"]; dT = TM - TB
    axT.plot(TB, p, "o-", color="0.5", ms=2.5, lw=1.0, label="all channels in fit")
    axT.plot(TM, p, "o-", color="C3",  ms=2.5, lw=1.0, label="spikes held out")
    axT.set_yscale("log"); axT.invert_yaxis()
    axT.set_xlabel("temperature [K]"); axT.set_ylabel("pressure [hPa]")
    axT.set_title(f"retrieved T(p)   (max |$\\Delta$T| = {np.max(np.abs(dT)):.2f} K)",
                  fontsize=10)
    axT.grid(True, alpha=0.3, which="both"); axT.legend(loc="upper left", fontsize=8)
    # inset: masked - baseline
    axi = axT.inset_axes([0.60, 0.55, 0.37, 0.40])
    axi.plot(dT, p, "-", color="C4", lw=1.0); axi.axvline(0, color="0.5", lw=0.6)
    axi.set_yscale("log"); axi.invert_yaxis()
    axi.tick_params(labelsize=6); axi.set_title("$\\Delta$T [K]", fontsize=7)
    axi.grid(True, alpha=0.3, which="both")
else:
    axT.axis("off")
    axT.text(0.5, 0.5, "T(p) CSV not found\n(run dump_qmask_Tp.jl)", ha="center", va="center",
             transform=axT.transAxes, fontsize=9)

# how much the spikes themselves move when their neighbours are dropped
spike_shift = np.max([abs(r_msk[j] - r_base[j]) for j in sidx])
fig.suptitle(
    f"Held-out spikes rms {Rb_h:.3f}$\\to${Rm_h:.3f} K  |  kept-channel rms {Rb_k:.3f}$\\to${Rm_k:.3f} K  |  "
    f"max spike shift {spike_shift:.2f} K  "
    f"($\\to$ {'LOCAL, not a T bias' if spike_shift < 0.3 else 'coupled to T'})",
    fontsize=12)
out = "data/iasi_joint_qmask.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
print(f"  excluded {nexcl} channels over {HELD_RANGES}")
print(f"  full-band rms  in-fit {Rb:.4f}  held-out-run {Rm:.4f}")
print(f"  kept-channel   in-fit {Rb_k:.4f}  held-out-run {Rm_k:.4f}")
print(f"  held-out block in-fit {Rb_h:.4f}  held-out-run {Rm_h:.4f}")
for s, j in zip(SPIKE, sidx):
    print(f"  {s:.2f}: {r_base[j]:+.3f} -> {r_msk[j]:+.3f} K")
