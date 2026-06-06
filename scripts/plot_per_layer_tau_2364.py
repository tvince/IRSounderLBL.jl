"""
Apples-to-apples comparison of per-layer optical depth at 2364.105 cm⁻¹
between Julia (instrumented in scripts/instrument_2364_line.jl) and LBLRTM
(from ODint_NNN files, IOD=1 run).

This is the most direct test of the cross-section recipe divergence.  If
Julia and LBLRTM agree per-layer on τ(2364.105), the line-core BT must
agree.  If they diverge — and the divergence is concentrated in the
saturating layers (z ≈ 95–105 km for this line) — we have the smoking gun.

Plots two panels:
  Top:    per-layer τ(2364.105) vs altitude — Julia and LBLRTM overlaid
  Bottom: relative difference (LBLRTM − Julia) / Julia, in %

Usage:
  python scripts/plot_per_layer_tau_2364.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JUL = np.genfromtxt("data/lblrtm/instrument_2364_line.csv", delimiter=",", names=True)
LBL = np.genfromtxt("data/lblrtm/lblrtm_per_layer_tau_2364.csv", delimiter=",", names=True)

assert len(JUL) == len(LBL), f"row mismatch: Julia {len(JUL)} LBLRTM {len(LBL)}"
n = len(JUL)
z = 0.5 * (JUL["z_from_km"] + JUL["z_to_km"])

tau_jul = JUL["tau_layer_jul"]      # Julia native pipeline
tau_lbl = LBL["tau_at_nu0"]         # LBLRTM ODint at 2364.105

# Cumulative τ from TOA down
def cum_from_top(t):
    return np.cumsum(t[::-1])[::-1]

cum_jul = cum_from_top(tau_jul)
cum_lbl = cum_from_top(tau_lbl)


def panel_summary(z, t, label):
    print(f"\n=== {label} per-layer τ(2364.105) ===")
    print(f"  total column τ           = {t.sum():.3e}")
    cum = cum_from_top(t)
    k_sat = np.where(cum >= 1.0)[0][-1] if (cum >= 1.0).any() else None
    if k_sat is not None:
        print(f"  saturation level (top)   = layer {k_sat+1}  z={z[k_sat]:.2f} km   "
              f"τ_cum={cum[k_sat]:.3f}")
    print(f"  peak τ_layer             = {t.max():.3e}  at z={z[np.argmax(t)]:.2f} km")


panel_summary(z, tau_jul, "Julia")
panel_summary(z, tau_lbl, "LBLRTM")

ratio = tau_lbl / np.where(tau_jul != 0, tau_jul, np.nan)

fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

ax[0].plot(z, tau_jul, color="tab:red",  lw=1.2, marker="o", ms=4, label="Julia")
ax[0].plot(z, tau_lbl, color="tab:blue", lw=1.2, marker="o", ms=4,
           alpha=0.8, label="LBLRTM (ODint, IOD=1)")
ax[0].set_yscale("log")
ax[0].set_ylabel(r"$\tau_{layer}(2364.105\ \rm{cm}^{-1})$")
ax[0].set_title("Per-layer optical depth at the band-head line core")
ax[0].grid(True, alpha=0.3, which="both")
ax[0].legend()

ax[1].plot(z, cum_jul, color="tab:red",  lw=1.5, marker="o", ms=4, label="Julia")
ax[1].plot(z, cum_lbl, color="tab:blue", lw=1.5, marker="o", ms=4,
           alpha=0.8, label="LBLRTM")
ax[1].axhline(1.0, color="grey", lw=0.7, ls="--", label=r"$\tau = 1$")
ax[1].set_yscale("log")
ax[1].set_ylabel(r"cumulative $\tau$ from TOA")
ax[1].set_title("Cumulative τ from TOA: saturation altitude")
ax[1].grid(True, alpha=0.3, which="both")
ax[1].legend()

ax[2].plot(z, (ratio - 1.0) * 100.0, color="tab:purple",
           lw=1.2, marker="o", ms=4)
ax[2].axhline(0, color="grey", lw=0.5)
ax[2].set_ylabel("(LBLRTM / Julia − 1) × 100  [%]")
ax[2].set_title("Per-layer LBLRTM-vs-Julia cross-section divergence")
ax[2].set_xlabel("layer mid-altitude [km]")
ax[2].axvspan(85, 115, color="orange", alpha=0.12,
              label="line-core saturation region")
ax[2].legend()
ax[2].grid(True, alpha=0.3)
ax[2].set_ylim(-20, 20)

fig.suptitle(
    "Per-layer τ(2364.105 cm$^{-1}$): LBLRTM (IOD=1 ODint) vs Julia (compute_voigt_cross_sections)\n"
    f"Total column τ — Julia {tau_jul.sum():.3e},  LBLRTM {tau_lbl.sum():.3e}",
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "data/lblrtm/per_layer_tau_2364.png"
plt.savefig(out, dpi=140)
print(f"\nwrote {out}")
