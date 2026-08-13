#!/usr/bin/env python3
"""
Steepness-conditioned uniqueness test for the 715.5/715.75 spike.

Any BAND-WIDE instrument-model error produces a residual that is a function of the
local spectral derivatives of the observed spectrum:
    • registration / shift error δ  →  res ≈ −δ·F'   (∝ first derivative)
    • ILS-width / shape / grid error →  res ≈  c·F''  (∝ second derivative)
So under any such mechanism the whitened residual should be well modelled, with
BAND-UNIVERSAL coefficients, by a linear combination of F' and F''. If we fit that
relation on the rest of the band and the 715.5/715.75 residuals sit far above the
prediction (i.e. they are outliers EVEN after conditioning on how steep the spectrum
is there), then no band-wide instrument mechanism can explain them → the cause is
line/channel-specific. If instead every equally-steep channel has an equally large
residual, the mechanism is band-wide.

Pure arithmetic on data/iasi_joint_fit.csv (no forward runs).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "data/iasi_joint_fit.csv"
d = np.genfromtxt(CSV, delimiter=",", names=True)
nu   = d["wavenumber_cm1"]
obs  = d["bt_obs_K"]
res  = d["res_joint_K"]
nedt = d["nedt_K"]
dnu  = 0.25

# Local spectral derivatives of the OBSERVED spectrum (central differences).
Fp  = np.gradient(obs, dnu)                       # F'  (K per cm-1)
Fpp = np.gradient(Fp, dnu)                        # F'' (K per cm-2)
w   = res / nedt                                  # whitened residual (sigma units)

# Channel indices of interest.
def ch(x): return int(np.argmin(np.abs(nu - x)))
spikes = [ch(715.50), ch(715.75)]
spike_mask = np.zeros(len(nu), bool); spike_mask[spikes] = True

# Comparators: steep channels elsewhere in the band (CO2 Q-branch flanks ~667,
# strong H2O lines 712.7 / 729.3, band edges).
print("=" * 78)
print("STEEPNESS RANKING — where do 715.50/715.75 sit among the steepest channels?")
print("=" * 78)
for label, metric in (("|F'| (shift lever)", np.abs(Fp)),
                      ("|F''| (width/shape lever)", np.abs(Fpp))):
    order = np.argsort(metric)[::-1]
    print(f"\nTop 12 by {label}:")
    print("  {:>9} {:>8} {:>9} {:>9} {:>8} {:>8}".format(
          "nu", "obs", "Fp", "Fpp", "res_K", "res_sig"))
    for i in order[:12]:
        mark = "  <== SPIKE" if spike_mask[i] else ""
        print(f"  {nu[i]:9.2f} {obs[i]:8.2f} {Fp[i]:9.2f} {Fpp[i]:9.1f}"
              f" {res[i]:8.3f} {w[i]:8.2f}{mark}")

# --- Regression: fit whitened residual on {F', F''} over the band, EXCLUDING the
# spike, then predict the spike and measure how many scatter-sigma it deviates. ---
fitmask = ~spike_mask & np.isfinite(w)
X = np.column_stack([np.ones(fitmask.sum()), Fp[fitmask], Fpp[fitmask]])
coef, *_ = np.linalg.lstsq(X, w[fitmask], rcond=None)
pred_all = coef[0] + coef[1] * Fp + coef[2] * Fpp
resid_fit = w[fitmask] - pred_all[fitmask]
sig = resid_fit.std()
R2 = 1.0 - np.var(resid_fit) / np.var(w[fitmask])

print("\n" + "=" * 78)
print("BAND-WIDE MODEL:  res/nedt ≈ a + b·F' + c·F''   (fit on band, spike excluded)")
print("=" * 78)
print(f"  a={coef[0]:+.4f}  b(shift)={coef[1]:+.5f}  c(width)={coef[2]:+.6f}")
print(f"  R² = {R2:.3f}   scatter σ = {sig:.3f} σ-units")
print(f"\n  Prediction vs actual at the spike (how well steepness alone explains it):")
print(f"  {'nu':>9} {'res_sig':>9} {'pred_sig':>9} {'excess':>9} {'excess/σ':>9}")
for i in spikes:
    excess = w[i] - pred_all[i]
    print(f"  {nu[i]:9.2f} {w[i]:9.2f} {pred_all[i]:9.2f} {excess:9.2f} {excess/sig:9.1f}")

# How many equally-or-steeper channels exist, and what are THEIR residuals?
for i in spikes:
    thr = abs(Fp[i])
    comp = (np.abs(Fp) >= 0.8 * thr) & ~spike_mask
    print(f"\n  Channels with |F'| ≥ 0.8×|F'|@{nu[i]:.2f} ({thr:.1f}): n={comp.sum()}")
    print(f"    their |res_sig|: median={np.median(np.abs(w[comp])):.2f}  "
          f"max={np.max(np.abs(w[comp])):.2f}   (spike |res_sig|={abs(w[i]):.2f})")

# --- Plot: whitened residual vs the two steepness levers ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, lev, name in ((ax[0], Fp, "F'  (local slope, K/cm⁻¹)"),
                     (ax[1], Fpp, "F''  (local curvature, K/cm⁻²)")):
    a.axhline(0, color="k", lw=0.5)
    a.scatter(lev[~spike_mask], w[~spike_mask], s=9, c="0.6", label="band")
    a.scatter(lev[spikes], w[spikes], s=70, c="crimson", zorder=5,
              label="715.50 / 715.75")
    for i in spikes:
        a.annotate(f"{nu[i]:.2f}", (lev[i], w[i]), fontsize=8,
                   xytext=(4, 2), textcoords="offset points", color="crimson")
    a.set_xlabel(name); a.set_ylabel("whitened residual  res/NEΔT  (σ)")
    a.legend(fontsize=8, loc="upper left")
ax[0].set_title("Residual vs shift lever")
ax[1].set_title("Residual vs width/shape lever")
fig.suptitle("Steepness-conditioned test: is the 715 spike explained by local steepness?")
fig.tight_layout()
fig.savefig("data/o3_steepness_test.png", dpi=130)
print("\nwrote data/o3_steepness_test.png")
