#!/usr/bin/env python3
"""ILS/spectral-registration reachability for the 715 residual spike. The spike is a sharp
2-channel +10σ feature at the strongest O3 line (715.71) with a derivative (+/-) shape ->
classic ILS/calibration signature. Two ILS error modes are analytic from the modeled
spectrum F = y + res (no forward runs): a registration (wavenumber-shift) error gives
residual ∝ F'(ν); an ILS-width error gives residual ∝ F''(ν) (convolving with a slightly
wrong Gaussian adds ∝ Δσ²·F''). Project the NEΔT-whitened residual onto {F', F''}.

RESULT (2026-07-09): only ~11% globally / 22% at the spike -> ILS/registration is a MINOR
contributor, NOT the cause. The spike resists shift AND width. See o3_fov_invariance.jl for
the systematic-vs-atmospheric fork.

    python3 scripts/o3_ils_shift_reachability.py
"""
import numpy as np

d = np.genfromtxt("data/iasi_joint_fit.csv", delimiter=",", names=True)
nu, y, res, nedt = d["wavenumber_cm1"], d["bt_obs_K"], d["res_joint_K"], d["nedt_K"]
F = y + res                      # modeled BT (smooth) = y + (F - y)
Fp  = np.gradient(F, nu)         # F'  ~ registration (shift) basis
Fpp = np.gradient(Fp, nu)        # F'' ~ ILS-width basis


def proj_report(lo, hi, label):
    m = (nu >= lo) & (nu <= hi)
    w = 1.0 / nedt[m]
    r = res[m] * w
    b1, b2 = Fp[m] * w, Fpp[m] * w

    def frac(G):
        Q, _ = np.linalg.qr(G)
        p = Q @ (Q.T @ r)
        return np.sum(p ** 2) / np.sum(r ** 2), p

    f1, _ = frac(b1[:, None])
    f2, _ = frac(b2[:, None])
    G = np.column_stack([b1, b2])
    f12, p12 = frac(G)
    coef, *_ = np.linalg.lstsq(G, r, rcond=None)
    print(f"\n-- {label}  [{lo},{hi}]  ({m.sum()} ch) --")
    print(f"  explained:  shift(F') {100*f1:5.1f}%   ILS-width(F'') {100*f2:5.1f}%   both {100*f12:5.1f}%")
    print(f"  implied shift dnu = {-coef[0]*1000:+.1f} mcm-1  ({-coef[0]/0.25*100:+.1f}% of a channel)")
    rms0 = np.sqrt(np.mean(res[m] ** 2))
    rms1 = np.sqrt(np.mean(((r - p12) * nedt[m]) ** 2))
    print(f"  710-730 RMS: {rms0:.3f} K  ->  {rms1:.3f} K after {{F',F''}} fit")


proj_report(710, 730, "full window")
proj_report(714, 717, "715 spike only")
proj_report(721, 725, "722-725 region")
