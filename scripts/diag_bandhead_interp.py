#!/usr/bin/env python3
"""
Test whether the 2386-2398 cm-1 band-head -3.9 K Julia-LBLRTM residual is an
interpolation artifact. LBLRTM (~0.0005 cm-1) is ~10x finer than Julia (0.005),
so we can compare three ways and see if the band head moves:
  (1) linear interp  (what diff_lblrtm_validation.py does now)
  (2) nearest native LBLRTM point  (NO interpolation at all)
  (3) cubic interp   (higher order)
If all three agree, interpolation is exonerated.
"""
import numpy as np

J = np.loadtxt("data/lblrtm/julia_bt_43um_contOFF.csv", delimiter=",", skiprows=1)
L = np.loadtxt("data/lblrtm/lblrtm_bt_43um_contOFF.csv", delimiter=",",
               skiprows=1, usecols=(0, 3))
nu_j, bt_j = J[:, 0], J[:, 1]
nu_l, bt_l = L[:, 0], L[:, 1]

def stats(name, nu, d):
    rms = np.sqrt(np.mean(d**2)); k = np.argmax(np.abs(d))
    print(f"  {name:24s} RMS {rms:7.4f} K  mean {d.mean():+7.4f}  "
          f"worst {d[k]:+7.3f} @ {nu[k]:.3f}  (n={d.size})")

for lo, hi, tag in [(2386.0, 2398.0, "BAND HEAD 2386-2398"),
                    (2000.5, 2386.0, "below head"),
                    (2398.0, 2499.5, "above head")]:
    m = (nu_j >= lo) & (nu_j <= hi)
    nu = nu_j[m]; bj = bt_j[m]
    # (1) linear
    d_lin = bj - np.interp(nu, nu_l, bt_l)
    # (2) nearest native LBLRTM point (no interpolation)
    idx = np.searchsorted(nu_l, nu)
    idx = np.clip(idx, 1, len(nu_l) - 1)
    left = nu - nu_l[idx - 1]; right = nu_l[idx] - nu
    near = np.where(left <= right, idx - 1, idx)
    max_off = np.max(np.abs(nu - nu_l[near]))
    d_near = bj - bt_l[near]
    # (3) cubic
    try:
        from scipy.interpolate import CubicSpline
        d_cub = bj - CubicSpline(nu_l, bt_l)(nu)
        have_cub = True
    except Exception as e:
        have_cub = False
    print(f"\n{tag}  (max nearest-point offset {max_off*1e3:.4f} mK-grid = {max_off:.6f} cm-1)")
    stats("(1) linear interp", nu, d_lin)
    stats("(2) nearest native [NO interp]", nu, d_near)
    if have_cub:
        stats("(3) cubic spline", nu, d_cub)
