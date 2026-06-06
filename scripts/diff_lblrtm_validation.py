#!/usr/bin/env python3
"""
Diff the single-session Julia validation outputs (validate_lblrtm_all.jl) against
the LBLRTM references and report RMS / worst-point per band.

LBLRTM is on a much finer grid (~0.0002-0.0005 cm-1) than Julia (0.005 cm-1), so
LBLRTM BT is linearly interpolated onto the Julia grid before differencing. The
comparison window is trimmed by `EDGE` cm-1 on each side to drop the band-edge
roll-off where the two grids don't fully overlap.

Run with:
  python scripts/diff_lblrtm_validation.py
"""

import numpy as np
import os

DATADIR = "data/lblrtm"
EDGE = 0.5  # cm-1 trimmed from each band edge before computing stats

# (label, julia_csv, lblrtm_csv, expected note)
PAIRS = [
    ("15 um  cont-OFF", "julia_bt_contOFF.csv",      "lblrtm_bt_contOFF_g12.csv",
     "~0.08 K (no regression from :cim)"),
    ("15 um  cont-ON ", "julia_bt_contON.csv",       "lblrtm_bt_contON_g12.csv",
     "~0.44 K (no regression from :cim)"),
    ("4.3 um cont-OFF", "julia_bt_43um_contOFF.csv", "lblrtm_bt_43um_contOFF.csv",
     "CIM should be well below the old +7 K comb"),
    ("4.3 um CO2-cont", "julia_bt_43um_co2cont.csv", "lblrtm_bt_43um_co2cont.csv",
     "MT-CKD CO2 continuum cross-check"),
]


def load_julia(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def load_lblrtm(path):
    # cols: wavenumber, radiance, transmittance, BT_K  -> take 0 and 3
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 3))
    return a[:, 0], a[:, 1]


def diff_pair(label, jcsv, lcsv, note):
    jpath, lpath = os.path.join(DATADIR, jcsv), os.path.join(DATADIR, lcsv)
    if not (os.path.exists(jpath) and os.path.exists(lpath)):
        print(f"{label}:  SKIP (missing {jcsv if not os.path.exists(jpath) else lcsv})")
        return
    nu_j, bt_j = load_julia(jpath)
    nu_l, bt_l = load_lblrtm(lpath)
    # interpolate LBLRTM onto Julia grid, restricted to the overlap minus EDGE
    lo = max(nu_j.min(), nu_l.min()) + EDGE
    hi = min(nu_j.max(), nu_l.max()) - EDGE
    m = (nu_j >= lo) & (nu_j <= hi)
    nu = nu_j[m]
    bt_li = np.interp(nu, nu_l, bt_l)
    d = bt_j[m] - bt_li
    rms = np.sqrt(np.mean(d**2))
    k = np.argmax(np.abs(d))
    print(f"{label}:  RMS {rms:7.4f} K   worst {d[k]:+7.3f} K @ {nu[k]:.3f} cm-1"
          f"   (mean {d.mean():+.4f}, n={d.size})")
    print(f"{'':17s}  expect: {note}")


def main():
    print("Julia (:cim default) - LBLRTM, interpolated onto Julia grid, "
          f"edges trimmed {EDGE} cm-1\n")
    for p in PAIRS:
        diff_pair(*p)


if __name__ == "__main__":
    main()
