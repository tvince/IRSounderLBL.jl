"""
Build a finer-layered version of the AFGL US Std 50-level profile by
inserting intermediate levels in the upper stratosphere/mesosphere
(50-95 km), where the 5-km layer spacing of the standard profile is
known to leave large finite-thickness source-function error in both Toon
LIT and CIM Padé approximations.

Strategy:
  - Keep all 50 original levels (0..120 km).
  - Insert new levels at z = 51, 52, ..., 94 km (44 new levels in the
    50-95 km saturation band for 4.3 µm CO₂).
  - Interpolate T linearly in z (LBLATM convention).
  - Interpolate p linearly in log(p) (hydrostatic-consistent for an
    isothermal scale-height region).
  - Interpolate VMR linearly in log(p) (same as Julia's interp_vmr).

The total comes to 94 levels.  Output CSV uses the same column layout as
data/afgl_us_standard_50lev.csv so both Julia (afgl_us_standard_50lev →
or a custom loader) and the LBLRTM TAPE5 builder can read it.

Usage:
  python scripts/build_afgl_subdiv.py
"""

import numpy as np
import csv
import argparse

IN  = "data/afgl_us_standard_50lev.csv"
OUT = "data/afgl_us_standard_subdiv.csv"

# Subdivision range: insert at STEP-km steps strictly between these (exclusive).
# Defaults match the original 50-95 km mesosphere study; override via CLI to
# extend into the 95-120 km thermosphere (where the 4.3 um band head forms and
# T rises +60 K per 5-km layer).
Z_LO, Z_HI = 50.0, 95.0
STEP       = 1.0


def main():
    global Z_LO, Z_HI, STEP, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-lo",  type=float, default=Z_LO)
    ap.add_argument("--z-hi",  type=float, default=Z_HI)
    ap.add_argument("--step",  type=float, default=STEP)
    ap.add_argument("-o", "--out", default=OUT)
    args = ap.parse_args()
    Z_LO, Z_HI, STEP, OUT = args.z_lo, args.z_hi, args.step, args.out

    # Load original profile
    a = np.genfromtxt(IN, delimiter=",", names=True)
    z = a["z_km"]
    p = a["p_hPa"]
    T = a["T_K"]
    vmrs = {n: a[n] for n in a.dtype.names if n.startswith("vmr_")}
    print(f"original profile: {len(z)} levels, z = {z[0]:.1f} … {z[-1]:.1f} km")

    # Build the new z grid: original z's + intermediate 1 km steps in (Z_LO, Z_HI)
    extra = np.arange(Z_LO + STEP, Z_HI, STEP)         # 51, 52, …, 94
    extra = np.array([zz for zz in extra if not np.any(np.isclose(z, zz))])
    z_new = np.sort(np.unique(np.concatenate([z, extra])))
    print(f"adding {len(extra)} intermediate levels → {len(z_new)} total")
    print(f"  intermediate z grid: {extra[0]:.0f} … {extra[-1]:.0f} km, step {STEP} km")

    # z is strictly increasing, so we can use np.interp directly on z.
    # T:    linear in z (LBLATM convention)
    # log(p): linear in z (approximate hydrostatic for thin sub-layers; exact
    #         for an isothermal scale-height region)
    # VMR:  linear in z (≈ linear in log(p) for thin sub-layers, since within
    #       each parent layer log(p) is itself ≈ linear in z; matches what
    #       LBLATM does when given AA flag in the JCHAR string)
    T_new    = np.interp(z_new, z, T)
    logp     = np.log(p)
    logp_new = np.interp(z_new, z, logp)
    p_new    = np.exp(logp_new)

    vmrs_new = {}
    for k, v in vmrs.items():
        vmrs_new[k] = np.interp(z_new, z, v)

    # Write CSV — same column order/format as the 50-level file
    headers = ["p_hPa", "T_K", "z_km",
               "vmr_H2O", "vmr_CO2", "vmr_O3", "vmr_N2O", "vmr_CH4", "vmr_CO"]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(len(z_new)):
            row = [
                f"{p_new[i]:.6e}",
                f"{T_new[i]:.1f}",
                f"{z_new[i]:.1f}",
                f"{vmrs_new['vmr_H2O'][i]:.6e}",
                f"{vmrs_new['vmr_CO2'][i]:.6e}",
                f"{vmrs_new['vmr_O3'][i]:.6e}",
                f"{vmrs_new['vmr_N2O'][i]:.6e}",
                f"{vmrs_new['vmr_CH4'][i]:.6e}",
                f"{vmrs_new['vmr_CO'][i]:.6e}",
            ]
            w.writerow(row)
    print(f"wrote {OUT} ({len(z_new)} levels)")

    # Sanity: print a few new levels around the stratopause/mesopause
    for ztest in [50.0, 60.0, 70.0, 80.0, 90.0, 95.0]:
        i = int(np.argmin(np.abs(z_new - ztest)))
        print(f"  z={z_new[i]:5.1f}  p={p_new[i]:.3e}  T={T_new[i]:.2f}  "
              f"vmr_CO2={vmrs_new['vmr_CO2'][i]:.3e}")


if __name__ == "__main__":
    main()
