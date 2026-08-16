"""
Extract MT-CKD 4.3 H2O continuum coefficients from the AER netCDF file
and write them to a simple CSV for use by Julia's continuum.jl.

Source: data/mt_ckd_h2o/absco-ref_wv-mt-ckd.nc (AER MT_CKD_H2O v4.3)

Output columns:
  nu_cm1              wavenumber (cm⁻¹)
  Cs_cm2_molec_cm1    self-continuum reference coefficient at T=296 K (cm²/molec/cm⁻¹)
  Cf_cm2_molec_cm1    foreign-continuum reference coefficient at T=296 K (cm²/molec/cm⁻¹)
  self_texp           self-continuum temperature exponent (dimensionless)

Usage formula (from mt_ckd_h2o_module.f90):
  rho_rat  = (p_hPa / 1013.0) * (296.0 / T)
  rad(ν,T) = ν * tanh(1.4387769*ν / (2*T))    [cm⁻¹]
  k_self    = Cs * (296/T)^texp * vmr * rho_rat * rad * n_h2o   [cm⁻¹]
  k_foreign = Cf * (1 - vmr)   * rho_rat       * rad * n_h2o   [cm⁻¹]
  where n_h2o = vmr * p_hPa*100 / (kB*T) * 1e-6  [molec/cm³]

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/extract_ckd43_csv.py
"""

import os, csv
import numpy as np
import netCDF4 as nc

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_FILE  = os.path.join(ROOT, "data", "mt_ckd_h2o", "absco-ref_wv-mt-ckd.nc")
OUT_FILE = os.path.join(ROOT, "data", "mt_ckd_h2o", "mt_ckd43_h2o_coeffs.csv")

ds     = nc.Dataset(IN_FILE)
nu     = np.array(ds.variables["wavenumbers"][:])
Cs     = np.array(ds.variables["self_absco_ref"][:])
Cf     = np.array(ds.variables["for_absco_ref"][:])
texp   = np.array(ds.variables["self_texp"][:])
p_ref  = float(ds.variables["ref_press"][:])   # mbar
T_ref  = float(ds.variables["ref_temp"][:])    # K
ds.close()

print(f"Source: {IN_FILE}")
print(f"  ref_press = {p_ref} mbar,  ref_temp = {T_ref} K")
print(f"  {len(nu)} grid points from {nu.min():.0f} to {nu.max():.0f} cm⁻¹  (Δν = {nu[1]-nu[0]:.0f} cm⁻¹)")

# Keep only ν ≥ 0 (negative wavenumbers are unphysical padding in the table)
mask = nu >= 0.0
nu, Cs, Cf, texp = nu[mask], Cs[mask], Cf[mask], texp[mask]
print(f"  Kept {len(nu)} points with ν ≥ 0")

# Quick sanity check
print(f"\nSelf-continuum Cs at reference points:")
for wn in [645, 700, 1000, 1200, 1600, 2000, 2500, 2760]:
    i = np.argmin(np.abs(nu - wn))
    print(f"  {nu[i]:.0f} cm⁻¹:  Cs={Cs[i]:.4e}  Cf={Cf[i]:.4e}  texp={texp[i]:.3f}")

with open(OUT_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nu_cm1", "Cs_cm2_molec_cm1", "Cf_cm2_molec_cm1", "self_texp"])
    for row in zip(nu, Cs, Cf, texp):
        w.writerow([f"{row[0]:.1f}", f"{row[1]:.6e}", f"{row[2]:.6e}", f"{row[3]:.6f}"])

print(f"\nSaved {len(nu)} rows → {OUT_FILE}")
