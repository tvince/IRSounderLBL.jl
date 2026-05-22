"""
Extract MT-CKD 3.50 H2O self and foreign continuum coefficients from pyarts.

Strategy:
  - Run two minimal ARTS propmat calculations at T_ref=296 K, p_ref=1013.25 hPa
    with only the H2O continuum species (no line absorption).
  - Run 1 (pure H2O): VMR=1 → only self-continuum (p_dry=0)
  - Run 2 (trace H2O in dry air): VMR→0, so foreign dominates (p_h2o→0, p_dry=p_ref)

From Run 1:
  k1 [1/m] = n_h2o * C_s * p_h2o     (p_h2o = p_ref, p_dry = 0)
  → C_s [m²/molec/Pa] = k1 / (n_h2o * p_h2o)

From Run 2 (with small VMR=eps):
  k2 [1/m] ≈ n_h2o * C_f * p_dry     (p_h2o ≈ 0, p_dry ≈ p_ref)
  → C_f [m²/molec/Pa] = k2 / (n_h2o * p_dry)

Convert to cm²/molec/atm to match Julia's internal units:
  C [cm²/molec/atm] = C [m²/molec/Pa] × (1e4)² × 101325

Output: data/ckd_mt350_coeffs.csv  (nu_cm1, Cs_296K, Cf_296K)

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/extract_ckd_coeffs.py
"""

import numpy as np
import csv
import os
import pyarts
import pyarts.workspace

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

C_CM = 29979245800.0   # cm/s
KB   = 1.380649e-23    # J/K
T_REF = 296.0          # K
P_REF = 101325.0       # Pa (= 1 atm)
P_REF_ATM = P_REF / 101325.0  # = 1.0

# Wavenumber grid matching our forward model
NU_MIN, NU_MAX, DNU = 645.0, 2760.0, 0.5   # 0.5 cm⁻¹ spacing for table
nu_cm1 = np.arange(NU_MIN, NU_MAX + DNU * 0.5, DNU)
f_hz   = nu_cm1 * C_CM

print(f"Grid: {len(nu_cm1)} points, {NU_MIN:.0f}–{NU_MAX:.0f} cm⁻¹, Δν={DNU} cm⁻¹")

def run_propmat(vmr_h2o):
    """Return absorption coefficient vector [1/m] at T_ref, p_ref for given VMR."""
    ws = pyarts.workspace.Workspace()
    ws.verbosity = 0
    ws.stokes_dim = 1
    ws.atmosphere_dim = 1
    ws.f_grid = f_hz

    ws.abs_speciesSet(species=["H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350"])
    ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines([])
    ws.abs_lines_per_speciesCreateFromLines()
    ws.propmat_clearsky_agendaAuto()
    ws.lbl_checkedCalc()

    # Set single-point atmospheric state directly (no geometry setup needed)
    ws.rtp_pressure    = float(P_REF)
    ws.rtp_temperature = float(T_REF)
    ws.rtp_vmr         = np.array([float(vmr_h2o)])
    ws.rtp_mag         = np.zeros(3)
    ws.rtp_los         = np.array([0.0, 0.0])
    ws.rtp_pos         = np.array([0.0, 0.0, 0.0])

    ws.propmat_clearsky_agenda_checked = 1
    ws.propmat_clearskyCalc()

    # propmat_clearsky is ArrayOfPropagationMatrix; one entry per abs_species group.
    # For stokes_dim=1 each PropagationMatrix.A() gives the absorption [1/m] per frequency.
    pm = ws.propmat_clearsky.value
    k = np.array([pm[0].A()[i] for i in range(len(f_hz))])
    return k

# Number density of H2O at T_ref, p_ref [molec/m³]
def n_molec(vmr, p_Pa, T_K):
    return vmr * p_Pa / (KB * T_K)

print("\nRun 1: pure H2O (VMR=1.0) → self-continuum...")
VMR_SELF = 1.0
k_self = run_propmat(VMR_SELF)
n_h2o  = n_molec(VMR_SELF, P_REF, T_REF)
p_h2o_Pa = VMR_SELF * P_REF   # Pa
p_h2o_atm = p_h2o_Pa / 101325.0
# k = n_h2o * C_s * p_h2o  (in SI: C_s [m²/molec/Pa], k [1/m])
# k = n_h2o * C_s_pa * p_h2o_Pa
# C_s_pa = k / (n_h2o * p_h2o_Pa)
C_s_m2_pa = k_self / (n_h2o * p_h2o_Pa)
# Convert to cm²/molec/atm:  1 m² = 1e4 cm²,  1 Pa = 1/101325 atm
# C [cm²/molec/atm] = C [m²/molec/Pa] * 1e4 * 101325
C_s_cm2_atm = C_s_m2_pa * 1e4 * 101325.0

print(f"  k range: {k_self.min():.3e} – {k_self.max():.3e} 1/m")
print(f"  C_s range: {C_s_cm2_atm.min():.3e} – {C_s_cm2_atm.max():.3e} cm²/molec/atm")
print(f"  C_s at 1000 cm⁻¹: {C_s_cm2_atm[np.argmin(np.abs(nu_cm1 - 1000))]: .3e}")
print(f"  C_s at 1600 cm⁻¹: {C_s_cm2_atm[np.argmin(np.abs(nu_cm1 - 1600))]: .3e}")

print("\nRun 2: trace H2O (VMR=1e-6) → foreign-continuum...")
VMR_FOREIGN = 1e-6
k_foreign = run_propmat(VMR_FOREIGN)
n_h2o_f   = n_molec(VMR_FOREIGN, P_REF, T_REF)
p_h2o_f   = VMR_FOREIGN * P_REF           # Pa (≈ 0)
p_dry_f   = (1.0 - VMR_FOREIGN) * P_REF   # Pa (≈ P_REF)
p_dry_f_atm = p_dry_f / 101325.0
# k ≈ n_h2o * C_f * p_dry
C_f_m2_pa = k_foreign / (n_h2o_f * p_dry_f)
C_f_cm2_atm = C_f_m2_pa * 1e4 * 101325.0

print(f"  k range: {k_foreign.min():.3e} – {k_foreign.max():.3e} 1/m")
print(f"  C_f range: {C_f_cm2_atm.min():.3e} – {C_f_cm2_atm.max():.3e} cm²/molec/atm")
print(f"  C_f at 1000 cm⁻¹: {C_f_cm2_atm[np.argmin(np.abs(nu_cm1 - 1000))]: .3e}")
print(f"  C_f at 1600 cm⁻¹: {C_f_cm2_atm[np.argmin(np.abs(nu_cm1 - 1600))]: .3e}")

# Check Cs/Cf ratio
ratio = C_s_cm2_atm / np.maximum(C_f_cm2_atm, 1e-100)
print(f"\n  Cs/Cf ratio at 1000 cm⁻¹: {ratio[np.argmin(np.abs(nu_cm1-1000))]:.1f}")
print(f"  Cs/Cf ratio at 1600 cm⁻¹: {ratio[np.argmin(np.abs(nu_cm1-1600))]:.1f}")

# Save
out = os.path.join(DATA_DIR, "ckd_mt350_coeffs.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nu_cm1", "Cs_296K_cm2_molec_atm", "Cf_296K_cm2_molec_atm"])
    for nu, cs, cf in zip(nu_cm1, C_s_cm2_atm, C_f_cm2_atm):
        w.writerow([f"{nu:.2f}", f"{cs:.6e}", f"{cf:.6e}"])
print(f"\nSaved {len(nu_cm1)} rows → {out}")
