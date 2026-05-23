"""
Probe ARTS CIA absorption coefficient per species at fixed (p, T) over a
ν-grid. Uses `CIARecord.compute_abs` — the same primitive
`propmat_clearskyAddCIA` calls internally — to bypass the RT solve entirely.

Output: data/arts_cia_sigma_dump.csv with columns
  nu_cm1, k_co2, k_n2, k_o2     (absorption coefficient in 1/cm)

Also dumps the per-block (ν-range × T-list) structure for each species so we
can see what ARTS actually has loaded.

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python \
    scripts/dump_arts_cia_sigma.py
"""

import os
import numpy as np
import pyarts

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(DATA_DIR, "arts_cia_sigma_dump.csv")

C_CM = 29979245800.0
KB   = 1.380649e-23

def wn2hz(wn):
    return np.asarray(wn, dtype=float) * C_CM

# Probe point (matches AFGL US Standard surface)
P_HPA   = 1013.0
T_K     = 288.0
VMR_H2O = 0.00626
VMR_CO2 = 365e-6
VMR_N2  = 0.78084 * (1.0 - VMR_H2O)
VMR_O2  = 0.20946 * (1.0 - VMR_H2O)

NU = np.arange(2000.0, 2700.0 + 0.001, 0.5)
f_grid = pyarts.arts.Vector(wn2hz(NU))

n_total = P_HPA * 100.0 / (KB * T_K) * 1e-6  # /cm³

print(f"Probe (p, T) = ({P_HPA} hPa, {T_K} K)")
print(f"  VMR: CO2={VMR_CO2:.3e}  N2={VMR_N2:.5f}  O2={VMR_O2:.5f}")
print(f"  ν-grid: {NU[0]} → {NU[-1]} cm⁻¹  (n={len(NU)})")

ws = pyarts.workspace.Workspace()
ws.verbosity = 0

def load(tag, fname):
    rec = pyarts.arts.CIARecord()
    ws.CIARecordReadFromFile(cia_record=rec, species_tag=tag,
                             filename=os.path.join(DATA_DIR, "cia", fname))
    return rec

records = {
    "co2": load("CO2-CIA-CO2", "CO2-CO2_2024.cia"),
    "n2":  load("N2-CIA-N2",   "N2-N2_2021.cia"),
    "o2":  load("O2-CIA-O2",   "O2-O2_2024.cia"),
}

print("\nBlock structure (ν-range × n_T) per species:")
for sp, rec in records.items():
    print(f"  {sp}: {len(rec.data)} ν-range blocks")
    for i, blk in enumerate(rec.data):
        f_hz = np.asarray(blk.get_grid(0))
        T    = np.asarray(blk.get_grid(1))
        print(f"    [{i}] ν=[{f_hz[0]/C_CM:9.3f},{f_hz[-1]/C_CM:9.3f}]  n_T={len(T):2d}  T_range=[{T.min():.1f},{T.max():.1f}]")

# Self pair: X0 = X1 = VMR_self for homogeneous pair
vmr_for = {"co2": VMR_CO2, "n2": VMR_N2, "o2": VMR_O2}

P_PA = P_HPA * 100.0

k_total = {}
k_per_block = {}  # to isolate which block contributes to the overlap region

for sp in ("co2", "n2", "o2"):
    rec = records[sp]
    x = vmr_for[sp]
    k_arr = np.asarray(rec.compute_abs(T_K, P_PA, x, x, f_grid,
                                        1000.0,  # T_extrapolfac
                                        1))      # robust
    # /m → /cm
    k_total[sp] = k_arr * 1e-2

# Now isolate per-block contributions for N2 — split into single-block
# CIARecords and call compute_abs on each
print("\nPer-block contribution probe (N2 at ν=2331, T=288):")
i_peak = int(np.argmin(np.abs(NU - 2331.0)))
nu_peak = NU[i_peak]
f_peak = pyarts.arts.Vector(wn2hz([nu_peak]))

rec_n2 = records["n2"]
for i in range(len(rec_n2.data)):
    blk = rec_n2.data[i]
    f_hz = np.asarray(blk.get_grid(0))
    nu_lo, nu_hi = f_hz[0]/C_CM, f_hz[-1]/C_CM
    if not (nu_lo <= nu_peak <= nu_hi):
        continue
    # Build a single-block record using the (data, sp0, sp1) constructor
    sp0, sp1 = rec_n2.specs
    rec_one = pyarts.arts.CIARecord(
        pyarts.arts.ArrayOfGriddedField2([blk]), sp0, sp1)
    k_one = np.asarray(rec_one.compute_abs(T_K, P_PA, VMR_N2, VMR_N2,
                                            f_peak, 1000.0, 1))[0]
    T_arr = np.asarray(blk.get_grid(1))
    print(f"  block {i}: ν=[{nu_lo:.1f},{nu_hi:.1f}]  T_range=[{T_arr.min():.1f},{T_arr.max():.1f}]"
          f"  → k = {k_one*1e-2:.4e} /cm")

# Sum-of-blocks vs full-record at peak
k_full_peak = np.asarray(rec_n2.compute_abs(T_K, P_PA, VMR_N2, VMR_N2,
                                             f_peak, 1000.0, 1))[0] * 1e-2
print(f"  FULL record k = {k_full_peak:.4e} /cm")

with open(OUT_FILE, "w") as f:
    f.write("nu_cm1,k_co2,k_n2,k_o2\n")
    for i, nu in enumerate(NU):
        f.write(f"{nu:.4f},{k_total['co2'][i]:.6e},{k_total['n2'][i]:.6e},{k_total['o2'][i]:.6e}\n")
print(f"\nSaved {len(NU)} channels → {OUT_FILE}")

print(f"\nAt ν = {NU[i_peak]} cm⁻¹ (N2 fundamental peak):")
print(f"  k_co2 = {k_total['co2'][i_peak]:.3e} /cm")
print(f"  k_n2  = {k_total['n2'][i_peak]:.3e} /cm")
print(f"  k_o2  = {k_total['o2'][i_peak]:.3e} /cm")
