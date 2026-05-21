"""
Probe: what does ARTS abs_lines_per_speciesAdaptHitranLineMixing actually put
into each line's lineshape?  We need to know the parameter type (T1, T4, …)
and which broadening species index carries Y for CO2-in-air.

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/probe_arts_Y_structure.py
"""

import os, time, copy
import numpy as np
import pyarts
import pyarts.workspace

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LM_DIR   = os.path.join(DATA_DIR, "Line-mixing_HITRAN2020", "data_new")

C_CM = 29979245800.0
def wn2hz(wn): return np.asarray(wn) * C_CM

NU_MIN, NU_MAX = 645.0, 800.0      # full 15 µm window — adaptation may need full band

T_K     = 288.20
P_atm   = 1.0132
P_Pa    = P_atm * 101325.0

ws = pyarts.workspace.Workspace()
ws.verbosity = 0
ws.stokes_dim     = 1
ws.atmosphere_dim = 1
ws.f_grid         = wn2hz(np.array([NU_MIN, NU_MAX]))
ws.Wigner3Init()
ws.Wigner6Init()

ws.abs_speciesSet(species=["CO2"])

print("Reading CO2 HITRAN catalog (iso 1+2+3) for 664–666 cm⁻¹ probe …")
all_bands = []
for fname in ("co2_645_2760.par", "co2_645_2760_iso2.par", "co2_645_2760_iso3.par"):
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"  skip missing {fname}"); continue
    ws.ReadHITRAN(filename=fpath, normalization_option="SFS", hitran_type="Online",
                  fmin=float(wn2hz(NU_MIN)), fmax=float(wn2hz(NU_MAX)))
    all_bands += copy.deepcopy(list(ws.abs_lines.value))

ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(all_bands)
print(f"  Total band groups: {len(all_bands)}")
ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.0)))
ws.abs_lines_per_speciesCreateFromLines()

print(f"\nLoading HITRAN relmat data from {LM_DIR} …")
t0 = time.time()
ws.abs_hitran_relmat_dataReadHitranRelmatDataAndLines(
    basedir        = LM_DIR,
    linemixinglimit= -1,
    fmin           = float(wn2hz(NU_MIN)),
    fmax           = float(wn2hz(NU_MAX)),
    stot           = 0,
    mode           = "VP_Y",
)
print(f"  done in {time.time()-t0:.1f} s")

print(f"\nAdapting line catalog (order=1) at T={T_K} K, P={P_Pa:.3f} Pa …")
t0 = time.time()
ws.abs_lines_per_speciesAdaptHitranLineMixing(
    t_grid   = pyarts.arts.Vector(np.linspace(200.0, 320.0, 13)),
    pressure = float(P_Pa),
    order    = 1,
)
print(f"  done in {time.time()-t0:.1f} s")

# Inspect one CO2 band's first line
co2_lps = list(ws.abs_lines_per_species.value[0])
print(f"\n#bands in CO2 species: {len(co2_lps)}")

# Find a band that has at least one line in 664–666 cm⁻¹
target_bands = []
for ib, band in enumerate(co2_lps):
    for line in band.lines:
        f_cm = line.F0 / C_CM
        if 664.0 <= f_cm <= 666.0:
            target_bands.append(ib)
            break
print(f"Bands with a line in 664–666 cm⁻¹: {len(target_bands)} (first 3 will be inspected)")

for ib in target_bands[:3]:
    band = co2_lps[ib]
    print(f"\n── Band {ib}: {band.quantumidentity}")
    print(f"   lineshapetype={band.lineshapetype}, "
          f"selfbroadening={band.selfbroadening}, "
          f"broadeningspecies={[str(s) for s in band.broadeningspecies]}")
    # Find the line in the target range
    for il, line in enumerate(band.lines):
        f_cm = line.F0 / C_CM
        if not (664.0 <= f_cm <= 666.0):
            continue
        print(f"  Line {il}: F0={f_cm:.5f} cm⁻¹  I0={line.I0:.3e}  E0={line.E0:.3e}")
        print(f"  lineshape: {type(line.lineshape).__name__}, "
              f"data len = {len(line.lineshape.data)}")
        for ispc, model in enumerate(line.lineshape.data):
            print(f"    species {ispc}:")
            for name in ("G0", "D0", "Y", "G", "DV"):
                p = getattr(model, name)
                print(f"      {name:3s}  type={p.type}  X0={p.X0: .6e}  X1={p.X1: .4f}  "
                      f"X2={p.X2: .4e}  X3={p.X3: .4e}")
        break
