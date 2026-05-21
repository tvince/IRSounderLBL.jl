"""
Dump ARTS per-line Y values for CO2 lines in 664.5–665.5 cm⁻¹ at the AFGL US
Standard surface state (T=288.20 K, P=1.0132 atm).

Procedure
---------
1. Build an ARTS workspace identical to scripts/arts_validation_cont_lm.py up
   to the line-mixing setup (CO2 catalog iso 1+2+3, 645–800 cm⁻¹, HITRAN
   relmat data, mode="VP_Y").
2. Run `abs_lines_per_speciesAdaptHitranLineMixing` (order=1) to convert the
   ARTS-internal relmat data into per-line Y polynomial coefficients stored on
   each AbsorptionSingleLine.lineshape.
3. For every line whose centre is in [664.5, 665.5] cm⁻¹, evaluate the line's
   Y in air-only mode (`VMR=[CO2=0, H2O=0, Bath=1]`) and write
       iso_name, iso_num, ν_cm, I0, G0_air_cm⁻¹, Y_dim, Y_per_atm
   to data/arts_Y_near_665.csv.

Diff against scripts/dump_julia_Y_near_665.jl to test the hypothesis (from
[[project-arts-lm-spike]]) that ARTS produces ~30× larger LM correction than
Julia at the 665 cm⁻¹ Q-branch because the Y values themselves diverge between
the two pipelines.

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/dump_arts_Y_near_665.py
"""

import os, time, copy
import numpy as np
import pyarts
import pyarts.workspace

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LM_DIR   = os.path.join(DATA_DIR, "Line-mixing_HITRAN2020", "data_new")
OUT_CSV  = os.path.join(DATA_DIR, "arts_Y_near_665.csv")

C_CM = 29979245800.0
def wn2hz(wn): return np.asarray(wn) * C_CM

# ── Match dump_julia_Y_near_665.jl ────────────────────────────────────────────
NU_LO, NU_HI = 664.5, 665.5
T_K   = 288.20
P_atm = 1.0132
P_Pa  = P_atm * 101325.0

# HITRAN CO2 isotopologue label → AFGL isotopologue index
ISO_NUM = {
    "CO2-626": 1, "CO2-636": 2, "CO2-628": 3, "CO2-627": 4,
    "CO2-638": 5, "CO2-637": 6, "CO2-828": 7, "CO2-728": 8,
    "CO2-727": 9, "CO2-838": 10,
}

# ── ARTS workspace setup (mirrors arts_validation_cont_lm.py) ────────────────
ws = pyarts.workspace.Workspace()
ws.verbosity = 0
ws.stokes_dim     = 1
ws.atmosphere_dim = 1
ws.f_grid         = wn2hz(np.array([645.0, 800.0]))
ws.Wigner3Init()
ws.Wigner6Init()
ws.abs_speciesSet(species=["CO2"])

print("Reading CO2 HITRAN catalog 645–800 cm⁻¹ (iso 1+2+3) …")
all_bands = []
for fname in ("co2_645_2760.par", "co2_645_2760_iso2.par", "co2_645_2760_iso3.par"):
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(fpath):
        print(f"  skip missing {fname}"); continue
    print(f"  {fname}")
    ws.ReadHITRAN(
        filename=fpath, normalization_option="SFS", hitran_type="Online",
        fmin=float(wn2hz(645.0)), fmax=float(wn2hz(800.0)),
    )
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
    fmin           = float(wn2hz(645.0)),
    fmax           = float(wn2hz(800.0)),
    stot           = 0,
    mode           = "VP_Y",
)
print(f"  done in {time.time()-t0:.1f} s")

print(f"\nAdapting lines (order=1, T grid 200–320 K @ 10 K, P={P_Pa:.0f} Pa) …")
t0 = time.time()
ws.abs_lines_per_speciesAdaptHitranLineMixing(
    t_grid   = pyarts.arts.Vector(np.linspace(200.0, 320.0, 13)),
    pressure = float(P_Pa),
    order    = 1,
)
print(f"  done in {time.time()-t0:.1f} s")

# ── Evaluate Y per line at T,P and pure-air VMR ──────────────────────────────
co2_lps = list(ws.abs_lines_per_species.value[0])
VMR_air = pyarts.arts.Vector([0.0, 0.0, 1.0])  # [CO2-self, H2O, Bath/air]

rows = []
n_total = 0
for band in co2_lps:
    iso_name = band.quantumidentity.isotopologue.name
    iso_num  = ISO_NUM.get(iso_name, -1)
    for il, line in enumerate(band.lines):
        f_cm = line.F0 / C_CM
        if not (NU_LO <= f_cm <= NU_HI):
            continue
        n_total += 1
        out = band.LineShapeOutput(il, T_K, P_Pa, VMR_air)
        rows.append((
            iso_name, iso_num, f_cm, line.I0,
            out.G0 / C_CM,           # G0 in cm⁻¹ (pure air)
            out.Y,                   # dimensionless = Y_per_atm * P_atm
            out.Y / P_atm,           # Y per atm (matches Julia Y_band units)
        ))

print(f"\nFound {n_total} CO2 lines in [{NU_LO}, {NU_HI}] cm⁻¹")

# Sort by ν0 ascending then by iso for readability
rows.sort(key=lambda r: (r[2], r[1]))

with open(OUT_CSV, "w") as f:
    f.write("iso_name,iso_num,nu_cm,I0,G0_air_cm,Y_dim_at_TP,Y_per_atm\n")
    for r in rows:
        f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6e},{r[4]:.5e},{r[5]:+.6e},{r[6]:+.6e}\n")
print(f"Wrote {len(rows)} rows → {OUT_CSV}")

# ── Top |Y_per_atm| summary ─────────────────────────────────────────────────
rows.sort(key=lambda r: -abs(r[6]))
print(f"\nTop 15 |Y_per_atm| lines (T={T_K} K, P={P_atm} atm, pure-air):")
print(f"{'iso':>8s}  {'ν (cm⁻¹)':>10s}  {'I0':>10s}  {'Y/atm':>9s}  {'Y·p':>9s}")
for r in rows[:15]:
    print(f"{str(r[0]):>8s}  {r[2]:>10.5f}  {r[3]:>10.2e}  "
          f"{r[6]:+9.5f}  {r[5]:+9.5f}")
