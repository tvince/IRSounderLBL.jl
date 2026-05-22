"""
ARTS 2.6 LBL validation with H2O MT-CKD continuum + HITRAN CIA
(CO2-CO2, N2-N2, O2-O2). Matches the Julia continuum stack post-commit
dee74da. Writes to data/arts_bt_iasi_cont.csv (overwrites the H2O-only
run; backup at data/arts_bt_iasi_cont_pre_cia.csv).

Run with:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/arts_validation_cont_cia.py
"""

import os, csv, time, copy
import numpy as np
import pyarts
import pyarts.workspace

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(DATA_DIR, "arts_bt_iasi_cont.csv")

C_CM = 29979245800.0

def wn2hz(wn):
    return np.asarray(wn) * C_CM

def hz2wn(hz):
    return np.asarray(hz) / C_CM

NU_MIN, NU_MAX = 645.0, 2760.0
DNU_HI  = 0.005
DNU_OUT = 0.25
APPLY_ILS = False

nu_hi   = np.arange(NU_MIN, NU_MAX + DNU_HI * 0.5, DNU_HI)
nu_iasi = np.arange(NU_MIN, NU_MAX + DNU_OUT * 0.5, DNU_OUT)

print("Loading AFGL US Standard atmosphere (50 levels, 0-120 km)...")
p_hPa, T_K, z_km = [], [], []
vmrs = {sp: [] for sp in ["H2O", "CO2", "O3", "N2O", "CH4", "CO"]}
keys = ["vmr_H2O", "vmr_CO2", "vmr_O3", "vmr_N2O", "vmr_CH4", "vmr_CO"]
sps  = ["H2O",    "CO2",    "O3",    "N2O",    "CH4",    "CO"]

with open(os.path.join(DATA_DIR, "afgl_us_standard_50lev.csv")) as f:
    for row in csv.DictReader(f):
        p_hPa.append(float(row["p_hPa"]))
        T_K.append(float(row["T_K"]))
        z_km.append(float(row["z_km"]))
        for k, sp in zip(keys, sps):
            vmrs[sp].append(float(row[k]))

p_Pa = np.array(p_hPa) * 100.0
T    = np.array(T_K)
z    = np.array(z_km) * 1000.0
for sp in sps:
    vmrs[sp] = np.array(vmrs[sp])

# AFGL US Standard 50-lev reaches 360 K in the thermosphere, but ARTS's CIA
# T-interpolation refuses to extrapolate more than ~15 K past the catalog T
# range (CO2-CO2 caps at 297 K for the 2510-2850 cm⁻¹ band). Cap profile T
# at 295 K; affected levels (z>100 km) have n² ≈ 0 so CIA contribution is
# negligible and BT is unaffected.
# (Originally we capped T at 295 K to dodge CIA T-extrapolation, but using
# T_extrapolfac=1000 in propmat_clearsky_agendaAuto is cleaner and keeps
# line absorption unchanged. T cap removed; profile passed verbatim.)

# N2 and O2 are not in the AFGL CSV — set from standard dry-air mole fractions
# scaled by (1 - vmr_H2O), matching the Julia wiring convention.
vmrs["N2"] = 0.78084 * (1.0 - vmrs["H2O"])
vmrs["O2"] = 0.20946 * (1.0 - vmrs["H2O"])

n_lev = len(p_Pa)
T_sfc = T[0]
print(f"  {n_lev} levels, T_sfc = {T_sfc:.2f} K, z_top = {z_km[-1]:.0f} km")

print("\nInitializing ARTS workspace...")
ws = pyarts.workspace.Workspace()
ws.verbosity = 0

ws.stokes_dim    = 1
ws.atmosphere_dim = 1
ws.f_grid        = wn2hz(nu_hi)

APPLY_CONTINUUM = True
SPECIES         = ["H2O", "CO2", "O3", "N2O", "CH4", "CO", "N2", "O2"]
SPECIES_ARTS    = ["H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
                   "CO2, CO2-CIA-CO2",
                   "O3", "N2O", "CH4", "CO",
                   "N2, N2-CIA-N2",
                   "O2, O2-CIA-O2"]
ws.abs_speciesSet(species=SPECIES_ARTS)
print(f"  Continuum: ON")
print(f"    H2O: SelfContCKDMT350 + ForeignContCKDMT350")
print(f"    CIA: CO2-CO2, N2-N2 (2 bands), O2-O2 (HITRAN tables)")

# Load HITRAN CIA tables individually — abs_cia_dataReadFromCIA's catalog
# layout was not cooperating, so we use CIARecordReadFromFile per species.
cia_dir = os.path.join(DATA_DIR, "cia")
ws.abs_cia_data = pyarts.arts.ArrayOfCIARecord()
for tag, fname in [("CO2-CIA-CO2", "CO2-CO2_2024.cia"),
                   ("N2-CIA-N2",   "N2-N2_2021.cia"),
                   ("O2-CIA-O2",   "O2-O2_2024.cia")]:
    rec = pyarts.arts.CIARecord()
    ws.CIARecordReadFromFile(cia_record=rec, species_tag=tag,
                             filename=os.path.join(cia_dir, fname))
    ws.abs_cia_dataAddCIARecord(cia_record=rec)
    print(f"    {tag}: {len(rec.data)} blocks")

HITRAN_FILES = [
    ("co2_645_2760.par",      True),
    ("co2_645_2760_iso2.par", True),
    ("co2_645_2760_iso3.par", True),
    ("h2o_645_2760.par",      True),
    ("h2o_645_2760_iso2.par", True),
    ("h2o_645_2760_iso3.par", True),
    ("o3_980_1090.par",       True),
    ("o3_980_1090_iso2.par",  True),
    ("o3_980_1090_iso3.par",  True),
    ("n2o_1200_2310.par",     True),
    ("n2o_1200_2310_iso2.par",True),
    ("n2o_1200_2310_iso3.par",True),
    ("ch4_1200_1800.par",     True),
    ("ch4_1200_1800_iso2.par",True),
    ("ch4_1200_1800_iso3.par",True),
    ("co_2000_2280.par",      True),
    ("co_2000_2280_iso2.par", True),
    ("co_2000_2280_iso3.par", True),
]

print("Reading HITRAN catalog...")
all_bands = []
for fname, required in HITRAN_FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(fpath):
        if required:
            print(f"  WARNING: missing {fname}")
        continue
    print(f"  {fname}")
    ws.ReadHITRAN(filename=fpath,
                  normalization_option="SFS",
                  hitran_type="Online")
    all_bands += copy.deepcopy(list(ws.abs_lines.value))

ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(all_bands)
print(f"  Total band groups: {len(all_bands)}")

ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.0)))
ws.abs_lines_per_speciesCreateFromLines()

ws.propmat_clearsky_agendaAuto(T_extrapolfac=1000.0)
ws.lbl_checkedCalc()

print("Setting up atmosphere...")
ws.p_grid   = p_Pa
ws.lat_grid = np.array([])
ws.lon_grid = np.array([])

ws.t_field   = T.reshape(n_lev, 1, 1)
ws.z_field   = z.reshape(n_lev, 1, 1)
ws.vmr_field = np.array([vmrs[sp] for sp in SPECIES]).reshape(len(SPECIES), n_lev, 1, 1)

ws.wind_u_field = np.zeros((n_lev, 1, 1))
ws.wind_v_field = np.zeros((n_lev, 1, 1))
ws.wind_w_field = np.zeros((n_lev, 1, 1))
ws.mag_u_field  = np.zeros((n_lev, 1, 1))
ws.mag_v_field  = np.zeros((n_lev, 1, 1))
ws.mag_w_field  = np.zeros((n_lev, 1, 1))

ws.refellipsoid     = pyarts.arts.Vector([6378137.0, 0.0])
ws.z_surface        = np.array([[z[0]]])
ws.lat_true         = np.array([0.0])
ws.lon_true         = np.array([0.0])

ws.jacobian_do = 0
ws.jacobianOff()
ws.cloudbox_on = 0
ws.cloudboxOff()
ws.cloudbox_checked = 1
ws.atmfields_checked = 1
ws.atmgeom_checkedCalc()

ws.iy_unit = "PlanckBT"
ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
ws.iy_space_agendaSet(option="CosmicBackground")

ws.water_p_eq_agendaSet()
ws.iy_main_agendaSet(option="Emission")
ws.ppath_agendaSet(option="FollowSensorLosPath")
ws.ppath_step_agendaSet(option="GeometricPath")
ws.ppath_lmax      = -1
ws.ppath_lraytrace = 1e4

sensor_z = z[0] + 800e3
ws.sensor_pos = np.array([[sensor_z]])
ws.sensor_los = np.array([[180.0]])
ws.sensorOff()
ws.sensor_checkedCalc()

ws.iy_aux_vars = []
ws.nlte_field  = pyarts.arts.EnergyLevelMap()

print("\nRunning ARTS LBL RT (with continuum)...")
t0 = time.time()
ws.yCalc()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f} s")

BT_hi = np.array(ws.y.value)
print(f"  BT range: {BT_hi.min():.1f} – {BT_hi.max():.1f} K  (n={len(BT_hi)})")

C1 = 1.191042953e-5
C2 = 1.4387769

def planck_wn(nu, T):
    return C1 * nu**3 / np.expm1(C2 * nu / T)

def inv_planck_wn(nu, R):
    return C2 * nu / np.log1p(C1 * nu**3 / np.maximum(R, 1e-30))

BT_iasi = np.interp(nu_iasi, nu_hi, BT_hi)

with open(OUT_FILE, "w") as f:
    f.write("nu_cm1,BT_K\n")
    for nu, bt in zip(nu_iasi, BT_iasi):
        f.write(f"{nu:.4f},{bt:.6f}\n")
print(f"\nSaved {len(nu_iasi)} channels → {OUT_FILE}")
