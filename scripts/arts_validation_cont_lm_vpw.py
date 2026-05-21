"""
ARTS 2.6 LBL validation with H2O MT-CKD 3.50 continuum + CO2 line mixing.
**VP_W variant** — tests whether ARTS's mode="VP_Y" silently behaves like
full-matrix VP_W in the Q-branch, which is what the Julia-vs-ARTS data
suggests.  If this run matches the VP_Y output (data/arts_bt_co2_15um_lm.csv),
the modes converge.  If they differ, ARTS's VP_Y is genuinely first-order
and the Q-branch spike comes from somewhere else.

Output: data/arts_bt_co2_15um_lm_vpw.csv
"""

import os, csv, time, copy
import numpy as np
import pyarts
import pyarts.workspace

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(DATA_DIR, "arts_bt_co2_15um_lm_vpw.csv")
LM_DIR   = os.path.join(DATA_DIR, "Line-mixing_HITRAN2020", "data_new")

C_CM = 29979245800.0

def wn2hz(wn):
    return np.asarray(wn) * C_CM

NU_MIN, NU_MAX = 645.0, 800.0
DNU_HI  = 0.005
DNU_OUT = 0.25
APPLY_ILS = False

nu_hi   = np.arange(NU_MIN, NU_MAX + DNU_HI * 0.5, DNU_HI)
nu_iasi = np.arange(NU_MIN, NU_MAX + DNU_OUT * 0.5, DNU_OUT)

# ── Load atmosphere ───────────────────────────────────────────────────────────
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
n_lev = len(p_Pa)
print(f"  {n_lev} levels, T_sfc = {T[0]:.2f} K, z_top = {z_km[-1]:.0f} km")

# ── ARTS workspace ────────────────────────────────────────────────────────────
print("\nInitializing ARTS workspace...")
ws = pyarts.workspace.Workspace()
ws.verbosity = 0

ws.stokes_dim     = 1
ws.atmosphere_dim = 1
ws.f_grid         = wn2hz(nu_hi)

# Wigner 3j/6j tables required for HITRAN line mixing relaxation matrix.
# Default largest_wigner_symbol_parameter=250 covers our J_max≈90 bands.
ws.Wigner3Init()
ws.Wigner6Init()

SPECIES      = ["H2O", "CO2", "O3", "N2O", "CH4", "CO"]
SPECIES_ARTS = ["H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
                "CO2", "O3", "N2O", "CH4", "CO"]
ws.abs_speciesSet(species=SPECIES_ARTS)
print("  Continuum: ON  (H2O-SelfContCKDMT350 + H2O-ForeignContCKDMT350)")
print("  Line mixing: ON  (HITRAN 2020, VP_W, 645–800 cm⁻¹)")

# ── Read HITRAN catalog ───────────────────────────────────────────────────────
HITRAN_FILES = [
    ("co2_645_2760.par",      True),
    ("co2_645_2760_iso2.par", True),
    ("co2_645_2760_iso3.par", True),
    ("h2o_645_2760.par",      True),
    ("h2o_645_2760_iso2.par", True),
    ("h2o_645_2760_iso3.par", True),
    # O3, N2O, CH4, CO have no significant lines in 645–800 cm⁻¹
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
    ws.ReadHITRAN(filename=fpath, normalization_option="SFS", hitran_type="Online")
    all_bands += copy.deepcopy(list(ws.abs_lines.value))

ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(all_bands)
print(f"  Total band groups: {len(all_bands)}")

# Apply 25 cm⁻¹ cutoff to non-CO2 lines; CO2 will be replaced by line mixing
ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.0)))
ws.abs_lines_per_speciesCreateFromLines()

# ── Load HITRAN line mixing data (replaces CO2 lines) ────────────────────────
print(f"\nLoading HITRAN 2020 line mixing data from:\n  {LM_DIR}")
t_lm = time.time()
ws.abs_hitran_relmat_dataReadHitranRelmatDataAndLines(
    basedir        = LM_DIR,
    linemixinglimit= -1,                  # apply line mixing at all pressures
    fmin           = float(wn2hz(NU_MIN)),
    fmax           = float(wn2hz(NU_MAX)),
    stot           = 0,
    mode           = "VP_W",
)
print(f"  Done in {time.time() - t_lm:.1f} s")

# ── Propagation matrix agenda ─────────────────────────────────────────────────
# propmat_clearsky_agendaAuto detects the HITRAN relmat data and includes
# propmat_clearskyAddHitranLineMixingLines in the agenda.
ws.propmat_clearsky_agendaAuto()
ws.lbl_checkedCalc()

# ── Atmospheric fields ────────────────────────────────────────────────────────
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
ws.cloudbox_checked  = 1
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

# ── Run ───────────────────────────────────────────────────────────────────────
print("\nRunning ARTS LBL RT (continuum + line mixing)...")
t0 = time.time()
ws.yCalc()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f} s")

BT_hi = np.array(ws.y.value)
print(f"  BT range: {BT_hi.min():.1f} – {BT_hi.max():.1f} K  (n={len(BT_hi)})")

BT_iasi = np.interp(nu_iasi, nu_hi, BT_hi)

with open(OUT_FILE, "w") as f:
    f.write("nu_cm1,BT_K\n")
    for nu, bt in zip(nu_iasi, BT_iasi):
        f.write(f"{nu:.4f},{bt:.6f}\n")
print(f"\nSaved {len(nu_iasi)} channels → {OUT_FILE}")
