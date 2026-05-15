"""
ARTS 2.6 line-by-line validation against RadiativeTransfer.jl.

Runs a nadir clear-sky thermal emission calculation using the same HITRAN
lines and US Standard atmosphere as the Julia forward model, outputs BT on
the IASI 0.25 cm-1 grid, and writes arts_bt_iasi.csv for comparison.

Run with the arts_env interpreter:
  /opt/homebrew/Caskroom/miniconda/base/envs/arts_env/bin/python scripts/arts_validation.py
"""

import os, csv, time, copy
import numpy as np
import pyarts
import pyarts.workspace

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_FILE = os.path.join(DATA_DIR, "arts_bt_iasi.csv")

C_CM = 29979245800.0   # speed of light, cm/s

def wn2hz(wn):
    return np.asarray(wn) * C_CM

def hz2wn(hz):
    return np.asarray(hz) / C_CM

# ── Spectral grids ────────────────────────────────────────────────────────────
NU_MIN, NU_MAX = 645.0, 2760.0
DNU_HI  = 0.005   # high-res LBL grid spacing (cm-1)
DNU_OUT = 0.25    # IASI output grid spacing (cm-1)
APPLY_ILS = False  # set True to convolve with IASI ILS before resampling

nu_hi   = np.arange(NU_MIN, NU_MAX + DNU_HI * 0.5, DNU_HI)
nu_iasi = np.arange(NU_MIN, NU_MAX + DNU_OUT * 0.5, DNU_OUT)

# ── Load atmosphere ───────────────────────────────────────────────────────────
print("Loading US Standard atmosphere...")
p_hPa, T_K = [], []
vmrs = {sp: [] for sp in ["H2O", "CO2", "O3", "N2O", "CH4", "CO"]}
keys = ["vmr_H2O", "vmr_CO2", "vmr_O3", "vmr_N2O", "vmr_CH4", "vmr_CO"]
sps  = ["H2O",    "CO2",    "O3",    "N2O",    "CH4",    "CO"]

with open(os.path.join(DATA_DIR, "us_standard_atm.csv")) as f:
    for row in csv.DictReader(f):
        p_hPa.append(float(row["p_hPa"]))
        T_K.append(float(row["T_K"]))
        for k, sp in zip(keys, sps):
            vmrs[sp].append(float(row[k]))

# ARTS convention: p_grid strictly decreasing (surface=index 0, TOA=last)
# Our CSV is already surface-first, so no reversal needed.
p_Pa = np.array(p_hPa) * 100.0   # hPa -> Pa
T    = np.array(T_K)
for sp in sps:
    vmrs[sp] = np.array(vmrs[sp])

n_lev = len(p_Pa)
T_sfc = T[0]    # surface = index 0
print(f"  {n_lev} levels, T_sfc = {T_sfc:.2f} K")

# Altitude via hypsometric formula, surface z=0, building upward
# z[i-1] = z[i] + (R*T_avg/g) * ln(p[i]/p[i-1])   (p[i] < p[i-1])
R_air, g = 287.058, 9.80665
z = np.zeros(n_lev)
for i in range(1, n_lev):
    T_avg = 0.5 * (T[i - 1] + T[i])
    z[i]  = z[i - 1] + (R_air * T_avg / g) * np.log(p_Pa[i - 1] / p_Pa[i])

# ── ARTS workspace ────────────────────────────────────────────────────────────
print("\nInitializing ARTS workspace...")
ws = pyarts.workspace.Workspace()
ws.verbosity = 0

ws.stokes_dim    = 1
ws.atmosphere_dim = 1
ws.f_grid        = wn2hz(nu_hi)

# Species — pure LBL, no continuum (set True to add MT-CKD 3.50 H2O continuum)
APPLY_CONTINUUM = False
SPECIES         = ["H2O", "CO2", "O3", "N2O", "CH4", "CO"]   # for VMR indexing
SPECIES_ARTS    = (
    ["H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350", "CO2", "O3", "N2O", "CH4", "CO"]
    if APPLY_CONTINUUM else
    ["H2O", "CO2", "O3", "N2O", "CH4", "CO"]
)
ws.abs_speciesSet(species=SPECIES_ARTS)

# ── Read HITRAN files and accumulate abs_lines ────────────────────────────────
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

# 25 cm-1 cutoff to match Julia forward model
ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.0)))
ws.abs_lines_per_speciesCreateFromLines()

# ── Propagation matrix (on-the-fly LBL, Voigt profile) ───────────────────────
ws.propmat_clearsky_agendaAuto()
ws.lbl_checkedCalc()

# ── Atmospheric fields ────────────────────────────────────────────────────────
print("Setting up atmosphere...")
ws.p_grid   = p_Pa
ws.lat_grid = np.array([])   # must be empty for atmosphere_dim=1
ws.lon_grid = np.array([])   # must be empty for atmosphere_dim=1

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
ws.z_surface        = np.array([[z[0]]])    # surface altitude = index 0
ws.lat_true         = np.array([0.0])
ws.lon_true         = np.array([0.0])

ws.jacobian_do = 0
ws.jacobianOff()      # initializes jacobian_quantities (needed by cloudboxOff)
ws.cloudbox_on = 0
ws.cloudboxOff()      # sets cloudbox_limits, pnd_field, etc.
ws.cloudbox_checked = 1
ws.atmfields_checked = 1
ws.atmgeom_checkedCalc()

# ── Surface ───────────────────────────────────────────────────────────────────

ws.iy_unit = "PlanckBT"
ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
ws.iy_space_agendaSet(option="CosmicBackground")

# ── Geometry / sensor ─────────────────────────────────────────────────────────
ws.water_p_eq_agendaSet()   # needed by iyEmissionStandard for refraction/humidity
ws.iy_main_agendaSet(option="Emission")
ws.rt_integration_option = "second order"   # match Julia's linear-in-τ source function
ws.ppath_agendaSet(option="FollowSensorLosPath")
ws.ppath_step_agendaSet(option="GeometricPath")
ws.ppath_lmax      = -1
ws.ppath_lraytrace = 1e4

# Sensor at 800 km altitude, nadir view
sensor_z = z[0] + 800e3   # above TOA
ws.sensor_pos = np.array([[sensor_z]])   # 1D: only altitude
ws.sensor_los = np.array([[180.0]])      # 1D: only zenith angle (180=nadir)
ws.sensorOff()            # sets sensor_response and sensor_checked=1
ws.sensor_checkedCalc()   # explicit check to be safe

ws.iy_aux_vars = []
ws.nlte_field  = pyarts.arts.EnergyLevelMap()

# ── Run ───────────────────────────────────────────────────────────────────────
print("\nRunning ARTS LBL RT...")
t0 = time.time()
ws.yCalc()
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f} s")

BT_hi = np.array(ws.y.value)
print(f"  BT range: {BT_hi.min():.1f} – {BT_hi.max():.1f} K  (n={len(BT_hi)})")

# ── Apply IASI ILS (Gaussian-tapered sinc, matching Julia's ils_kernel) ──────
# ILS must be applied to radiances, not BT.
C1 = 1.191042953e-5   # mW/(m²·sr·cm⁻⁴)   — HITRAN convention, matches Julia
C2 = 1.4387769        # cm·K

def planck_wn(nu, T):
    return C1 * nu**3 / np.expm1(C2 * nu / T)

def inv_planck_wn(nu, R):
    return C2 * nu / np.log1p(C1 * nu**3 / np.maximum(R, 1e-30))

def make_ils_kernel(dnu, opd_max=2.0, fwhm_gauss=0.5, n_pts=2048):
    sigma = fwhm_gauss / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    opd = np.linspace(0.0, opd_max, n_pts)
    dL  = opd_max / (n_pts - 1)
    A   = np.exp(-2 * np.pi**2 * sigma**2 * opd**2)
    n_half = int(np.ceil(16.0 / dnu))
    delta_nu = np.arange(-n_half, n_half + 1) * dnu
    cos_mat = np.cos(2 * np.pi * np.outer(delta_nu, opd))
    ils = 2.0 * (cos_mat @ A) * dL
    ils /= (ils.sum() * dnu)
    return delta_nu, ils

if APPLY_ILS:
    print("Applying IASI ILS...")
    from scipy.signal import fftconvolve
    R_hi        = planck_wn(nu_hi, BT_hi)
    _, ils_kern = make_ils_kernel(DNU_HI)
    R_apod      = fftconvolve(R_hi, ils_kern, mode="same") * DNU_HI
    BT_apod     = inv_planck_wn(nu_hi, R_apod)
else:
    BT_apod = BT_hi

# ── Resample to IASI 0.25 cm-1 grid ──────────────────────────────────────────
print("Resampling to IASI grid...")
BT_iasi = np.interp(nu_iasi, nu_hi, BT_apod)

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w") as f:
    f.write("nu_cm1,BT_K\n")
    for nu, bt in zip(nu_iasi, BT_iasi):
        f.write(f"{nu:.4f},{bt:.6f}\n")
print(f"\nSaved {len(nu_iasi)} channels → {OUT_FILE}")
