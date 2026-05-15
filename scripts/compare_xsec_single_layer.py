"""
Single-layer H2O cross-section comparison: ARTS (SFS vs None) vs Julia.

Focus: H2O 1380–1800 cm⁻¹ (6 µm band) where the largest BT bias is observed.

Run order:
  julia --project scripts/compare_xsec_julia.jl
  python scripts/compare_xsec_single_layer.py
"""

import os, csv, time, copy
import numpy as np
import pyarts
import pyarts.workspace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

T_K     = 255.0
P_HPA   = 500.0
VMR_H2O = 3.0e-3
P_PA    = P_HPA * 100.0
C_CM    = 29979245800.0

NU_MIN, NU_MAX = 1380.0, 1800.0
DNU = 0.005
nu  = np.arange(NU_MIN, NU_MAX + DNU * 0.5, DNU)
f_hz = nu * C_CM

k_B        = 1.380649e-23
n_total_m3 = P_PA / (k_B * T_K)
n_h2o_m3   = VMR_H2O * n_total_m3


def run_arts_h2o(norm_option):
    """H2O cross-section [cm²/molec] via ARTS propmat_clearskyInit + AddLines."""
    ws = pyarts.workspace.Workspace()
    ws.verbosity = 0

    ws.stokes_dim = 1
    ws.f_grid     = f_hz

    ws.abs_speciesSet(species=["H2O"])

    all_bands = []
    for fname in ["h2o_645_2760.par", "h2o_645_2760_iso2.par",
                  "h2o_645_2760_iso3.par"]:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ws.ReadHITRAN(filename=fpath,
                      normalization_option=norm_option,
                      hitran_type="Online")
        all_bands += copy.deepcopy(list(ws.abs_lines.value))
    ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(all_bands)
    ws.abs_linesCutoff(option="ByLine", value=float(25.0 * C_CM))
    ws.abs_lines_per_speciesCreateFromLines()

    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agendaAuto()
    ws.propmat_clearsky_agenda_checkedCalc()

    # Set single-point RTP conditions
    ws.rtp_pressure    = float(P_PA)
    ws.rtp_temperature = float(T_K)
    ws.rtp_vmr         = np.array([VMR_H2O])
    ws.rtp_nlte        = pyarts.arts.EnergyLevelMap()

    ws.jacobian_do = 0
    ws.jacobianOff()
    ws.propmat_clearskyInit()
    ws.propmat_clearskyAddLines()

    # propmat data shape: (1, 1, n_freq, 1) for stokes_dim=1
    pm = ws.propmat_clearsky.value
    alpha_m = np.array(pm.data)[0, 0, :, 0]  # H2O LBL extinction [1/m]

    return alpha_m / n_h2o_m3 * 1e4   # cm²/molec


print("Computing ARTS SFS...")
t0 = time.time()
sigma_sfs = run_arts_h2o("SFS")
print(f"  {time.time()-t0:.1f} s")

print("Computing ARTS None...")
t0 = time.time()
sigma_none = run_arts_h2o("None")
print(f"  {time.time()-t0:.1f} s")

# ── Load Julia cross-section ──────────────────────────────────────────────────
julia_file = os.path.join(DATA_DIR, "julia_xsec_h2o_layer.csv")
nu_j, sigma_j = [], []
with open(julia_file) as f:
    for row in csv.DictReader(f):
        nu_j.append(float(row["nu_cm1"]))
        sigma_j.append(float(row["sigma_cm2"]))
nu_j    = np.array(nu_j)
sigma_j = np.array(sigma_j)
sigma_ji = np.interp(nu, nu_j, sigma_j)

# ── Stats ─────────────────────────────────────────────────────────────────────
mask = sigma_ji > 1e-30

def pct_diff(a, b):
    r = (a - b) / np.maximum(b, 1e-60) * 100.0
    return r[mask].mean(), np.abs(r[mask]).max()

m1, x1 = pct_diff(sigma_sfs, sigma_none)
m2, x2 = pct_diff(sigma_sfs, sigma_ji)
m3, x3 = pct_diff(sigma_none, sigma_ji)

print(f"\n{'Comparison':40s}  {'Mean%':>8s}  {'Max|%|':>8s}")
print(f"{'ARTS SFS vs ARTS None':40s}  {m1:+8.3f}  {x1:8.3f}")
print(f"{'ARTS SFS vs Julia':40s}  {m2:+8.3f}  {x2:8.3f}")
print(f"{'ARTS None vs Julia':40s}  {m3:+8.3f}  {x3:8.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
fig.suptitle(f"H2O cross-section: T={T_K} K, p={P_HPA} hPa, vmr={VMR_H2O:.3e}")

ax = axes[0]
ax.semilogy(nu, sigma_sfs, lw=0.5, color="steelblue", label="ARTS SFS")
ax.semilogy(nu, sigma_none, lw=0.5, color="tomato",    label="ARTS None", alpha=0.8)
ax.semilogy(nu, sigma_ji,  lw=0.5, color="green",     label="Julia",     alpha=0.8)
ax.set_ylabel("σ (cm²/molec)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(NU_MIN, NU_MAX)

ax = axes[1]
ratio_sn = (sigma_sfs / np.maximum(sigma_none, 1e-60) - 1) * 100
ax.plot(nu, ratio_sn, lw=0.4, color="purple")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("(SFS/None − 1) %")
ax.set_title("SFS vs None normalization")
ax.set_xlim(NU_MIN, NU_MAX)
ax.set_ylim(-15, 15)
ax.grid(True, alpha=0.3)

ax = axes[2]
ratio_nj = (sigma_none / np.maximum(sigma_ji, 1e-60) - 1) * 100
ax.plot(nu, ratio_nj, lw=0.4, color="darkgreen")
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("(ARTS-None/Julia − 1) %")
ax.set_title("Residual: ARTS-None vs Julia")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_xlim(NU_MIN, NU_MAX)
ax.set_ylim(-15, 15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(DATA_DIR, "xsec_comparison.png")
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
