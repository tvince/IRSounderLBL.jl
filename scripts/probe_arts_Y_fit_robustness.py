"""
Sanity check: does the ARTS-adapted Y depend on the t_grid used for fitting?

If Y at T=288.20 is identical across two very different t_grids, the polynomial
fit is robust and the Julia-vs-ARTS divergence is in ARTS's underlying
relaxation-matrix / Y calculation, not in the polynomial fit.

We check two grids:
  A: 200–320 K, 13 points (Δ=10 K)         [what dump_arts_Y_near_665.py uses]
  B: 270–310 K, 9 points (Δ=5 K)           [denser, narrower band around 288]
"""
import os, copy, time
import numpy as np
import pyarts, pyarts.workspace

DATA = "/Users/tonyvincent/RadiativeTransfer/data"
LM   = os.path.join(DATA, "Line-mixing_HITRAN2020", "data_new")
C_CM = 29979245800.0
def wn2hz(x): return np.asarray(x) * C_CM
T = 288.20
P = 1.0132 * 101325.0
VMR = pyarts.arts.Vector([0., 0., 1.])

# Lines of interest (the worst iso-1 P-branch divergences)
TARGETS = [
    (1, 665.413951), (1, 664.989626), (1, 664.626547),
    (1, 664.558605), (1, 665.272936), (1, 664.977315),
]
ISO_NUM = {"CO2-626":1,"CO2-636":2,"CO2-628":3,"CO2-627":4,
           "CO2-638":5,"CO2-637":6,"CO2-828":7,"CO2-728":8,
           "CO2-727":9,"CO2-838":10}

results = {}
for tag, tgrid in [("A_200-320_13pt", np.linspace(200, 320, 13)),
                   ("B_270-310_9pt",  np.linspace(270, 310, 9))]:
    print(f"\n=== Fit grid {tag} ===")
    ws = pyarts.workspace.Workspace(); ws.verbosity = 0
    ws.stokes_dim=1; ws.atmosphere_dim=1
    ws.f_grid = wn2hz([645., 800.])
    ws.Wigner3Init(); ws.Wigner6Init()
    ws.abs_speciesSet(species=["CO2"])
    all_bands=[]
    for fn in ("co2_645_2760.par","co2_645_2760_iso2.par","co2_645_2760_iso3.par"):
        ws.ReadHITRAN(filename=os.path.join(DATA,fn),normalization_option="SFS",
                      hitran_type="Online",fmin=float(wn2hz(645.)),fmax=float(wn2hz(800.)))
        all_bands += copy.deepcopy(list(ws.abs_lines.value))
    ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(all_bands)
    ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.)))
    ws.abs_lines_per_speciesCreateFromLines()
    ws.abs_hitran_relmat_dataReadHitranRelmatDataAndLines(
        basedir=LM, linemixinglimit=-1,
        fmin=float(wn2hz(645.)), fmax=float(wn2hz(800.)),
        stot=0, mode="VP_Y")
    t0=time.time()
    ws.abs_lines_per_speciesAdaptHitranLineMixing(
        t_grid=pyarts.arts.Vector(tgrid), pressure=P, order=1)
    print(f"  adapt done in {time.time()-t0:.1f} s")

    results[tag] = {}
    for band in ws.abs_lines_per_species.value[0]:
        iso = ISO_NUM.get(band.quantumidentity.isotopologue.name, -1)
        for il, line in enumerate(band.lines):
            f = line.F0 / C_CM
            for ti, tnu in TARGETS:
                if iso == ti and abs(f - tnu) < 1e-4:
                    out = band.LineShapeOutput(il, T, P, VMR)
                    results[tag][(ti, tnu)] = out.Y / 1.0132  # per atm
                    break

print(f"\n{'iso ν':>15s}  {'Grid A':>10s}  {'Grid B':>10s}  {'A−B':>10s}  {'A/B':>8s}")
for ti, tnu in TARGETS:
    ya = results["A_200-320_13pt"].get((ti, tnu), float("nan"))
    yb = results["B_270-310_9pt"].get((ti, tnu),  float("nan"))
    diff = ya - yb
    ratio = ya / yb if yb != 0 else float("nan")
    print(f"  iso{ti} {tnu:.4f}  {ya:+10.5f}  {yb:+10.5f}  {diff:+10.5f}  {ratio:8.4f}")
