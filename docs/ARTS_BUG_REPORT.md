# `abs_lines_per_speciesAdaptHitranLineMixing` produces wrong-sign / inflated Y for low-J P-branch CO₂-626 hot-band lines near 665 cm⁻¹

**Reporter**: R. Anthony Vincent &lt;10545693+tvince@users.noreply.github.com&gt, (using Claude Opus);
**Component**: HITRAN line mixing (`lm_hitran_2017`, `src/linemixing_hitran.cc`)
**Affected method**: `abs_lines_per_speciesAdaptHitranLineMixing`
**ARTS / pyarts version**: pyarts 2.6.18 (conda-forge build, Python 3.14, macOS arm64)

**Attachments** (drag-and-drop these onto the issue after posting):
- `data/arts_bug_iso1_p_branch_diff.csv` — 32 iso-1 P-branch lines in 664.5–665.5 cm⁻¹ with Julia Y / ARTS Y / Δ / ratio (sorted by |ΔY/atm|).
- `data/diff_arts_julia_Y_near_665.csv` — full 427-line three-way diff (all isos, all branches in 664.5–665.5 cm⁻¹).

## Summary

Running `abs_lines_per_speciesAdaptHitranLineMixing` (order=1) on the HITRAN 2020 CO₂ line-mixing package produces first-order Rosenkranz Y values that disagree with the Lamouroux *et al.* reference `LM_calc_15um.for` (shipped with the HITRAN package) by:

- **wrong sign** for several strong low-J P-branch lines of CO₂-626 hot bands;
- **5–25× larger magnitude** than the reference.

In our use case (downwelling IASI brightness-temperature simulation, AFGL US Standard, 645–800 cm⁻¹), this produces a **~33 K BT spike at ν = 665.00 cm⁻¹** that does not appear in the Lamouroux reference (or our independent Julia reimplementation that matches the reference to ~1.5%).

## Reproducer

```python
import os, copy, time
import numpy as np
import pyarts, pyarts.workspace

DATA = "/path/to/Line-mixing_HITRAN2020"           # HITRAN package
LM   = os.path.join(DATA, "data_new")
HITRAN_PAR = "/path/to/co2_645_2760.par"           # HITRAN par for iso 1

C_CM = 29979245800.0
def wn2hz(x): return np.asarray(x) * C_CM
T = 288.20
P = 1.0132 * 101325.0
VMR_air = pyarts.arts.Vector([0., 0., 1.])         # [CO2-self, H2O, Bath]

ws = pyarts.workspace.Workspace(); ws.verbosity = 0
ws.stokes_dim = 1; ws.atmosphere_dim = 1
ws.f_grid = wn2hz([645., 800.])
ws.Wigner3Init(); ws.Wigner6Init()
ws.abs_speciesSet(species=["CO2"])
ws.ReadHITRAN(filename=HITRAN_PAR, normalization_option="SFS",
              hitran_type="Online",
              fmin=float(wn2hz(645.)), fmax=float(wn2hz(800.)))
ws.abs_lines = pyarts.arts.ArrayOfAbsorptionLines(list(ws.abs_lines.value))
ws.abs_linesCutoff(option="ByLine", value=float(wn2hz(25.)))
ws.abs_lines_per_speciesCreateFromLines()

ws.abs_hitran_relmat_dataReadHitranRelmatDataAndLines(
    basedir=LM, linemixinglimit=-1,
    fmin=float(wn2hz(645.)), fmax=float(wn2hz(800.)),
    stot=0, mode="VP_Y")

ws.abs_lines_per_speciesAdaptHitranLineMixing(
    t_grid   = pyarts.arts.Vector(np.linspace(200., 320., 13)),
    pressure = P,
    order    = 1,
)

# Dump Y at T, pure-air VMR for a few iso-1 P-branch lines we expect to be small
TARGETS = [(665.413951, 3, -1), (664.626547, 4, -1),
           (664.989626, 4, -1), (664.558605, 5, -1)]

for band in ws.abs_lines_per_species.value[0]:
    if band.quantumidentity.isotopologue.name != "CO2-626":
        continue
    for il, line in enumerate(band.lines):
        f = line.F0 / C_CM
        for nu_t, *_ in TARGETS:
            if abs(f - nu_t) < 1e-4:
                out = band.LineShapeOutput(il, T, P, VMR_air)
                print(f"{band.quantumidentity}  ν={f:.5f}  Y(dim)={out.Y:+.5f}  "
                      f"Y/atm={out.Y/1.0132:+.5f}")
```

## Observed vs reference

All Y values evaluated at T=288.20 K, pure-air, P=1.0132 atm. Reference is Lamouroux `LM_calc_15um.for` patched to the same (T, P) and recompiled with `gfortran -O2 -std=legacy -ffixed-line-length-none`. Julia (independent reimplementation) is shown for additional cross-check.

| Band             | Branch | ν (cm⁻¹)  | Lamouroux Y/atm | Julia Y/atm | **ARTS Y/atm** |
|------------------|--------|-----------|-----------------|-------------|----------------|
| S10220101101     | P(2,3) | 665.4140  | **+0.02184**    | +0.02155    | **−0.0878**    |
| S10220101101     | P(3,4) | 664.6265  | **+0.00901**    | +0.00889    | **−0.0038**    |
| S10330102201     | P(3,4) | 664.9896  | **+0.00511**    | +0.00504    | **−0.1125**    |
| S10440103301     | P(4,5) | 664.5586  | **−0.00343**    | −0.00338    | **−0.1274**    |

Julia matches Lamouroux to ~1.5% on every line; ARTS is sign-flipped on three of four and 5–25× too large in magnitude.

The fundamental ν₂ band S10110100001 (also CO₂-626) is **correct** in ARTS for its P-branch (no large divergence). The bug is specific to hot bands like S10220101101 / S10330102201 / S10440103301 where odd J is allowed.

## Why this matters

The wrong Y values propagate through the first-order Rosenkranz line-mixing correction (`AbsY = Re[Σ Sᵢ·F(σ,σᵢ)·(1 + i·p·Yᵢ)]`). For a strong line at ν=665.413 with S=3.6×10⁻²² and ARTS Y/atm = −0.088 (vs reference +0.022), the dispersive contribution at that frequency is off by ~12% × |line strength|, which is enough to add tens of K of BT error at the Q-branch wing.

End-to-end impact in our setup: at ν = 665.00 cm⁻¹, `ws.yCalc()` with mode="VP_Y" + the HITRAN relmat data gives BT ≈ 253 K; the same atmosphere through the Lamouroux Fortran or our Julia implementation gives BT ≈ 219 K. **A 33 K spike.** Setting `mode="VP_W"` does *not* close the gap (ARTS VP_W = ARTS VP_Y at this point), so the issue is not a first-order vs full-matrix question — the first-order Y itself is wrong.

## Where to look in the source

Based on the error message I got while exploring (`hitran_lm_eigenvalue_adaptation` in `src/linemixing_hitran.cc:2230`), the candidate locations are:

- `lm_hitran_2017::calcw` (W-matrix construction); specifically the J pairing / e-f symmetry block handling for asymmetric hot-band (li, lf) symmetry pairs.
- `lm_hitran_2017::hitran_lm_eigenvalue_adaptation` (the polynomial fit consuming Y).

I have already ruled out:

- **Polynomial fit instability**: re-running the adapt with `t_grid = np.linspace(270, 310, 9)` instead of `(200, 320, 13)` gives the same problematic Y values to 5+ significant figures, so the fit itself is not the culprit.
- **Sign convention for the W(T) = W₀·(T₀/T)^(−B₀) temperature scaling**: T₀/T = 1.028 at T=288, so any sign flip on the exponent would only move W by a few percent, not 5–25×.

For comparison, the Lamouroux Fortran also has an explicit safety clip `if (abs(YT(iLine)) .gt. 0.08) YT(iLine)=0.0` (LM_calc_15um.for:773–781), but it only fires on a handful of low-J Q-branch lines (Q2/Q4/Q6 of the fundamental); the disputed P-branch lines have small Y in Fortran's own calc and pass through unzeroed.

## Environment

- pyarts 2.6.18 (conda-forge)
- Python 3.14, macOS 14 (Darwin 25.4) arm64
- HITRAN line-mixing package "Line-mixing_HITRAN2020" downloaded from HITRANonline
- HITRAN line lists `co2_645_2760.par`, `co2_645_2760_iso2.par`, `co2_645_2760_iso3.par` from HITRANonline (vers. 2020)
