# `CIARecord::Extract` sums overlapping ν-range entries in `mdata` instead of treating them as alternative samples of σ(ν, T)

**Reporter**: R. Anthony Vincent &lt;10545693+tvince@users.noreply.github.com&gt; (using Claude Opus)
**Component**: Collision-induced absorption (`src/core/absorption/cia.cc`,
`CIARecord::Extract`; pyarts exposes this as `CIARecord.compute_abs` and
it is the primitive called by `propmat_clearskyAddCIA` /
`spectral_propmatAddCIA`)
**ARTS / pyarts version**: pyarts 2.6.18 (conda-forge, Python 3.14, macOS arm64)

**Attached** (CSV dumps and overlay plot):
- `arts_cia_sigma_dump.csv` — ARTS k_N2 per ν on 2000–2700 cm⁻¹, T=288 K.
- `julia_cia_sigma_dump.csv` — independent reimplementation, same grid.
- `cia_sigma_diff.png` — 4-panel overlay + ratio plot.

## Summary

When a `CIARecord` contains multiple `GriddedField2` entries in `mdata`
whose ν-grids overlap, `CIARecord::Extract` **sums** their σ
contributions at the query (ν, T) — each per-block contribution is
T-interpolated/extrapolated independently inside `cia_interpolation`,
then accumulated via `res += result`. For the HITRAN N₂–N₂ catalog
(`N2-N2_2021.cia`), this double-counts the 4-µm fundamental band: the
file contains **two independent laboratory measurements** of the same
physical σ(ν, T) with complementary T coverage —

- ν=1850.004–3000.094 cm⁻¹, T=[300.9, 323.6, 343.5, 355.3, 362.5] K ("0-8 atm" comment in .cia header — hot sub-band)
- ν=1999.900–2697.900 cm⁻¹, T=[228.2, 233.7, 243.2, 253.2, 272.1] K ("0-10 atm" comment — cold sub-band)

These are two *samples* of one physical quantity (the N₂–N₂ CIA cross
section in the fundamental band region), not two additive contributions.
Atmospheric T_sfc ≈ 288 K falls in the 272.1–300.9 K gap between them.
With `T_extrapolfac=1000`, ARTS additionally extrapolates each block
outside its own T-grid before summing, compounding the error.

End-to-end impact: in a matched-config IASI BT validation
(`abs_speciesSet`+`CO2-CIA-CO2`+`N2-CIA-N2`+`O2-CIA-O2`, AFGL US
Standard, matched H₂O continuum on both sides), this changes the
ARTS-vs-reference RMS from **0.11 K → 0.81 K** with a 5.45 K peak at
ν=2405 cm⁻¹ — squarely on top of the N₂ fundamental.

## Reproducer

```python
import os, numpy as np, pyarts

C_CM = 29979245800.0
def wn2hz(x): return np.asarray(x, dtype=float) * C_CM
T_K, P_PA, VMR_N2 = 288.0, 101325.0, 0.7759

ws = pyarts.workspace.Workspace(); ws.verbosity = 0
rec = pyarts.arts.CIARecord()
ws.CIARecordReadFromFile(cia_record=rec, species_tag="N2-CIA-N2",
                         filename="N2-N2_2021.cia")  # HITRAN download

# Full record (both overlapping blocks present)
f = pyarts.arts.Vector(wn2hz([2331.0]))
k_full = float(rec.compute_abs(T_K, P_PA, VMR_N2, VMR_N2, f, 1000.0, 1)[0])

# Same query against each overlapping ν-range entry individually
sp0, sp1 = rec.specs
k_per = {}
for i, blk in enumerate(rec.data):
    f_hz = np.asarray(blk.get_grid(0))
    if not (f_hz[0]/C_CM <= 2331.0 <= f_hz[-1]/C_CM):
        continue
    rec_one = pyarts.arts.CIARecord(
        pyarts.arts.ArrayOfGriddedField2([blk]), sp0, sp1)
    k_per[i] = float(rec_one.compute_abs(T_K, P_PA, VMR_N2, VMR_N2, f, 1000.0, 1)[0])

print(f"per-block:   {k_per}")
print(f"sum of per-block:  {sum(k_per.values()):.4e}  /m")
print(f"full record:       {k_full:.4e}  /m")
```

Output:

```
per-block:        {3: 1.1298e-04, 4: 2.6586e-04}        # /m
sum of per-block: 3.7884e-04 /m
full record:      3.7884e-04 /m         ← identical, confirming summation
```

Block 3 carries T-extrapolation 12.9 K *below* its own T-grid (300.9 →
288 K) and block 4 carries T-extrapolation 15.9 K *above* its own
T-grid (272.1 → 288 K). Each separately extrapolated, then summed.

## Observed vs reference

Independent reimplementation builds a unified σ(ν, T) from the same
file by iterating all per-T tables, finding the bracketing pair
(T_lo=272.1 K from block 4, T_hi=300.9 K from block 3) and linearly
interpolating between exact-table σ values — no extrapolation needed
because the union of blocks brackets atmospheric T.

| ν (cm⁻¹) | k_N2 ARTS [/cm] | k_N2 reference [/cm] | ARTS / ref |
|----------|-----------------|----------------------|------------|
| 2100.0   | 1.489e-08       | 1.235e-08            | 1.21       |
| 2200.0   | 1.952e-07       | 1.926e-07            | 1.01       |
| **2331.0** (peak) | **3.788e-06** | **1.108e-06**  | **3.42**   |
| 2400.0   | 2.152e-06       | 7.620e-07            | 2.82       |
| 2500.0   | 1.998e-07       | 1.893e-07            | 1.06       |
| 2600.0   | 2.236e-07       | 2.288e-08            | 9.77       |

Mean ARTS/reference ratio inside the overlap window 1999.9–2697.9 cm⁻¹:
**10.8×**. Outside the overlap (where one block has σ≈0 in its tabulated
range so the sum coincides with the active block): **0.86×**.

The ARTS/reference ratio peaks at the *band core* — the 2240–2510 cm⁻¹
region — where both source datasets carry substantial non-zero σ, so
summation roughly doubles the value and per-block T-extrapolation lifts
it further.

## Why summation is wrong for this dataset

The two ν-overlapping `GriddedField2` entries in N2-N2_2021.cia represent
**alternative laboratory measurements of the same physical σ(ν, T)**.
HITRAN's CIA format (Karman et al. 2019, doi:10.1016/j.icarus.2019.02.034)
documents these sub-blocks as complementary T-coverage of one CIA pair,
not as additive sub-processes. Two tests make this concrete:

1. Construct a hypothetical CIA file where *two laboratories independently
   published σ at the same (ν, T)*. The correct combined value is one of
   them (or an average); summing gives 2× — clearly nonphysical.
2. In the band wings (ν≈2200 cm⁻¹ in our test), one block's tabulated σ
   is essentially zero outside its measurement's effective range, so the
   sum coincides with the active block alone and ARTS happens to match
   reference. This is exactly the behavior expected from a buggy
   summation that produces correct answers only when datasets don't
   substantively overlap.

## Suggested fix direction

`compute_abs` should treat the collection of `GriddedField2` entries
within one `CIARecord` (sharing the same species pair) as a single
σ(ν, T) field with non-uniform (ν, T) coverage:

- For each query ν, identify the subset of blocks whose ν-grid contains ν.
- Across that subset, locate the T_lo / T_hi bracketing the query T using
  *exact T-grid values from any block* (not per-block extrapolation).
- Linearly interpolate σ in T between those bracketing values; clamp at
  the extremes of the full T set.

Per-block T-extrapolation should only be used when the *union* of blocks
fails to bracket the query T — which for atmospheric T and the N₂–N₂
4-µm region never happens (the union covers 228.2–362.5 K).

## Where it lives in the source (atmtools/arts@main)

The accumulating loop is in `src/core/absorption/cia.cc` inside the
vector-form `CIARecord::Extract` (declared in `src/core/absorption/cia.h`
around line 95). The relevant excerpt (≈ lines 286–292):

```cpp
Vector result(res.size());
for (auto& this_cia : mdata) {
  cia_interpolation(
      result, f_grid, temperature, this_cia, T_extrapolfac, robust);
  res += result;                       // ← sums every overlapping block
}
```

Per-block T-interpolation/extrapolation happens inside
`cia_interpolation` (`cia.cc:~84`), which forwards `T_extrapolfac` to
`lagrange_interp::make_lags` and is therefore agnostic to whether other
blocks in `mdata` cover the same (ν, T).

Not requesting any specific implementation here — the ARTS maintainers
will likely see a better path once the diagnosis is confirmed. For
discussion only, two directions that come to mind:

1. **Two-pass within `Extract`**: first walk `mdata` to collect, per
   ν-bin, the set of (T_block, σ_block) values available from blocks
   whose ν-grid contains ν. Then interpolate σ in T using those (T, σ)
   values directly — without per-block T-extrapolation. Only invoke
   `cia_interpolation` with extrapolation when the union of bracketing
   T's still fails to bracket the query T.

2. **Defer to upstream loaders**: refuse to read overlapping
   `GriddedField2` entries into one `CIARecord` and force the user to
   pick one, but this throws away valid lab data and shifts the burden
   onto every user.

## What does NOT fix it

- `T_extrapolfac=0` (no extrapolation) — fails because neither block's
  T-grid covers atmospheric T_sfc ≈ 288 K, raising a runtime error.
- Pre-pruning the input file (deleting block 3 or block 4 before loading)
  — works as a user-side workaround but masks the algorithmic bug and
  drops valid lab data.

## Environment

- pyarts 2.6.18 (conda-forge)
- Python 3.14, macOS 14 (Darwin 25.4) arm64
- HITRAN file: `N2-N2_2021.cia` downloaded from hitran.org/cia/
  (6 ν-range entries, 58 total T-blocks; SHA-256
  `f0525f75c667802e8dbbfd7054a0a7e07863d8a4ffdb449fe80666bc658c849d`)

## Cross-reference

Filed alongside issue
[#1130](https://github.com/atmtools/arts/issues/1130) (HITRAN line-mixing
Y values) from the same validation campaign.
