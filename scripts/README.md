# scripts/

Runnable examples, plus the scripts that generated the data tables shipped with
the package and the ones cited from the source as evidence for its defaults.

## examples/

A guided sequence. Each runs standalone, but they build on each other — start at
01 if you are new to the package. All four need only the default 15 µm line-list
set, which `download_data(:linelists)` installs (a free HITRAN API key is
required for line lists; the CIA tables `download_data()` fetches need no key).

| Script | What it shows | Time |
|---|---|---|
| `01_forward_spectrum.jl` | Profile + line lists + instrument → brightness-temperature spectrum, and how to read the CO₂ ν₂ band | ~40 s |
| `02_weighting_functions.jl` | The analytic Jacobian ∂BT/∂T, where each channel gets its signal, and what the band can resolve vertically | ~2 min |
| `03_retrieval_synthetic.jl` | A closed-loop optimal-estimation retrieval of a temperature profile: χ², degrees of freedom, averaging kernels, error budget | ~4 min |
| `04_instrument_comparison.jl` | One atmosphere through IASI, IASI-NG, CrIS and MTG-IRS — the same radiative transfer, four instrument models | ~2 min |

Timings are for an M1 Pro with 6 threads; run with `-t auto`. Examples 02–04 use
a coarser internal grid than the production default to stay quick, which each
script notes where it matters.

```
julia --project=. -t auto scripts/examples/01_forward_spectrum.jl
```

If a script exits telling you data is missing, run `data_status()` for a
checklist of what is present, what downloads automatically, and what has to be
fetched by hand.

## provenance/

How the data tables bundled with the package were derived from their upstream
sources. You do not need to run these — the outputs are committed — but they
document exactly what was extracted and how, and the source cites them where the
tables are loaded.

| Script | Produces |
|---|---|
| `extract_ckd_co2.py` | MT-CKD CO₂ continuum coefficients, pulled from LBLRTM's `contnm.f90` `BLOCK DATA BFCO2` |
| `extract_ckd43_csv.py` | MT-CKD 4.3 H₂O self/foreign continuum coefficients, from AER's `absco-ref_wv-mt-ckd.nc` |
| `extract_tips.py` | TIPS-2024 partition sums → `src/HITRAN/tips2024_data.jl` |
| `build_afgl_atmospheres.py` | The six AFGL 50-level reference atmospheres in `data/` |

## validation/

Convergence studies and sweeps that justify specific defaults in the source. The
code cites these by name where a constant is chosen, so the reasoning behind a
number is one file away.

| Script | Establishes |
|---|---|
| `grid_convergence_iasing.jl` | Why `internal_dnu` defaults to 0.001 cm⁻¹ |
| `convergence_sweep_43um.jl` | Why the CIM source function is the default over Toon |
| `validate_line_rejection_bt.jl` | The `dptmn` line-rejection threshold |
| `sweep_band_cutoff.jl` | The optional `min_band_strength` line-mixing band cutoff |
| `julia_bt_export.jl` | The full-spectrum benchmark quoted in the README |

These need line lists beyond the default 15 µm set, and some need reference
output from LBLRTM or ARTS.

## Looking for the validation campaign?

The ~165 scripts behind the ARTS and LBLRTM comparisons, the CO₂ line-mixing
investigation, the ozone and 715 cm⁻¹ residual studies, and the plotting for all
of it live on the **`validation-scripts`** branch:

```
git checkout validation-scripts
```

They are kept out of the released package because most require external tools
(pyarts, a built LBLRTM), reference data that cannot be redistributed, or
plotting dependencies the package deliberately does not carry. They remain the
evidence trail for the accuracy figures quoted in the README.
