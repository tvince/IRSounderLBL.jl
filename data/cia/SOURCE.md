# CIA data provenance

Collision-induced absorption (CIA) cross-section tables used by `co2_cia`,
`n2_cia`, and `o2_cia` in `src/CrossSections/continuum.jl`.

These files are **not tracked in git** and do **not** ship with the package:
HITRAN's terms permit use with citation but **not redistribution**. If a file is
missing the loader emits a one-time warning and returns zeros.

## Getting the data

```julia
using IRSounderLBL
download_data()      # fetches all three, verifies SHA-256
data_status()        # shows what is present and where it was found
```

`download_data()` writes into a package-owned `Scratch.jl` space by default (or
`IRSOUNDER_DATA_DIR` / the `data_dir` Preference, if set — see `set_data_dir!`).
Files are resolved lazily on first use, so a download during a live session takes
effect without restarting Julia. Alternatively, drop the files into this directory
by hand under the exact filenames below — the registry lives in
`src/Utils/datasets.jl`, which also pins the expected checksums.

## Source

HITRAN Collision-Induced Absorption database: <https://hitran.org/cia/>

Direct URLs (as used by `download_data()`): `https://hitran.org/data/CIA/main/<file>`

| File                | Collision pair | Coverage (this copy)            |
|---------------------|----------------|---------------------------------|
| `CO2-CO2_2024.cia`  | CO₂–CO₂        | 1.0–4499.6 cm⁻¹, 192–800 K, 37 T-blocks |
| `N2-N2_2021.cia`    | N₂–N₂          | 0.0–5000.0 cm⁻¹, 70–400 K, 58 T-blocks  |
| `O2-O2_2024.cia`    | O₂–O₂          | 1150–33670 cm⁻¹, 193–353 K, 46 T-blocks |

Format is standard HITRAN `.cia` ASCII: per-temperature header lines
(pair, ν_min, ν_max, n_points, T, σ_max, …) followed by `ν  σ(ν,T)` rows, with
σ in cm⁵·molec⁻². The `_2024` / `_2021` suffixes are HITRAN's release tags; grab
the current release if re-downloading and update the filenames here and in
`continuum.jl` if they change.

## Citation

The HITRAN CIA cross sections are from:

> Karman, T., Gordon, I. E., van der Avoird, A., Baranov, Y. I., Boulet, C.,
> Drouin, B. J., … Vigasin, A. A. (2019). *Update of the HITRAN collision-induced
> absorption section.* Icarus, 328, 160–175.
> <https://doi.org/10.1016/j.icarus.2019.02.034>

Please also cite the current HITRAN edition per hitran.org's usage guidelines.
