# IRSounderLBL.jl

Line-by-line thermal infrared radiance simulation for nadir-viewing
hyperspectral Fourier transform sounders (IASI, CrIS, IASI-NG, MTG-IRS).

## Status

Research code, validated against ARTS 2.6 on the IASI spectral range
(645–2760 cm⁻¹):

| Component | RMS vs ARTS | Notes |
|---|---|---|
| Line-by-line (no continua) | **0.095 K** | bias −0.028 K; ceiling-limited by ARTS layer integration |
| With MT-CKD 4.3 continuum | **0.051 K** | full IASI spectrum |
| With CO₂ line mixing (15 µm) | **1.918 K** | ceiling-limited by ARTS issue [#1130](https://github.com/atmtools/arts/issues/1130); Julia agrees with Lamouroux Fortran reference to ~1.5 % |

The CO₂ bands are additionally validated against **LBLRTM v12.17** (MT_CKD 4.3):
~0.08 K RMS at 15 µm (line-by-line) and ~0.05 K at 4.3 µm with the CO₂
continuum. Those comparisons drove several method choices ported from LBLRTM —
the CIM source function, the DPTMIN line-rejection criterion, the mass-weighted
layer temperature (TBAR), and the AER band-head pedestal.

## What's in the box

- HITRAN line-by-line absorption (HITRAN 2020 API + local `.par` loader)
- Three Voigt-profile evaluators — `Weideman`, `PseudoVoigt`, `FullFaddeeva` (default)
- TIPS-2024 partition functions (46 isotopologues)
- MT-CKD 4.3 H₂O self + foreign continuum (tabulated AER data) and CO₂ continuum
- HITRAN CIA tables for CO₂, N₂, O₂
- CO₂ line mixing — first-order (Niro/Lamouroux VP_Y) and full-matrix (VP_W)
- Schwarzschild layer-by-layer RTE solver
- IASI instrument response: Gaussian apodization (L1C default) or Norton-Beer
- Off-nadir geometry: scan-angle → local-zenith conversion
- Standard atmospheres: US Standard, Tropical, Subarctic; 43- and 50-level AFGL
- Analytic Jacobians (temperature, VMR, surface), validated against finite
  differences including continuum and line-mixing coupling
- Optimal-estimation retrieval (`optimal_estimation`) with averaging kernels,
  DOF and the Rodgers error budget; a-priori covariance builder (`build_sa`);
  reduced VMR bases (`ColumnScale`, `PartialColumns`)
- Real-data ingest: IASI L1C granules (`read_iasi_l1c`) and EUMETSAT instrument
  noise covariance as retrieval Sₑ (`read_iasi_ncm`)

## Install

The package is not yet registered. From the Julia REPL:

```julia
] add https://github.com/tvince/IRSounderLBL.jl
```

or, for local development, `] dev /path/to/IRSounderLBL`.

Requires Julia ≥ 1.10. A `HITRAN_API_KEY` environment variable is needed
if you want to fetch lines via `fetch_hitran_api`; otherwise local `.par`
files work standalone.

## Getting the data

Spectroscopic data that HITRAN does not allow us to redistribute is fetched
rather than bundled. Start here — this prints a checklist of what you have, what
you're missing, and the exact next step for each gap:

```julia
using IRSounderLBL
data_status()
```

**One-time setup:**

```julia
download_data()              # HITRAN CIA tables, ~12 MB, SHA-256 verified, no key
download_data(:linelists)    # HITRAN line lists — needs a free API key (below)
```

Then the forward model runs:

```julia
prof = afgl_us_standard_50lev()
ν, R, BT = forward_model(prof, default_linelists())
```

### HITRAN API key

Line lists come from the HITRAN API, which needs a free key:

1. register at <https://hitran.org/register/>
2. copy your key from <https://hitran.org/profile/>
3. `export HITRAN_API_KEY=<your key>` in your shell — never in the repo

`download_data(:linelists)` defaults to the **15 µm working set** (CO₂ isotopologues
1–4, H₂O 1–3, 620–825 cm⁻¹ — the ν₂ band the package is validated against, plus the
±25 cm⁻¹ line-wing margin). Widen it when you need to:

```julia
download_data(:linelists; ν_min = 620.0, ν_max = 2785.0,
              species = [CO2 => 1:4, H2O => 1:3, O3 => 1:4])   # full IASI range
```

Unlike the CIA tables, line lists carry no pinnable checksum — HITRAN serves the
current release and revises it over time. Since a retrieval's results depend on the
line list behind it, each download appends to `linelists/PROVENANCE.md` recording
the query, the date, and the line counts.

### Where it goes

Downloads land in a package-owned scratch space by default; `IRSOUNDER_DATA_DIR` or
`set_data_dir!(path)` overrides that. Resolution order is override → `<pkg>/data` →
scratch, so a copy you drop in `data/` always wins.

### Optional data we cannot fetch for you

`data_status()` also reports these, with links — they sit behind accounts or are
scene-specific:

| Data | Needed for | Source |
|---|---|---|
| CO₂ line-mixing (HITRAN2020) | `VPYLineMixing`, `VPWLineMixing` | [hitran.org/supplementary](https://hitran.org/supplementary/) → Line-Mixing (login) |
| IASI L1C granules | real-radiance retrievals | [NOAA CLASS](https://www.class.noaa.gov/) |
| EUMETSAT IASI NCM | instrument noise covariance as `Se` | [EUMETSAT Data Store](https://data.eumetsat.int/) |

Without the CIA tables, `co2_cia`/`n2_cia`/`o2_cia` warn once and return zeros and
everything else works normally. Without a line list, the forward model cannot run.

## Quick start

Once `download_data(:linelists)` has run (see [Getting the data](#getting-the-data)):

```julia
using IRSounderLBL

prof = afgl_us_standard_50lev()
ν, R, BT = forward_model(prof, default_linelists())
```

`default_linelists()` loads whatever `download_data(:linelists)` installed. To use
your own `.par` files instead:

```julia
ll = load_linelist("path/to/co2_645_700", 1:4; ν_min=620.0, ν_max=825.0)
ν, R, BT = forward_model(prof, Dict{GasSpecies, HITRANLinelist}(CO2 => ll))
```

Returns the sensor channel grid `ν` (cm⁻¹), spectral radiance `R`
(W m⁻² sr⁻¹ (cm⁻¹)⁻¹), and brightness temperature `BT` (K).

Common keyword arguments to `forward_model`:

- `sounder::Sounder` — defaults to `IASIInstrument()`; also `IASINGInstrument()`,
  `CrISInstrument()`, `MTGIRSInstrument()`
- `geom::ViewingGeometry` — `nadir_geometry()` or use `scan_angle_to_local_zenith`
- `T_sfc`, `ε_sfc` — surface temperature override and emissivity
- `apply_continuum::Bool` — MT-CKD + CIA on/off
- `with_ils::Bool` — convolve with the instrument ILS
- `apodization::Symbol` — `:gaussian` (L1C) or `:norton_beer`
- `line_mixing` — `VPYLineMixing(...)` or `VPWLineMixing(...)`
- `internal_dnu::Float64` — absolute internal monochromatic grid spacing in cm⁻¹
  (default `0.001`), auto-adapting to any sensor Δν. Converges ILS-convolved BT
  to ≤6 mK RMS against a 0.0005 reference

The older `iasi_forward_model` / `iasi_grid` names remain as aliases, and
`high_res_factor` still overrides `internal_dnu` as a legacy escape hatch — but
it is oversampling relative to the *sensor* Δν, so it does not adapt across
instruments. The former default of 4 (0.0625 cm⁻¹ for IASI) is ≈1 K off in
dense bands; prefer `internal_dnu`.

See `smoke_test.jl` and `scripts/` for end-to-end examples including
line-mixing runs and ARTS comparison drivers.

## Performance

Full IASI spectrum (645–2760 cm⁻¹, 50-level profile, 6 species × 3 isotopologues)
on an M1 Pro:

- 6 threads: **130 s**
- single thread: 391 s

Run with `julia --project -t auto scripts/julia_bt_export.jl`. The outer
layer loop is still sequential; parallelising it is the next opportunity.

## Layout

```
src/
├── Atmosphere/    profiles, layers, species, standard atmospheres
├── HITRAN/        linelist I/O, partition functions, broadening
├── CrossSections/ Voigt, continuum, line mixing, optical depth
├── Solver/        Planck, transmittance, Schwarzschild RTE
├── Sensor/        viewing geometry, ILS, sounder presets, forward model
├── Estimation/    Jacobians, optimal estimation, covariances, state vectors
├── Parallel/      backend detection (CPU / CUDA / Metal)
└── Utils/         wavenumber grid, interpolation
test/              unit tests
scripts/           validation drivers, ARTS comparison, plotting
```

## Acknowledgements

The architecture and much of the methodology of this package were guided by Anu
Dudhia's [Reference Forward Model (RFM)](http://eodg.atm.ox.ac.uk/RFM/) (Dudhia,
JQSRT 186, 243–253, 2017). The RFM's influence runs through the design: the
pipeline decomposition (profile → layer quantities → line-by-line cross-sections
→ optical depth → RTE → instrument convolution), the Curtis–Godson
effective-pressure/VMR treatment of inhomogeneous layers, the plane-parallel
non-scattering Schwarzschild formulation including the reflected-downwelling
surface term (RFM Eq. 14), and the IASI L1C ingest, which is ported from
Dudhia's reference reader
[`read_iasi_l1c.py`](https://eodg.atm.ox.ac.uk/user/dudhia/iasi/read_iasi_l1c/).

LBLRTM (AER) served as the validation reference for the CO₂ bands and
contributed the CIM source function, DPTMIN criterion, TBAR layer-temperature
convention and band-head pedestal; ARTS 2.6 was the primary full-spectrum
validation reference.

## References

- HITRAN 2020 — Gordon et al., JQSRT 277 (2022)
- TIPS-2024 v1.2 — Gamache et al., JQSRT 345 (2025)
- MT-CKD 4.3 — Mlawer et al. (AER); data files under `data/mt_ckd_h2o/` and `data/mt_ckd_co2/`
- Niro line mixing — Niro et al., JQSRT 88 (2004)
- Norton-Beer apodization — Norton & Beer, JOSA 66 (1976)
- RFM — Dudhia, *The Reference Forward Model (RFM)*, JQSRT 186, 243–253 (2017),
  [doi:10.1016/j.jqsrt.2016.06.018](https://doi.org/10.1016/j.jqsrt.2016.06.018);
  <http://eodg.atm.ox.ac.uk/RFM/>. Source of the reflected-downwelling surface
  formulation (RFM Eq. 14); the IASI L1C reader layout follows Dudhia's
  [`read_iasi_l1c.py`](https://eodg.atm.ox.ac.uk/user/dudhia/iasi/read_iasi_l1c/)
- LBLRTM — Clough et al., *Atmospheric radiative transfer modeling: a summary of
  the AER codes*, JQSRT 91, 233–244 (2005),
  [doi:10.1016/j.jqsrt.2004.05.058](https://doi.org/10.1016/j.jqsrt.2004.05.058);
  <https://github.com/AER-RC/LBLRTM>. Validation reference for the CO₂ bands and
  source of the CIM source function, DPTMIN criterion, TBAR layer-temperature
  convention and band-head pedestal
- ARTS 2.6 — Buehler et al., GMD 11, 1537–1556 (2018),
  [doi:10.5194/gmd-11-1537-2018](https://doi.org/10.5194/gmd-11-1537-2018).
  Primary full-spectrum validation reference

## Data and third-party licenses

The MIT license below covers the **software only**. It does not extend to the
spectroscopic and atmospheric reference data the package reads at runtime, each
of which carries its provider's own usage and citation terms. See the
[`NOTICE`](NOTICE) file for the full breakdown; the short version:

| Data | Source | Bundled? | Redistribution | You must cite |
|---|---|---|---|---|
| Line lists (`*.par`), CIA (`*.cia`) | [HITRAN](https://hitran.org) | no | **No** — fetch it yourself | HITRAN2020; Karman et al. 2019 |
| TIPS-2024 partition sums | Gamache et al. ([Zenodo](https://doi.org/10.5281/zenodo.17191976)) | yes | Yes — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), modified subset | Gamache et al. 2025 |
| MT-CKD continuum tables | AER ([MT_CKD](https://github.com/AER-RC/MT_CKD)) | yes | Yes, with attribution — **research use only** | Mlawer et al. 2012 |
| AFGL standard atmospheres | AFGL | yes | freely | Anderson et al. 1986 |
| CO₂ line-mixing data | Lamouroux/Hartmann (via HITRAN) | no | **No** — fetch it yourself | Lamouroux et al. 2015 |

**HITRAN data is not redistributable** and you are responsible for obtaining it
from hitran.org under HITRAN's data-use policy and for meeting its citation
requirement. This covers both the line lists/CIA and the CO₂ line-mixing
relaxation-matrix package — neither is bundled here. See
[`data/cia/SOURCE.md`](data/cia/SOURCE.md) for the exact files and download
steps.

**The distribution as a whole is not OSI open source.** The MIT license covers
the code, but the bundled AER MT-CKD coefficient tables carry a research-use
license that forbids incorporation into proprietary or commercial software
without AER's written consent. If that matters to you, drop
`data/mt_ckd_*/` and supply your own continuum coefficients. The other bundled
data (TIPS under CC-BY-4.0, AFGL) carries no such restriction.

## Citing this software

If you use IRSounderLBL.jl in published work, please cite it via the
[`CITATION.cff`](CITATION.cff) file (GitHub's "Cite this repository" button), in
addition to the underlying data sources above.

## License

MIT — see [LICENSE](LICENSE). Applies to the source code only; see
[Data and third-party licenses](#data-and-third-party-licenses) above and
[`NOTICE`](NOTICE) for data terms.
