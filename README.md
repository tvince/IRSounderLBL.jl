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

## What's in the box

- HITRAN line-by-line absorption (HITRAN 2020 API + local `.par` loader)
- Three Voigt-profile evaluators — `Weideman`, `PseudoVoigt`, `FullFaddeeva` (default)
- TIPS-2024 partition functions (46 isotopologues)
- MT-CKD 4.3 H₂O self + foreign continuum (tabulated AER data)
- HITRAN CIA tables for CO₂, N₂, O₂
- CO₂ line mixing — first-order (Niro/Lamouroux VP_Y) and full-matrix (VP_W)
- Schwarzschild layer-by-layer RTE solver
- IASI instrument response: Gaussian apodization (L1C default) or Norton-Beer
- Off-nadir geometry: scan-angle → local-zenith conversion
- Standard atmospheres: US Standard, Tropical, Subarctic; 43- and 50-level AFGL

## Install

The package is not registered. From the Julia REPL:

```julia
] dev /path/to/IRSounderLBL
```

Requires Julia ≥ 1.10. A `HITRAN_API_KEY` environment variable is needed
if you want to fetch lines via `fetch_hitran_api`; otherwise local `.par`
files work standalone.

## Quick start

```julia
using IRSounderLBL

ll = load_hitran_par("data/co2_645_700.par"; ν_min=645.0, ν_max=700.0)
prof = us_standard_atmosphere()
linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll)

ν, R, BT = iasi_forward_model(prof, linelists)
```

Returns the IASI channel grid `ν` (cm⁻¹), spectral radiance `R`
(W m⁻² sr⁻¹ (cm⁻¹)⁻¹), and brightness temperature `BT` (K).

Common keyword arguments to `iasi_forward_model`:

- `iasi::IASIInstrument` — defaults to standard IASI L1C
- `geom::ViewingGeometry` — `nadir_geometry()` or use `scan_angle_to_local_zenith`
- `T_sfc`, `ε_sfc` — surface temperature override and emissivity
- `apply_continuum::Bool` — MT-CKD + CIA on/off
- `with_ils::Bool` — convolve with the IASI ILS
- `apodization::Symbol` — `:gaussian` (L1C) or `:norton_beer`
- `line_mixing` — `VPYLineMixing(...)` or `VPWLineMixing(...)`
- `high_res_factor::Int` — internal grid oversampling (default 4)

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
├── Sensor/        viewing geometry, ILS, IASI forward model
├── Parallel/      backend detection (CPU / CUDA / Metal)
└── Utils/         wavenumber grid, interpolation
test/              unit tests
scripts/           validation drivers, ARTS comparison, plotting
```

## References

- HITRAN 2020 — Gordon et al., JQSRT 277 (2022)
- TIPS-2024 v1.2 — Gamache et al., JQSRT 345 (2025)
- MT-CKD 4.3 — Mlawer et al. (AER); data files under `data/mt_ckd_h2o/` and `data/mt_ckd_co2/`
- Niro line mixing — Niro et al., JQSRT 88 (2004)
- Norton-Beer apodization — Norton & Beer, JOSA 66 (1976)

## Data and third-party licenses

The MIT license below covers the **software only**. It does not extend to the
spectroscopic and atmospheric reference data the package reads at runtime, each
of which carries its provider's own usage and citation terms. See the
[`NOTICE`](NOTICE) file for the full breakdown; the short version:

| Data | Source | Redistribution | You must cite |
|---|---|---|---|
| Line lists (`*.par`), CIA (`*.cia`) | [HITRAN](https://hitran.org) | **No** — fetch it yourself | HITRAN2020; Karman et al. 2019 |
| TIPS-2024 partition sums | Gamache et al. | with attribution | Gamache et al. 2025 |
| MT-CKD continuum tables | AER ([MT_CKD](https://github.com/AER-RC/MT_CKD)) | with attribution | Mlawer et al. 2012 |
| AFGL standard atmospheres | AFGL | freely | Anderson et al. 1986 |
| CO₂ line-mixing data | Lamouroux/Hartmann | with attribution | Lamouroux et al. 2015 |

**HITRAN data is not redistributable** and you are responsible for obtaining it
from hitran.org under HITRAN's data-use policy and for meeting its citation
requirement. See [`data/cia/SOURCE.md`](data/cia/SOURCE.md) for the exact files
and download steps.

## Citing this software

If you use IRSounderLBL.jl in published work, please cite it via the
[`CITATION.cff`](CITATION.cff) file (GitHub's "Cite this repository" button), in
addition to the underlying data sources above.

## License

MIT — see [LICENSE](LICENSE). Applies to the source code only; see
[Data and third-party licenses](#data-and-third-party-licenses) above and
[`NOTICE`](NOTICE) for data terms.
