# CO₂ line-mixing data provenance

Relaxation-matrix data for CO₂ line mixing, read by `load_hitran_relmat` in
`src/CrossSections/line_mixing.jl` and used by the `VPYLineMixing` (first-order)
and `VPWLineMixing` (full-matrix) models.

These files are **not tracked in git** and do **not** ship with the package:
they are part of HITRAN and carry HITRAN's data-use policy, which permits use
with citation but **not redistribution**. Unlike the CIA tables, this package is
behind a login, so `download_data()` cannot fetch it — get it yourself.

## Source

HITRAN supplementary material: <https://hitran.org/supplementary/> → the
**Line-Mixing** section → <https://hitran.org/files/LM/>

`/files/LM/` requires a (free) HITRAN account; it redirects to the login page
otherwise. Download the CO₂ line-mixing package for **HITRAN2020** and unpack it
here. HITRAN's own note: the earlier HITRAN2016 line-mixing package is *not*
recommended for use with HITRAN2020 line data, so make sure you take the 2020 one
— this package's linelists are HITRAN2020.

## Layout expected by the loader

| Path | Role |
|---|---|
| `data_new/` | What you pass as `basedir` to `load_hitran_relmat`. ~6,600 files: `BandInfo.dat`, `Excluded_bands.dat`, and per-band `S<band>.dat` relaxation-matrix files. |
| `data_new/BandInfo.dat` | Band index — parsed to select bands overlapping `[ν_min, ν_max]` above `stot_min`. |
| `ABSCO_HIT2020.dat` | HITRAN2020 absorption coefficients supplied with the package. |
| `LM_calc_CO2.for`, `LM_calc_15um.for`, `parameters.inc` | Authors' Fortran reference implementation. Not read by Julia, but this is what the Julia `VP_Y` values were validated against (agreement ≈1.5%, which established that the ARTS Y divergence near 665 cm⁻¹ was a bug in ARTS, not here — see `docs/ARTS_CIA_BUG_REPORT.md` lineage and atmtools/arts#1130). |
| `Instructions_LM_CO2.pdf` | Authors' usage notes, supplied with the package. |

Note the S-file names carry **trailing spaces** (`S00001100001  .dat`); the loader
tries the padded name first and falls back to the unpadded form.

Size is ~430 MB unpacked (~419 MB of it `data_new/`).

## Citation

> J. Lamouroux, L. Régalia, X. Thomas, J. Vander Auwera, R. R. Gamache,
> J.-M. Hartmann, *CO₂ line-mixing database and software update and its tests in
> the 2.1 µm and 4.3 µm regions*, J. Quant. Spectrosc. Radiat. Transfer 151,
> 88–96 (2015). <https://doi.org/10.1016/j.jqsrt.2014.09.017>

> J.-M. Hartmann, C. Boulet, D. Robert, *Line mixing and finite duration of
> collision effects in pure CO₂ infrared spectra*, J. Quant. Spectrosc. Radiat.
> Transfer 110, 2019–2026 (2009).

Please also cite the current HITRAN edition per hitran.org's usage guidelines.
See the repository `NOTICE` (item 5) for the licensing summary.

## Related, different lineage

The CH₄ ν₄ line-mixing work uses a **separate** dataset — Tran, Flaud & Hartmann
(2006) — not this package.
