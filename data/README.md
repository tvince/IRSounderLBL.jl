# `data/` — runtime inputs and generated artifacts

This directory is `.gitignore`d. Files here are either (a) downloaded from
external sources, or (b) outputs from scripts in `scripts/`. Use this README
as a map; regenerate anything missing rather than checking it in.

(This README itself is force-added: `git add -f data/README.md`.)

## External inputs (download, do not regenerate)

| Path | Source |
| --- | --- |
| `*.par` (e.g. `co2_645_2760.par`, `h2o_645_2760.par`, …) | HITRANonline (HITRAN 2020). Main + `_iso2`/`_iso3` variants are separate isotopologue downloads. |
| `arts_pf/*.xml` | ARTS distribution — partition functions. |
| `broadening/` | HITRAN broadening parameters (γ_self, n_self, δ_self). |
| `mt_ckd_h2o/` | MT-CKD 4.3 H₂O continuum tables (AER). |
| `tips/` | TIPS-2024 raw tables. |
| `Line-mixing_HITRAN2020/` | Lamouroux/HITRAN line-mixing data + Fortran reference. |
| `standard_atmospheres/` | AFGL standard atmospheres (raw). |

## Repo-tracked seeds (pre-prepared inputs, not generated)

These are committed under `src/` or `data_in/` in some form; the copies here
are derived for runtime convenience.

| File | Built by |
| --- | --- |
| `afgl_us_standard_50lev.csv` | `scripts/build_afgl_50lev.py` |
| `us_standard_atm.csv` | (hand-prepared from AFGL profile) |
| `tips2024_qt.csv` | `scripts/dump_tips2024.jl` (via `extract_tips.py`) |

## Generated artifacts (regenerable)

### ARTS reference BT spectra
| File | Script |
| --- | --- |
| `arts_bt_iasi.csv` | `scripts/arts_validation.py` |
| `arts_bt_iasi_cont.csv` | `scripts/arts_validation_cont.py` |
| `arts_bt_co2_15um_lm.csv` | `scripts/arts_validation_cont_lm.py` |
| `arts_bt_co2_15um_lm_vpw.csv` | `scripts/arts_validation_cont_lm_vpw.py` |

### Julia outputs
| File | Script |
| --- | --- |
| `julia_bt_645_800.csv` | `scripts/julia_bt_export.jl` |
| `julia_bt_cont.csv` | `scripts/julia_bt_export_cont.jl` |
| `julia_bt_co2_15um_lm.csv` | `scripts/julia_bt_co2_lm_export.jl` |
| `julia_bt_co2_15um_vpw.csv`, `…_vpw_iso12.csv` | `scripts/julia_bt_co2_15um_vpw_export.jl` |
| `julia_xsec_*.csv` | `scripts/compare_xsec_julia.jl`, `compare_xsec_single_layer.py` |

### MT-CKD continuum extraction
| File | Script |
| --- | --- |
| `ckd_mt350_coeffs.csv` (regenerate as needed) | `scripts/extract_ckd_coeffs.py` |
| (MT-CKD 4.3 csv) | `scripts/extract_ckd43_csv.py` |

### ARTS issue #1130 reproducer artifacts (15 µm CO₂ Y bug)
Cited in `ARTS_BUG_REPORT.md` / GitHub issue #1130.
| File | Script |
| --- | --- |
| `julia_Y_near_665.csv` | `scripts/dump_julia_Y_near_665.jl` |
| `arts_Y_near_665.csv` | `scripts/dump_arts_Y_near_665.py` |
| `diff_arts_julia_Y_near_665.csv` | `scripts/diff_arts_julia_Y.py` (full 427-row 3-way diff) |
| `arts_bug_iso1_p_branch_diff.csv` | `scripts/diff_arts_julia_Y.py` (32-row iso-1 P-branch focus) |

Full rebuild (~30 s):
```
julia --project=. scripts/dump_julia_Y_near_665.jl
.../arts_env/bin/python scripts/dump_arts_Y_near_665.py
.../arts_env/bin/python scripts/diff_arts_julia_Y.py
```

### Plots (regenerate from the CSVs above)
`arts_julia_*.png`, `vpw_*.png`, `arts_lm_effect_co2_15um.png`,
`lm_split_diagnostic.png`, `partition_function_comparison.png` — produced by
the various `compare_*.py` / `plot_*.jl` scripts.
