"""
Single-session LBLRTM validation driver.

Runs BOTH cross-checks in ONE Julia process so the package compiles only once
and there is no second cold start to contend on the juliaup lock. (Launching two
`julia` jobs in parallel previously deadlocked on juliaup's startup self-update;
that auto-update is now disabled, but single-session is still faster and safer.
See memory note project-cim-default.)

Picks up the current `iasi_forward_model` defaults (source_function=:cim).
Config mirrors the LBLRTM TAPE5/TAPE3 setup for both bands:
  - CO₂ only, all 3 isotopologues (CO2-only TAPE3), line mixing OFF (pure Voigt)
  - AFGL US Standard 50-level profile, nadir from TOA, blackbody surface
  - no ILS, fine monochromatic grid Δν = 0.005 cm⁻¹ (high_res_factor = 1)

Outputs (data/lblrtm/), each diffed by scripts/diff_lblrtm_validation.py:
  15 µm  645–800:   julia_bt_contOFF.csv        vs lblrtm_bt_contOFF_g12.csv
                    julia_bt_contON.csv         vs lblrtm_bt_contON_g12.csv
  4.3 µm 2000–2500: julia_bt_43um_contOFF.csv   vs lblrtm_bt_43um_contOFF.csv
                    julia_bt_43um_co2cont.csv   vs lblrtm_bt_43um_co2cont.csv

Run with:
  julia --project -t auto scripts/validate_lblrtm_all.jl
"""

using IRSounderLBL
using Printf

const DNU    = 0.005     # fine monochromatic grid
const CUTOFF = 25.0      # matches LBLRTM ILBLF4=1 (25 cm⁻¹)
const OUTDIR = "data/lblrtm"

# CO₂ all-iso linelist (same files merged into the LNFL TAPE1), 25 cm⁻¹ margin.
function load_co2(ν_min, ν_max; base = "co2_645_2760")
    all_lines = HITRANLine[]
    for iso_id in 1:3
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min = ν_min - CUTOFF, ν_max = ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

# One forward-model run + CSV write. `continua === nothing` means "use the
# iasi_forward_model default continua" (the 15 µm cont-ON convention).
function run_and_write(prof, linelists, iasi, outfile; apply_cont, continua)
    @printf("\n  fwd: apply_continuum=%s continua=%s …\n", apply_cont,
            continua === nothing ? "default" : continua)
    t = time()
    kw = (; iasi, high_res_factor = 1, cutoff = CUTOFF,
            apply_continuum = apply_cont, with_ils = false, line_mixing = nothing)
    ν_out, _, BT = continua === nothing ?
        iasi_forward_model(prof, linelists; kw...) :
        iasi_forward_model(prof, linelists; kw..., continua = continua)
    @printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
            time() - t, minimum(BT), maximum(BT), sum(BT) / length(BT))
    open(outfile, "w") do f
        write(f, "nu_cm1,BT_K\n")
        for (ν, bt) in zip(ν_out.ν, BT)
            write(f, "$(round(ν, digits = 5)),$(round(bt, digits = 6))\n")
        end
    end
    println("  wrote ", outfile)
end

# Build the IASI grid for a band (internal grid == output grid, no oversampling).
function band_instrument(ν_min, ν_max)
    n_ch = round(Int, (ν_max - ν_min) / DNU) + 1
    @printf("grid: %.1f–%.1f cm⁻¹, Δν=%.4f, %d points\n", ν_min, ν_max, DNU, n_ch)
    return IASIInstrument(ν_min, ν_max, DNU, n_ch, 2.0, 0.5)
end

prof = afgl_us_standard_50lev()

# ── 15 µm band (645–800) ──────────────────────────────────────────────────────
println("\n=== 15 µm band (645–800 cm⁻¹) ===")
iasi_15 = band_instrument(645.0, 800.0)
println("Loading CO2 linelist (iso 1–3)…")
ll_15 = load_co2(645.0, 800.0); @printf("  CO2: %d lines\n", length(ll_15))
ll_dict_15 = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_15)
run_and_write(prof, ll_dict_15, iasi_15, joinpath(OUTDIR, "julia_bt_contOFF.csv");
              apply_cont = false, continua = ())
run_and_write(prof, ll_dict_15, iasi_15, joinpath(OUTDIR, "julia_bt_contON.csv");
              apply_cont = true,  continua = nothing)   # default continua

# ── 4.3 µm band (2000–2500) ───────────────────────────────────────────────────
println("\n=== 4.3 µm band (2000–2500 cm⁻¹) ===")
iasi_43 = band_instrument(2000.0, 2500.0)
println("Loading CO2 linelist (iso 1–3)…")
ll_43 = load_co2(2000.0, 2500.0); @printf("  CO2: %d lines\n", length(ll_43))
ll_dict_43 = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_43)
run_and_write(prof, ll_dict_43, iasi_43, joinpath(OUTDIR, "julia_bt_43um_contOFF.csv");
              apply_cont = false, continua = ())
run_and_write(prof, ll_dict_43, iasi_43, joinpath(OUTDIR, "julia_bt_43um_co2cont.csv");
              apply_cont = true,  continua = (:co2,))

println("\nDone. Diff with: python scripts/diff_lblrtm_validation.py")
