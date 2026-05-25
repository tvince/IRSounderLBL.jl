"""
Export Julia BT spectra matching the LBLRTM validation run (Phase 3).

Config mirrors the LBLRTM TAPE5/TAPE3 setup:
  - CO2 only, all 3 isotopologues (no H2O lines; matches CO2-only TAPE3)
  - line mixing OFF (pure Voigt)
  - AFGL US Standard 50-level profile, nadir from TOA, blackbody surface
  - no ILS, fine monochromatic grid Δν = 0.005 cm⁻¹ (high_res_factor = 1)

Runs the forward model TWICE:
  - apply_continuum = false  -> data/lblrtm/julia_bt_contOFF.csv
  - apply_continuum = true   -> data/lblrtm/julia_bt_contON.csv

Run with:
  julia --project -t auto scripts/julia_bt_lblrtm_compare_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 645.0
const NU_MAX  = 800.0
const DNU     = 0.005          # fine monochromatic grid
const CUTOFF  = 25.0           # matches LBLRTM ILBLF4=1 (25 cm⁻¹)
const OUTDIR  = "data/lblrtm"

# Fine output grid with no oversampling (internal grid == output grid).
n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

# CO2 all-iso linelist (same files merged into the LNFL TAPE1), 25 cm⁻¹ margin.
function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

println("Loading CO2 linelist (iso 1–3)…")
ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
@printf("  CO2: %d lines\n", length(ll_co2))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)
prof = afgl_us_standard_50lev()

function run_and_write(apply_cont::Bool, outfile)
    @printf("\nForward model: apply_continuum=%s …\n", apply_cont)
    t = time()
    ν_out, _, BT = iasi_forward_model(prof, linelists;
                                       iasi            = iasi,
                                       high_res_factor = 1,
                                       cutoff          = CUTOFF,
                                       apply_continuum = apply_cont,
                                       with_ils        = false,
                                       line_mixing     = nothing)
    @printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
            time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))
    open(outfile, "w") do f
        write(f, "nu_cm1,BT_K\n")
        for (ν, bt) in zip(ν_out.ν, BT)
            write(f, "$(round(ν, digits=5)),$(round(bt, digits=6))\n")
        end
    end
    println("  wrote ", outfile)
end

run_and_write(false, joinpath(OUTDIR, "julia_bt_contOFF.csv"))
run_and_write(true,  joinpath(OUTDIR, "julia_bt_contON.csv"))
println("\nDone.")
