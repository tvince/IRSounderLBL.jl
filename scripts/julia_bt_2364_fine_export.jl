"""
Phase-2 grid-spacing test: rerun Julia at Δν = 0.0005 cm⁻¹ (matching LBLRTM's
auto-set DV for the 4.3 µm run) over the narrow 2355-2375 cm⁻¹ window that
contains the +7 K Julia-vs-LBLRTM band-head spike at 2364 cm⁻¹.

If the spike collapses when we go from Δν = 0.005 to 0.0005 cm⁻¹ while
everything else is held fixed, the residual is grid undersampling of narrow
high-altitude Doppler lines, not far-wing/cutoff physics.

Mirrors scripts/julia_bt_43um_export.jl exactly, except:
  - NU_MIN/NU_MAX:  2355–2375 cm⁻¹ (narrow window, 20 cm⁻¹ wide)
  - DNU:            0.0005 cm⁻¹ (10× finer than the original run)
  - cont-OFF only   (the spike is in the pure-Voigt comparison)

Lines are loaded with a 25 cm⁻¹ margin on each side so the cutoff window
sees the same set of contributing transitions as the wider run.

Run with:
  julia --project -t auto scripts/julia_bt_2364_fine_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2355.0
const NU_MAX  = 2375.0
const DNU     = 0.0005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const OUTFILE = joinpath(OUTDIR, "julia_bt_2364_fine_contOFF.csv")

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.5f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

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

println("Loading CO2 linelist (iso 1–3) over $(NU_MIN-CUTOFF)–$(NU_MAX+CUTOFF) cm⁻¹…")
ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
@printf("  CO2: %d lines\n", length(ll_co2))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)
prof = afgl_us_standard_50lev()

@printf("\nForward model: cont-OFF, Δν=%.5f cm⁻¹ …\n", DNU)
t = time()
ν_out, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = 1,
                                   cutoff          = CUTOFF,
                                   apply_continuum = false,
                                   continua        = (),
                                   with_ils        = false,
                                   line_mixing     = nothing)
@printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
        time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))

open(OUTFILE, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out.ν, BT)
        write(f, "$(round(ν, digits=6)),$(round(bt, digits=6))\n")
    end
end
println("  wrote ", OUTFILE)
println("\nDone.")
