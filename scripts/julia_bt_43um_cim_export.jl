"""
Phase-6 test: rerun the 2355-2375 cm⁻¹ window with the LBLRTM-style
Clough-Iacono-Moncet source function (source_function=:cim) AND mass-weighted
layer T (T_method=:mass_weighted), so that both opacity (cross-section) and
emission (Planck) are evaluated at the SAME Curtis-Godson T_AVE per layer.

This is the CG-consistent recipe.  If the +5..+7 K Julia−LBLRTM comb at line
cores collapses, the residual was driven by the source-function/T-mean
inconsistency in Julia's earlier code (CG TAVE for σ, level T's for B).

Same window/grid as scripts/julia_bt_43um_massT_export.jl, only the source
function changes.

Run with:
  julia --project -t auto scripts/julia_bt_43um_cim_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2355.0
const NU_MAX  = 2375.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const OUTFILE = joinpath(OUTDIR, "julia_bt_43um_cim_contOFF.csv")

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

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

@printf("\nForward model: cont-OFF, source_function=:cim, T_method=:mass_weighted …\n")
t = time()
ν_out, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = 1,
                                   cutoff          = CUTOFF,
                                   apply_continuum = false,
                                   continua        = (),
                                   with_ils        = false,
                                   line_mixing     = nothing,
                                   T_method        = :mass_weighted,
                                   source_function = :cim)
@printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
        time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))

open(OUTFILE, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out.ν, BT)
        write(f, "$(round(ν, digits=5)),$(round(bt, digits=6))\n")
    end
end
println("  wrote ", OUTFILE)
println("\nDone.")
