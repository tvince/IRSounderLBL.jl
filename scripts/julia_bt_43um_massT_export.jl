"""
Phase-3 interpolation test: rerun the 2355-2375 cm⁻¹ window with the
LBLATM-style mass-weighted layer temperature (T_method=:mass_weighted in
layer_properties) and see whether the +5 to +7 K per-line core comb at
CO₂ ν₃ band-head lines collapses.

Default Julia uses T_method=:logp_at_pcg (T linearly interpolated in log(p)
at the Curtis-Godson pressure). LBLRTM TAPE6 prints TBAR = ∫T dp/Δp, which
is the mass-weighted layer T. The two differ in layers with strong T
gradient (5-km layers above 50 km), with a sign-flipping offset around the
stratopause where T(z) peaks — exactly the pattern of the residual.

Same window/grid as scripts/julia_bt_2364_fine_export.jl was, but at the
original Δν=0.005 cm⁻¹ (we already showed grid spacing doesn't matter).

Run with:
  julia --project -t auto scripts/julia_bt_43um_massT_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2355.0
const NU_MAX  = 2375.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const OUTFILE = joinpath(OUTDIR, "julia_bt_43um_massT_contOFF.csv")

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

println("Loading CO2 linelist (iso 1–3) over $(NU_MIN-CUTOFF)–$(NU_MAX+CUTOFF) cm⁻¹…")
ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
@printf("  CO2: %d lines\n", length(ll_co2))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)
prof = afgl_us_standard_50lev()

@printf("\nForward model: cont-OFF, T_method=:mass_weighted, Δν=%.4f …\n", DNU)
t = time()
ν_out, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = 1,
                                   cutoff          = CUTOFF,
                                   apply_continuum = false,
                                   continua        = (),
                                   with_ils        = false,
                                   line_mixing     = nothing,
                                   T_method        = :mass_weighted)
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
