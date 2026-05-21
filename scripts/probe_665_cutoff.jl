# Test whether the per-line cutoff explains the 33 K ARTS gap at 665 cm⁻¹.
#
# ARTS evaluates each relmat line across the entire f_grid with no per-line
# cutoff.  Julia clips each line at ±25 cm⁻¹.  The dispersive Im[w] tail
# decays as 1/Δν (slower than the Voigt 1/Δν² Re[w] tail), so a 25 cm⁻¹
# cutoff may leave significant un-cancelled tails from Q-branch lines.
#
# Run Julia VP_Y over a narrow 663-668 cm⁻¹ window with:
#   1. cutoff = 25  (current default; matches existing memory result)
#   2. cutoff = 200 (effectively no cutoff inside the 645-800 band — every
#                     line covers the whole CO2 ν₂ region)
#
# Run:  julia --project -t auto scripts/probe_665_cutoff.jl

using RadiativeTransfer
using Printf

const NU_MIN  = 663.0
const NU_MAX  = 668.0
const LM_DIR  = "data/Line-mixing_HITRAN2020/data_new"
const DNU_OUT = 0.005          # native high-res, no downsampling
const HRF     = 1

function load_multi(iso_ids, ν_min, ν_max, base, cutoff)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - cutoff, ν_max=ν_max + cutoff)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

function run_case(label, cutoff)
    println("Loading relmat with cutoff = $cutoff cm⁻¹ …")
    # Load relmat over band-cutoff-bracketed window so all lines that can
    # contribute to 663-668 within the cutoff are included.
    relmat = load_hitran_relmat(LM_DIR,
                                 NU_MIN - cutoff,
                                 NU_MAX + cutoff;
                                 stot_min=0.0)
    ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760", cutoff)
    ll_h2o = load_multi(1:3, NU_MIN, NU_MAX, "h2o_645_2760", cutoff)
    @printf("  relmat: %d bands   CO2 lines: %d   H2O lines: %d\n",
            length(relmat.bands), length(ll_co2), length(ll_h2o))

    linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2, H2O => ll_h2o)
    prof = afgl_us_standard_50lev()

    n_ch = round(Int, (NU_MAX - NU_MIN) / DNU_OUT) + 1
    iasi = IASIInstrument(NU_MIN, NU_MAX, DNU_OUT, n_ch, 2.0, 0.5)

    t1 = time()
    ν, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = HRF,
                                   cutoff          = Float64(cutoff),
                                   apply_continuum = true,
                                   apply_ils       = false,
                                   line_mixing     = VPYLineMixing(relmat))
    @printf("  %-15s forward model: %.1f s\n", label, time() - t1)
    return ν.ν, BT
end

ν1, BT_c25  = run_case("cutoff=25",  25)
ν2, BT_c200 = run_case("cutoff=200", 200)

# Sanity: grids match
@assert ν1 == ν2

println("\nBT at 665.00 cm⁻¹:")
i = argmin(abs.(ν1 .- 665.0))
@printf("  cutoff=25:   %.3f K\n", BT_c25[i])
@printf("  cutoff=200:  %.3f K\n", BT_c200[i])
@printf("  Δ:           %+.3f K  (ARTS−Julia(25) gap is +33.5 K)\n",
        BT_c200[i] - BT_c25[i])

println("\nValues around 665.00 cm⁻¹ (cutoff=25 → cutoff=200, Δ):")
for ν_query in [664.5, 664.75, 664.95, 665.0, 665.025, 665.03, 665.04, 665.05, 665.25, 665.5, 666.0, 667.0]
    i = argmin(abs.(ν1 .- ν_query))
    @printf("  %.3f   %7.3f → %7.3f   Δ=%+6.3f K\n",
            ν_query, BT_c25[i], BT_c200[i], BT_c200[i] - BT_c25[i])
end

# Mean / max |Δ| over the whole 663-668 window
ΔBT = BT_c200 .- BT_c25
@printf("\n663-668 stats:  bias %+.4f K   RMS %.4f K   max|Δ| %.3f K at ν=%.3f\n",
        sum(ΔBT)/length(ΔBT), sqrt(sum(ΔBT.^2)/length(ΔBT)),
        maximum(abs.(ΔBT)), ν1[argmax(abs.(ΔBT))])
