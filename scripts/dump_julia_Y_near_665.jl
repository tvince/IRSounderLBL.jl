# Dump Julia's computed Y values for CO2 LM lines near 665 cm⁻¹.
# Memory: Julia's Y matches Fortran reference (LM_calc_15um.for) to 4+ sig
# figs (Q8, Q14 at T=260K, P=0.5atm).  So Julia/Fortran Y values are "ground
# truth"; if ARTS disagrees, it's an ARTS implementation difference.
#
# Conditions: AFGL US Standard surface layer, ~T=288K, P~1 atm.
#
# What we report per line:
#   ν₀ (cm⁻¹)
#   Y (per atm)
#   |Y·p|  — first-order Rosenkranz validity bound is ≲ 0.1
#   S(T)·γ_L — to weight LM importance

using RadiativeTransfer
using Printf

const NU_LO = 664.5
const NU_HI = 665.5
const T = 288.20      # AFGL US Std surface temperature
const P_atm = 1.0132  # surface pressure
const LM_DIR = "data/Line-mixing_HITRAN2020/data_new"

# Load relmat + WTfit
relmat = load_hitran_relmat(LM_DIR, 645.0, 800.0; stot_min=0.0)
@printf("Loaded %d bands\n\n", length(relmat.bands))

# Internal helpers — replicated from line_mixing.jl
const _CT_LM = 1.4387769
const _T0_LM = 296.0
const _CTGAMD = 3.5812e-7
const _CO2_MASS = Dict(1=>43.98983, 2=>44.99318, 3=>45.99398, 4=>44.99403,
                       5=>45.99706, 6=>44.99706, 7=>47.99832, 8=>47.00134,
                       9=>48.00196, 10=>47.00427)

println("Lines in $(NU_LO)–$(NU_HI) cm⁻¹ with |Y| > 0:")
@printf("%-13s  %3s  %3s  %-2s  %9s  %7s  %8s  %9s  %9s\n",
        "band", "iso", "Ji", "br", "ν₀", "γ_air", "Y/atm", "Y·p", "S(T)")

# `_calc_W_and_Y` is internal but `getfield` can pull it from the flat module.
calc_W_and_Y = getfield(RadiativeTransfer, :_calc_W_and_Y)

big_Y = NamedTuple[]
for band in relmat.bands
    Int(band.li) > 8 && continue
    lli = Int8(min(band.li, band.lf))
    llf = Int8(max(band.li, band.lf))
    wtfit = get(relmat.wtfit, (lli, llf), nothing)
    wtfit === nothing && continue

    Y_band = calc_W_and_Y(band, wtfit, T)

    # Per-line: S(T) using same formula as _lm_band_dispersive
    iso = Int(band.isot == 10 ? 10 : band.isot)
    Q_ratio = partition_function(2, iso, _T0_LM) / partition_function(2, iso, T)

    for (i, rl) in enumerate(band.lines)
        NU_LO ≤ rl.ν ≤ NU_HI || continue
        abs(Y_band[i]) > 0 || continue

        stim_T0 = 1.0 - exp(-_CT_LM * rl.ν / _T0_LM)
        stim_T  = 1.0 - exp(-_CT_LM * rl.ν / T)
        S0 = rl.DipoT^2 * rl.PopuT0 * rl.ν * stim_T0
        S_T = S0 * Q_ratio *
              exp(-_CT_LM * rl.E_lower * (1.0/T - 1.0/_T0_LM)) *
              stim_T / stim_T0

        Yp = Y_band[i] * P_atm
        @printf("%-13s  %3d  %3d  %+2d  %9.5f  %7.4f  %+8.4f  %+9.5f  %.2e\n",
                band.name, iso, Int(rl.Ji), Int(rl.branch),
                rl.ν, rl.gV_air, Y_band[i], Yp, S_T)

        if abs(Yp) > 0.02
            push!(big_Y, (band=band.name, iso=iso, Ji=Int(rl.Ji), br=Int(rl.branch),
                          ν=rl.ν, Y=Y_band[i], Yp=Yp, S=S_T))
        end
    end
end

println("\nLines with |Y·p| > 0.02 (would be significant):")
sort!(big_Y, by=x->-abs(x.Yp))
for e in big_Y
    @printf("  %s  J=%d  br=%+d  ν=%.4f  Y·p=%+.4f  S=%.2e\n",
            e.band, e.Ji, e.br, e.ν, e.Yp, e.S)
end

@printf("\nRosenkranz validity bound: |Y·p| ≲ 0.1.  Max |Y·p| found: %.4f\n",
        isempty(big_Y) ? 0.0 : maximum(abs(e.Yp) for e in big_Y))
