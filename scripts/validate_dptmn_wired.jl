#!/usr/bin/env julia
#=
Re-validate the WIRED dptmn line-rejection in iasi_forward_model: run the full
forward model (real IASI ILS, :cim) with dptmn=0 (full line set) vs dptmn=1e-6,
in both the 15 µm and 4.3 µm CO2 bands, and compare channel BT. Target <0.01 K.

Run:  julia --project -t auto scripts/validate_dptmn_wired.jl
=#
using IRSounderLBL
const M = IRSounderLBL
using Printf, Statistics

const CUTOFF = 25.0
prof = M.afgl_us_standard_50lev()

load_co2(νlo, νhi) = Dict{M.GasSpecies,M.HITRANLinelist}(
    M.CO2 => M.load_linelist(joinpath("data","co2_645_2760"), 1:3;
                             ν_min = νlo - CUTOFF, ν_max = νhi + CUTOFF))

windows = [("15µm ", 645.0, 800.0), ("4.3µm", 2200.0, 2400.0)]

@printf("%-6s  %-9s %-9s %-9s\n", "band", "max|ΔBT|", "RMS ΔBT", "(target<0.01K)")
for (name, νlo, νhi) in windows
    ll  = load_co2(νlo, νhi)
    nch = round(Int, (νhi - νlo) / 0.25) + 1
    inst = M.IASIInstrument(νlo, νhi, 0.25, nch, 2.0, 0.5)
    kw = (; iasi = inst, cutoff = CUTOFF, apply_continuum = false, continua = ())
    _, _, BT_full = M.iasi_forward_model(prof, ll; kw..., dptmn = 0.0)
    _, _, BT_rej  = M.iasi_forward_model(prof, ll; kw..., dptmn = 1e-6)
    d = BT_rej .- BT_full
    @printf("%-6s  %.2e  %.2e  %s\n", name, maximum(abs.(d)),
            sqrt(mean(d.^2)), maximum(abs.(d)) < 0.01 ? "✓" : "✗")
end
println("Done.")
