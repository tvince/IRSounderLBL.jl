"""
Probe Julia's CIA absorption coefficient at the same (p, T, ν-grid) as
`scripts/dump_arts_cia_sigma.py`. Writes data/julia_cia_sigma_dump.csv.

Run with:
  julia --project=. scripts/dump_julia_cia_sigma.jl
"""

using RadiativeTransfer
using DelimitedFiles
using Printf

const ROOT     = dirname(@__DIR__)
const OUT_FILE = joinpath(ROOT, "data", "julia_cia_sigma_dump.csv")

const P_HPA   = 1013.0
const T_K     = 288.0
const VMR_H2O = 0.00626
const VMR_CO2 = 365e-6
const VMR_N2  = 0.78084 * (1.0 - VMR_H2O)
const VMR_O2  = 0.20946 * (1.0 - VMR_H2O)

const NU = collect(2000.0:0.5:2700.0)
grid = wavenumber_grid(2000.0, 2700.0, 0.5)

k_co2 = co2_continuum(grid, VMR_CO2, P_HPA, T_K)
k_n2  = n2_continuum( grid, VMR_N2,  P_HPA, T_K)
k_o2  = o2_continuum( grid, VMR_O2,  P_HPA, T_K)

i_peak = argmin(abs.(NU .- 2331.0))
@printf "At ν = %.1f cm⁻¹ (N2 fundamental peak), T=%.1f K, p=%.1f hPa:\n" NU[i_peak] T_K P_HPA
@printf "  k_co2 = %.3e /cm\n" k_co2[i_peak]
@printf "  k_n2  = %.3e /cm\n" k_n2[i_peak]
@printf "  k_o2  = %.3e /cm\n" k_o2[i_peak]

open(OUT_FILE, "w") do f
    write(f, "nu_cm1,k_co2,k_n2,k_o2\n")
    for i in eachindex(NU)
        @printf f "%.4f,%.6e,%.6e,%.6e\n" NU[i] k_co2[i] k_n2[i] k_o2[i]
    end
end
println("\nSaved $(length(NU)) channels → $OUT_FILE")
