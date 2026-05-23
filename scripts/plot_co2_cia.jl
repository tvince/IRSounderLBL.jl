"""
Compare the new HITRAN CO2-CO2 CIA cross sections to the old toy fit
(piecewise Gaussian/step) over the IASI range at T=296 K, p=1013.25 hPa.

Output: data/co2_cia_old_vs_new.png
"""

using IRSounderLBL
using Plots
using Printf

const NU_MIN = 645.0
const NU_MAX = 2760.0
const DNU    = 0.25
const T      = 296.0
const P_HPA  = 1013.25
const VMR    = 4.15e-4

# Old toy fit (verbatim from pre-change continuum.jl)
function toy_co2_continuum(ν_grid, vmr_co2, p_hPa, T)
    KB = 1.380649e-23
    n_co2 = vmr_co2 * p_hPa * 100.0 / (KB * T) * 1e-6
    n_air = p_hPa            * 100.0 / (KB * T) * 1e-6
    function σ(ν)
        if 1200.0 ≤ ν ≤ 1500.0
            return 1.0e-47 * exp(-((ν - 1350.0) / 150.0)^2)
        elseif 2300.0 ≤ ν ≤ 2400.0
            return 5.0e-48
        else
            return 0.0
        end
    end
    return [σ(ν) * n_co2 * n_air * (296.0 / T)^0.5 for ν in ν_grid.ν]
end

ν   = wavenumber_grid(NU_MIN, NU_MAX, DNU)
new = co2_continuum(ν, VMR, P_HPA, T)
old = toy_co2_continuum(ν, VMR, P_HPA, T)

# Plot in log10 of k+ε for visibility; mark zero regions
ε = 1e-20

plt = plot(ν.ν, new .+ ε, yscale=:log10, label="HITRAN CIA (new)",
           lw=1.5, color=:steelblue,
           xlabel="ν (cm⁻¹)",
           ylabel="k_CO₂ (cm⁻¹)  [log scale, ε=1e-20 floor]",
           title=@sprintf("CO₂ continuum: toy fit vs HITRAN CIA @ T=%.0f K, p=%.0f hPa, VMR=%.0e",
                          T, P_HPA, VMR),
           legend=:bottomright, size=(1100, 500), grid=true)
plot!(plt, ν.ν, old .+ ε, label="toy fit (old)", lw=1.5, color=:tomato, ls=:dash)
ylims!(plt, 1e-20, 1e-7)

mkpath("data")
savefig(plt, "data/co2_cia_old_vs_new.png")
println("Wrote data/co2_cia_old_vs_new.png")

println()
println("Peak comparison (T=$T K, p=$P_HPA hPa, VMR=$VMR):")
@printf "  Old toy: peak %.3e at ν=%.1f\n"  maximum(old) ν.ν[argmax(old)]
@printf "  HITRAN : peak %.3e at ν=%.1f\n"  maximum(new) ν.ν[argmax(new)]
@printf "  ∫k_old  dν = %.3e\n"  sum(old) * DNU
@printf "  ∫k_new  dν = %.3e\n"  sum(new) * DNU
