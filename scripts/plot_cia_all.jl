"""
Compare CO2, N2, O2 CIA contributions across the IASI band at
T=296 K, p=1013.25 hPa, dry-air composition.

Output: data/cia_co2_n2_o2.png
"""

using IRSounderLBL
using Plots
using Printf

const NU_MIN = 645.0
const NU_MAX = 2760.0
const DNU    = 0.25
const T      = 296.0
const P_HPA  = 1013.25

ν     = wavenumber_grid(NU_MIN, NU_MAX, DNU)
k_co2 = co2_continuum(ν, 4.15e-4, P_HPA, T)
k_n2  = n2_continuum(ν,  0.78084, P_HPA, T)
k_o2  = o2_continuum(ν,  0.20946, P_HPA, T)

ε = 1e-15
plt = plot(ν.ν, k_n2  .+ ε, yscale=:log10, label="N₂–N₂", lw=1.5, color=:steelblue,
           xlabel="ν (cm⁻¹)",
           ylabel="k (cm⁻¹)  [log scale, ε=1e-15]",
           title=@sprintf("CIA contributions @ T=%.0f K, p=%.0f hPa (dry-air composition)", T, P_HPA),
           legend=:bottomright, size=(1100, 500), grid=true)
plot!(plt, ν.ν, k_o2  .+ ε, label="O₂–O₂", lw=1.5, color=:seagreen)
plot!(plt, ν.ν, k_co2 .+ ε, label="CO₂–CO₂", lw=1.5, color=:tomato)
ylims!(plt, 1e-15, 1e-3)

mkpath("data")
savefig(plt, "data/cia_co2_n2_o2.png")
println("Wrote data/cia_co2_n2_o2.png")

println()
@printf "Peak k values (T=%.0f K, p=%.0f hPa):\n"  T P_HPA
@printf "  N₂–N₂ : peak %.3e at ν=%.1f cm⁻¹\n"  maximum(k_n2)  ν.ν[argmax(k_n2)]
@printf "  O₂–O₂ : peak %.3e at ν=%.1f cm⁻¹\n"  maximum(k_o2)  ν.ν[argmax(k_o2)]
@printf "  CO₂–CO₂: peak %.3e at ν=%.1f cm⁻¹\n" maximum(k_co2) ν.ν[argmax(k_co2)]
@printf "\nIntegrated absorption ∫k dν over 645-2760 cm⁻¹:\n"
@printf "  N₂–N₂ : %.3e cm⁻¹·(cm⁻¹)\n"  sum(k_n2)  * DNU
@printf "  O₂–O₂ : %.3e\n"               sum(k_o2)  * DNU
@printf "  CO₂–CO₂: %.3e\n"              sum(k_co2) * DNU
