"""
Diff three BT spectra over 645–2760 cm⁻¹:
  Julia (new, with HITRAN CIA: H2O + CO2 + N2 + O2)
  Julia (old, H2O MT-CKD + toy CO2 fit)            — preserved as julia_bt_cont_pre_cia.csv
  ARTS  (H2O-only continuum)                       — arts_bt_iasi_cont.csv

Outputs:
  data/julia_cia_diff_summary.txt
  data/julia_cia_vs_arts.png
"""

using DelimitedFiles
using Statistics
using Plots
using Printf

function read_bt(path)
    data, _ = readdlm(path, ',', Float64, '\n'; header=true)
    return data[:, 1], data[:, 2]  # ν, BT
end

ν_new, bt_new = read_bt("data/julia_bt_cont.csv")
ν_old, bt_old = read_bt("data/julia_bt_cont_pre_cia.csv")
ν_arts, bt_arts = read_bt("data/arts_bt_iasi_cont.csv")

@assert ν_new == ν_old "new and old Julia grids differ"
@assert ν_new == ν_arts "Julia and ARTS grids differ"

Δ_cia      = bt_new  .- bt_old           # CIA addition impact
Δ_vs_arts  = bt_new  .- bt_arts          # current residual
Δ_old_arts = bt_old  .- bt_arts          # previous residual

function stats(label, Δ)
    @printf "%-30s bias=%+.4f K  RMS=%.4f K  max|Δ|=%.4f K @ ν=%.1f\n" label mean(Δ) sqrt(mean(Δ.^2)) maximum(abs.(Δ)) ν_new[argmax(abs.(Δ))]
end

println("Julia (new, HITRAN CIA) vs Julia (old, H2O+toy CO2):")
stats("  ΔBT(new − old)", Δ_cia)
println()
println("Julia (new) vs ARTS (H2O-only continuum):")
stats("  ΔBT(new Julia − ARTS)", Δ_vs_arts)
println()
println("Julia (old) vs ARTS (H2O-only continuum)  [for reference]:")
stats("  ΔBT(old Julia − ARTS)", Δ_old_arts)

# Plot the CIA addition impact and the new vs ARTS residual
plt = plot(ν_new, Δ_cia,
           label="ΔBT (new − old Julia): impact of HITRAN CIA",
           lw=1.0, color=:steelblue,
           xlabel="ν (cm⁻¹)", ylabel="ΔBT (K)",
           title="Effect of HITRAN CIA addition + new residual vs ARTS",
           legend=:bottomleft, size=(1100, 550), grid=true)
plot!(plt, ν_new, Δ_vs_arts,
      label="ΔBT (new Julia − ARTS, H2O-only)", lw=1.0, color=:tomato)
hline!(plt, [0.0], color=:black, lw=0.5, ls=:dash, label=false)
savefig(plt, "data/julia_cia_vs_arts.png")
println("\nWrote data/julia_cia_vs_arts.png")

open("data/julia_cia_diff_summary.txt", "w") do f
    redirect_stdout(f) do
        println("Julia (new, HITRAN CIA) vs Julia (old, H2O+toy CO2):")
        stats("  ΔBT(new − old)", Δ_cia)
        println()
        println("Julia (new) vs ARTS (H2O-only continuum):")
        stats("  ΔBT(new Julia − ARTS)", Δ_vs_arts)
        println()
        println("Julia (old) vs ARTS (H2O-only continuum)  [for reference]:")
        stats("  ΔBT(old Julia − ARTS)", Δ_old_arts)
    end
end
