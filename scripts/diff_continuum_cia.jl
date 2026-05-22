"""
Continuum comparison with HITRAN CIA on both sides.

Loads:
  data/arts_bt_iasi_cont_pre_cia.csv   — ARTS, H2O-only (old)
  data/arts_bt_iasi_cont.csv           — ARTS, H2O + CIA (new)
  data/julia_bt_cont_pre_cia.csv       — Julia, H2O + toy CO2 (old)
  data/julia_bt_cont.csv               — Julia, H2O + HITRAN CIA (new)

Reports:
  ΔBT(new Julia − new ARTS)            — matched-config residual (headline)
  ΔBT(new − old) per side              — CIA impact on each model
  ΔBT(old Julia − old ARTS)            — previous validation result (for context)

Outputs:
  data/continuum_cia_diff_summary.txt
  data/continuum_cia_diff.png
"""

using DelimitedFiles
using Statistics
using Plots
using Printf

function read_bt(path)
    data, _ = readdlm(path, ',', Float64, '\n'; header=true)
    return data[:, 1], data[:, 2]
end

ν, bt_arts_old   = read_bt("data/arts_bt_iasi_cont_pre_cia.csv")
_, bt_arts_new   = read_bt("data/arts_bt_iasi_cont.csv")
_, bt_julia_old  = read_bt("data/julia_bt_cont_pre_cia.csv")
_, bt_julia_new  = read_bt("data/julia_bt_cont.csv")

Δ_julia_cia = bt_julia_new .- bt_julia_old
Δ_arts_cia  = bt_arts_new  .- bt_arts_old
Δ_matched   = bt_julia_new .- bt_arts_new
Δ_old       = bt_julia_old .- bt_arts_old

function stats(label, Δ, ν)
    @printf "%-40s bias=%+.4f K  RMS=%.4f K  max|Δ|=%.4f K @ ν=%.1f\n" label mean(Δ) sqrt(mean(Δ.^2)) maximum(abs.(Δ)) ν[argmax(abs.(Δ))]
end

println("=" ^ 90)
println("HEADLINE: matched-config residual (both sides with H2O + CIA)")
println("=" ^ 90)
stats("  Julia − ARTS (new, matched)", Δ_matched, ν)
println()
println("Previous baseline (H2O-only on ARTS side, toy CO2 on Julia side):")
stats("  Julia − ARTS (old, mismatched)", Δ_old, ν)
println()
println("=" ^ 90)
println("CIA impact per side (new − old):")
println("=" ^ 90)
stats("  ARTS:  +HITRAN CIA vs H2O-only", Δ_arts_cia, ν)
stats("  Julia: HITRAN CIA vs (H2O + toy CO2)", Δ_julia_cia, ν)

# Plot
plt = plot(ν, Δ_matched, label="Julia − ARTS (matched: both H2O+CIA)", lw=1.0, color=:steelblue,
           xlabel="ν (cm⁻¹)", ylabel="ΔBT (K)",
           title="ARTS continuum comparison with HITRAN CIA on both sides",
           legend=:bottomleft, size=(1100, 600), grid=true)
plot!(plt, ν, Δ_old, label="Julia − ARTS (old: H2O-only on ARTS)", lw=1.0, color=:tomato, ls=:dash, alpha=0.6)
hline!(plt, [0.0], color=:black, lw=0.5, ls=:dot, label=false)
savefig(plt, "data/continuum_cia_diff.png")
println("\nWrote data/continuum_cia_diff.png")

open("data/continuum_cia_diff_summary.txt", "w") do f
    redirect_stdout(f) do
        println("Matched config (Julia HITRAN CIA vs ARTS HITRAN CIA):")
        stats("  Julia − ARTS (matched)", Δ_matched, ν)
        println()
        println("Old baseline (mismatched continua):")
        stats("  Julia − ARTS (old)", Δ_old, ν)
        println()
        println("CIA impact:")
        stats("  ARTS:  +CIA vs H2O-only", Δ_arts_cia, ν)
        stats("  Julia: HITRAN CIA vs toy CO2", Δ_julia_cia, ν)
    end
end
