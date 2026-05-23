"""
Compare ARTS vs Julia CIA absorption coefficients on the IASI 4-µm band.
Reads:
  data/arts_cia_sigma_dump.csv
  data/julia_cia_sigma_dump.csv
Reports per-species bias/RMS/peak, prints a per-block summation marker for N2,
and writes a 4-panel plot to data/cia_sigma_diff.png.
"""

using DelimitedFiles
using Statistics
using Plots
using Printf

function read_dump(path)
    data, _ = readdlm(path, ',', Float64, '\n'; header=true)
    return data[:, 1], data[:, 2], data[:, 3], data[:, 4]
end

ν, k_co2_a, k_n2_a, k_o2_a = read_dump("data/arts_cia_sigma_dump.csv")
_, k_co2_j, k_n2_j, k_o2_j = read_dump("data/julia_cia_sigma_dump.csv")

function stats(label, a, j)
    Δ = a .- j
    rel = (a .- j) ./ max.(j, 1e-30)
    nz  = j .> 1e-12
    if !any(nz)
        @printf "%-10s  (Julia all-zero in this region — no stats)\n" label
        return
    end
    bias = mean(Δ[nz])
    rms  = sqrt(mean(Δ[nz].^2))
    rmax = maximum(abs.(rel[nz]))
    @printf "%-10s  ARTS-Julia bias=%+.3e  RMS=%.3e  max|rel|=%.3f  (in non-zero Julia region)\n" label bias rms rmax
end

println("=" ^ 80)
println("Per-species ARTS vs Julia σ_CIA, 2000–2700 cm⁻¹, T=288 K, p=1013 hPa")
println("=" ^ 80)
stats("CO2", k_co2_a, k_co2_j)
stats("N2",  k_n2_a,  k_n2_j)
stats("O2",  k_o2_a,  k_o2_j)

println()
println("Ratios ARTS/Julia at sampled points (N2 only, where Julia ≥ 1e-10):")
sample_ν = [2100.0, 2200.0, 2331.0, 2400.0, 2500.0, 2600.0]
@printf "%-10s %-12s %-12s %-8s\n" "ν (cm⁻¹)" "ARTS" "Julia" "A/J"
for νs in sample_ν
    i = argmin(abs.(ν .- νs))
    if k_n2_j[i] > 1e-12
        @printf "%-10.1f %-12.3e %-12.3e %-8.3f\n" ν[i] k_n2_a[i] k_n2_j[i] (k_n2_a[i] / k_n2_j[i])
    else
        @printf "%-10.1f %-12.3e %-12.3e —\n" ν[i] k_n2_a[i] k_n2_j[i]
    end
end

# Detection of "double-counted" region: 1999.9 ≤ ν ≤ 2697.9 (block-3 ∩ block-4)
mask_overlap = (ν .≥ 1999.9) .& (ν .≤ 2697.9)
mean_ratio_overlap = mean((k_n2_a[mask_overlap] .+ 1e-30) ./ (k_n2_j[mask_overlap] .+ 1e-30))
mask_outside = .!mask_overlap .& (k_n2_j .> 1e-10)
mean_ratio_outside = isempty(k_n2_j[mask_outside]) ? NaN :
                     mean((k_n2_a[mask_outside] .+ 1e-30) ./ (k_n2_j[mask_outside] .+ 1e-30))
@printf "\nMean ARTS/Julia ratio inside  1999.9-2697.9 cm⁻¹ overlap: %.3f\n" mean_ratio_overlap
@printf "Mean ARTS/Julia ratio outside overlap (Julia ≥ 1e-10):    %.3f\n" mean_ratio_outside

plt = plot(layout=(2, 2), size=(1200, 800))
plot!(plt[1], ν, k_co2_a, label="ARTS",  color=:steelblue, lw=1.0)
plot!(plt[1], ν, k_co2_j, label="Julia", color=:tomato,    lw=1.0, ls=:dash)
plot!(plt[1], title="CO2 k (/cm)", xlabel="ν (cm⁻¹)", legend=:topright)
plot!(plt[2], ν, k_n2_a, label="ARTS",  color=:steelblue, lw=1.0)
plot!(plt[2], ν, k_n2_j, label="Julia", color=:tomato,    lw=1.0, ls=:dash)
plot!(plt[2], title="N2 k (/cm)  — note shaded overlap region", xlabel="ν (cm⁻¹)")
vspan!(plt[2], [1999.9, 2697.9], alpha=0.1, color=:gray, label="block 3∩4")
plot!(plt[3], ν, k_o2_a, label="ARTS",  color=:steelblue, lw=1.0)
plot!(plt[3], ν, k_o2_j, label="Julia", color=:tomato,    lw=1.0, ls=:dash)
plot!(plt[3], title="O2 k (/cm)", xlabel="ν (cm⁻¹)")
plot!(plt[4], ν, k_n2_a ./ max.(k_n2_j, 1e-30), label="N2 ARTS/Julia",
      color=:darkgreen, lw=1.2, xlabel="ν (cm⁻¹)", ylabel="ratio",
      title="N2 ratio (= 2 implies block-summing)", ylim=(0, 5))
hline!(plt[4], [1.0], color=:black, lw=0.5, ls=:dot, label=false)
hline!(plt[4], [2.0], color=:red,   lw=0.5, ls=:dash, label="ratio=2")
vspan!(plt[4], [1999.9, 2697.9], alpha=0.1, color=:gray, label=false)

savefig(plt, "data/cia_sigma_diff.png")
println("\nWrote data/cia_sigma_diff.png")
