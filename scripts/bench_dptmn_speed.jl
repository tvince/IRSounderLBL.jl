#!/usr/bin/env julia
#=
Speed of the per-layer cross-section build with vs without DPTMN line rejection,
in the same 14 µm region as the LBLRTM benchmark (710-720 cm⁻¹, 49 layers, CUTOFF
25, FullFaddeeva, 0.001 grid). LBLRTM v12.17 reference there = 1.45 s (1 core).

Run:  julia --project -t auto scripts/bench_dptmn_speed.jl
=#
using IRSounderLBL
const M = IRSounderLBL
using Printf, Statistics

const CUTOFF = 25.0
const LBLRTM_S = 1.45                      # memory: 710-720, 49 layers, cont OFF
ν1, ν2, Δhi = 710.0, 720.0, 0.001

ν_hi  = M.wavenumber_grid(ν1, ν2, Δhi)
ll    = M.load_linelist(joinpath("data","co2_645_2760"), 1:3;
                        ν_min = ν1 - CUTOFF, ν_max = ν2 + CUTOFF)
prof  = M.afgl_us_standard_50lev()
layers = M.layer_properties(prof)
nlay  = length(layers.p_mid)
Nair  = 2.1209e22
@printf("region %.0f-%.0f  grid=%d  lines=%d  layers=%d  threads=%d\n",
        ν1, ν2, ν_hi.n, length(ll.lines), nlay, Threads.nthreads())

# Full per-layer cross-section build for the CO2 column, optionally rejecting.
function build_tau(dptmn)
    τ = zeros(ν_hi.n, nlay); kept = 0
    for k in 1:nlay
        vmr = layers.vmr_cg[M.CO2][k]; vmr == 0 && continue
        T = layers.T_cg[M.CO2][k]; p = layers.p_cg[M.CO2][k]/1013.25
        coef = vmr * layers.Δp[k] * Nair
        lu = M._reject_weak_lines(ll, T, p, coef; dptmn=dptmn)
        kept += length(lu.lines)
        σ = M.compute_voigt_cross_sections(ν_hi, lu, T, p; cutoff=CUTOFF, method=M.FullFaddeeva)
        τ[:,k] .+= σ .* coef
    end
    τ, kept
end

build_tau(0.0); build_tau(1e-6)            # compile
bench(d, n=5) = minimum(@elapsed(build_tau(d)) for _ in 1:n)

t_full = bench(0.0); _, kfull = build_tau(0.0)
t_rej  = bench(1e-6); _, krej  = build_tau(1e-6)
@printf("\n %-10s %-10s %-9s %-12s %-10s\n", "dptmn", "time (s)", "×LBLRTM", "line-evals", "speedup")
@printf(" %-10s %8.3f  %7.2f  %10d   %s\n", "0 (full)", t_full, t_full/LBLRTM_S, kfull, "1.00×")
@printf(" %-10s %8.3f  %7.2f  %10d   %.2f×\n", "1e-6",   t_rej,  t_rej/LBLRTM_S,  krej,
        t_full/t_rej)
@printf("\n line-evals reduced %.1f×; LBLRTM ref = %.2f s\n",
        kfull/krej, LBLRTM_S)
println("Done.")
