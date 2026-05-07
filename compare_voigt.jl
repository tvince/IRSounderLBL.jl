using RadiativeTransfer
using Printf
using Statistics

# ── 1. H-function accuracy: Humlicek and PseudoVoigt vs FullFaddeeva ─────────

println("=" ^ 65)
println("  Voigt H-function accuracy  (reference = FullFaddeeva / erfcx)")
println("=" ^ 65)

# Sample over a range of (x, y) pairs covering all Humlicek regions
xs = vcat(range(-30, 30, length=200), range(-5, 5, length=200))
ys = vcat(fill(1e-4, 200), fill(0.05, 100), fill(1.0, 100), fill(5.0, 200))

errs_hum = Float64[]
errs_pv  = Float64[]

for (x, y) in zip(xs, ys)
    ref  = faddeeva_voigt(x, y)
    ref == 0.0 && continue
    push!(errs_hum, abs(humlicek_voigt(x, y) - ref) / ref)
    # PseudoVoigt doesn't have an H-function equivalent; compare at profile level
end

# Profile-level accuracy: scan a single line with various γ_L/γ_D ratios
ν0 = 1000.0
ν_pts = range(999.5, 1000.5, length=1000)

function profile_errors(γ_L, γ_D)
    ref_vals = [voigt_profile(ν, ν0, γ_L, γ_D, FullFaddeeva) for ν in ν_pts]
    hum_vals = [voigt_profile(ν, ν0, γ_L, γ_D, Humlicek)     for ν in ν_pts]
    pv_vals  = [voigt_profile(ν, ν0, γ_L, γ_D, PseudoVoigt)  for ν in ν_pts]
    mask = ref_vals .> 1e-6 .* maximum(ref_vals)
    rel_hum = maximum(abs.(hum_vals[mask] .- ref_vals[mask]) ./ ref_vals[mask])
    rel_pv  = maximum(abs.(pv_vals[mask]  .- ref_vals[mask]) ./ ref_vals[mask])
    return rel_hum, rel_pv
end

cases = [
    # Realistic atmospheric CO2 conditions (surface → stratosphere)
    ("Surface (1 atm,296K):  γ_L=0.06, γ_D=6.2e-4",  0.060,  6.2e-4),
    ("100 hPa (0.1atm,230K): γ_L=7e-3, γ_D=5.5e-4",  7e-3,   5.5e-4),
    ("10 hPa  (0.01atm,200K):γ_L=8e-4, γ_D=5.1e-4",  8e-4,   5.1e-4),
    # Academic extreme — sub-hPa or light molecules
    ("Extreme near-Gaussian: γ_L=0.001, γ_D=0.05  ",  0.001,  0.050),
]

@printf("\n%-42s  %12s  %12s\n", "Case", "Humlicek", "PseudoVoigt")
@printf("%-42s  %12s  %12s\n", "-"^42, "-"^12, "-"^12)
for (label, γL, γD) in cases
    rh, rp = profile_errors(γL, γD)
    @printf("%-42s  %12.2e  %12.2e\n", label, rh, rp)
end

println("Note: The extreme near-Gaussian case (y≈0.017) is unphysical for CO2/H2O at")
println("      atmospheric pressures (y≥1 everywhere). It appears only for very light")
println("      molecules at sub-hPa pressures and reveals a known Humlicek limitation")
println("      in the Region III/IV boundary zone (s≈3–5, y<0.05).")

# H-function summary
println()
@printf("Humlicek H-function max relative error : %.2e\n", maximum(errs_hum))
@printf("Humlicek H-function mean relative error: %.2e\n", mean(errs_hum))

# ── 2. Speed: scalar voigt_profile ───────────────────────────────────────────

println()
println("=" ^ 65)
println("  Scalar voigt_profile speed  (ν₀=1000, γ_L=0.02, γ_D=0.02)")
println("=" ^ 65)

γL, γD = 0.02, 0.02
ν_bench = collect(range(990.0, 1010.0, length=10_000))
N_scalar = 500

t_hum = @elapsed for _ in 1:N_scalar
    for ν in ν_bench
        voigt_profile(ν, ν0, γL, γD, Humlicek)
    end
end

t_pv = @elapsed for _ in 1:N_scalar
    for ν in ν_bench
        voigt_profile(ν, ν0, γL, γD, PseudoVoigt)
    end
end

t_fad = @elapsed for _ in 1:N_scalar
    for ν in ν_bench
        voigt_profile(ν, ν0, γL, γD, FullFaddeeva)
    end
end

n_evals = N_scalar * length(ν_bench)
@printf("\n%s  %10s  %8s\n", "Method        ", "  ns/eval", "vs Faddeeva")
@printf("%s  %10s  %8s\n", "-"^14, "-"^10, "-"^10)
@printf("%-14s  %10.1f  %8.2fx\n", "FullFaddeeva",  t_fad/n_evals*1e9, 1.0)
@printf("%-14s  %10.1f  %8.2fx\n", "Humlicek",      t_hum/n_evals*1e9, t_fad/t_hum)
@printf("%-14s  %10.1f  %8.2fx\n", "PseudoVoigt",   t_pv/n_evals*1e9,  t_fad/t_pv)

# ── 3. Speed: full cross-section spectrum (requires HITRAN data) ──────────────

hitran_file = "data/co2_645_2760.par"
if isfile(hitran_file)
    println()
    println("=" ^ 65)
    println("  Cross-section speed  (CO2, 700–800 cm⁻¹, Δν=0.125, T=250K)")
    println("=" ^ 65)

    ll   = load_hitran_par(hitran_file; ν_min=675.0, ν_max=825.0)
    grid = wavenumber_grid(700.0, 800.0, 0.125)
    T_atm, p_atm = 250.0, 0.5

    @printf("\nLines: %d   Grid points: %d\n", length(ll.lines), grid.n)

    # Warm-up
    compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=Humlicek)
    compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=PseudoVoigt)
    compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=FullFaddeeva)

    N_cs = 10
    t_cs_hum = @elapsed for _ in 1:N_cs
        compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=Humlicek)
    end
    t_cs_pv  = @elapsed for _ in 1:N_cs
        compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=PseudoVoigt)
    end
    t_cs_fad = @elapsed for _ in 1:N_cs
        compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=FullFaddeeva)
    end

    σ_ref = compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=FullFaddeeva)
    σ_hum = compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=Humlicek)
    σ_pv  = compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=PseudoVoigt)

    mask = σ_ref .> 1e-30
    err_hum = maximum(abs.(σ_hum[mask] .- σ_ref[mask]) ./ σ_ref[mask])
    err_pv  = maximum(abs.(σ_pv[mask]  .- σ_ref[mask]) ./ σ_ref[mask])

    @printf("\n%-14s  %10s  %10s  %10s\n", "Method", "ms/call", "vs Faddeeva", "max rel err")
    @printf("%-14s  %10s  %10s  %10s\n", "-"^14, "-"^10, "-"^10, "-"^10)
    @printf("%-14s  %10.2f  %10s  %10s\n",  "FullFaddeeva", t_cs_fad/N_cs*1e3, "1.00x", "reference")
    @printf("%-14s  %10.2f  %10.2fx  %10.2e\n", "Humlicek",    t_cs_hum/N_cs*1e3, t_cs_fad/t_cs_hum, err_hum)
    @printf("%-14s  %10.2f  %10.2fx  %10.2e\n", "PseudoVoigt", t_cs_pv/N_cs*1e3,  t_cs_fad/t_cs_pv,  err_pv)
else
    println("\nSkipping cross-section benchmark — $hitran_file not found.")
    println("Run fetch_hitran_api(...) first to download CO2 line data.")
end

println()
