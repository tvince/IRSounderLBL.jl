using RadiativeTransfer
using Printf
using KernelAbstractions

# ── Original (baseline) kernel — per-line sqrt inside the loop ────────────────
@kernel function voigt_kernel_orig!(σ, ν_grid, lines_ν0, lines_S, lines_γL, lines_γD, cutoff)
    i = @index(Global, Linear)
    ν       = ν_grid[i]
    n_lines = length(lines_ν0)
    j_lo = RadiativeTransfer._lower_bound(lines_ν0, ν - cutoff, n_lines)
    j_hi = RadiativeTransfer._upper_bound(lines_ν0, ν + cutoff, n_lines)
    acc = 0.0
    ln2 = log(2.0)
    for j in j_lo:j_hi
        Δν = ν - lines_ν0[j]
        gD = lines_γD[j]
        f  = sqrt(ln2) / gD
        x  = Δν * f
        y  = lines_γL[j] * f
        H  = RadiativeTransfer.humlicek_voigt(x, y)
        acc += lines_S[j] * f * H / sqrt(π)
    end
    σ[i] = max(acc, 0.0)
end

function compute_voigt_orig(ν_grid, linelist, T, p_atm; cutoff=25.0)
    n_ν = ν_grid.n
    n_L = length(linelist.lines)
    ν0  = Vector{Float64}(undef, n_L)
    S   = Vector{Float64}(undef, n_L)
    γL  = Vector{Float64}(undef, n_L)
    γD  = Vector{Float64}(undef, n_L)
    for (j, line) in enumerate(linelist.lines)
        ν0[j] = RadiativeTransfer.pressure_shift(line, p_atm)
        S[j]  = RadiativeTransfer.temperature_scaled_intensity(line, T)
        gl, gd = RadiativeTransfer.pressure_broadened_width(line, p_atm, T)
        γL[j] = gl
        γD[j] = max(gd, 1e-10)
    end
    σ   = KernelAbstractions.zeros(CPU(), Float64, n_ν)
    ν_d = KernelAbstractions.allocate(CPU(), Float64, n_ν); copyto!(ν_d, ν_grid.ν)
    ν0_d = KernelAbstractions.allocate(CPU(), Float64, n_L); copyto!(ν0_d, ν0)
    S_d  = KernelAbstractions.allocate(CPU(), Float64, n_L); copyto!(S_d, S)
    γL_d = KernelAbstractions.allocate(CPU(), Float64, n_L); copyto!(γL_d, γL)
    γD_d = KernelAbstractions.allocate(CPU(), Float64, n_L); copyto!(γD_d, γD)
    k! = voigt_kernel_orig!(CPU(), 256)
    k!(σ, ν_d, ν0_d, S_d, γL_d, γD_d, cutoff; ndrange=n_ν)
    KernelAbstractions.synchronize(CPU())
    return Array(σ)
end

# ── Test setup ────────────────────────────────────────────────────────────────
# CO2, 100 cm⁻¹ chunk centred near the 15 µm band, mid-troposphere
ll   = load_hitran_par("data/co2_645_2760.par"; ν_min=675.0, ν_max=825.0)
grid = wavenumber_grid(700.0, 800.0, 0.125)   # Δν_hi = IASI 0.25 cm⁻¹ / HRF 2
T     = 250.0
p_atm = 0.5   # ~500 hPa

@printf("Lines in chunk : %d\n", length(ll.lines))
@printf("Grid points    : %d\n", grid.n)

# Warm-up both implementations
σ_new = compute_voigt_cross_sections(grid, ll, T, p_atm; cutoff=25.0)
σ_old = compute_voigt_orig(grid, ll, T, p_atm; cutoff=25.0)

# Correctness check
max_diff = maximum(abs, σ_new .- σ_old)
rel_diff = max_diff / maximum(σ_old)
@printf("Max abs diff   : %.3e  (rel: %.3e)\n", max_diff, rel_diff)

# Timed runs
N = 20
t_old = @elapsed for _ in 1:N; compute_voigt_orig(grid, ll, T, p_atm; cutoff=25.0); end
t_new = @elapsed for _ in 1:N; compute_voigt_cross_sections(grid, ll, T, p_atm; cutoff=25.0); end

ms_old = t_old / N * 1000
ms_new = t_new / N * 1000
@printf("\nBaseline (original) : %6.2f ms/call\n", ms_old)
@printf("Optimized           : %6.2f ms/call\n", ms_new)
@printf("Speedup             : %.2fx  (%.0f%% faster)\n", ms_old/ms_new, (1 - ms_new/ms_old)*100)
@printf("Time saved/call     : %.2f ms\n", ms_old - ms_new)
