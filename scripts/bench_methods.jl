using IRSounderLBL
using KernelAbstractions
using Metal
using Printf
using Statistics

hitran_file = "data/co2_645_2760.par"
ll   = load_hitran_par(hitran_file; ν_min=675.0, ν_max=825.0)
grid = wavenumber_grid(700.0, 800.0, 0.125)
T_atm, p_atm = 250.0, 0.5

nthreads = Threads.nthreads()
@printf("Lines: %d   Grid points: %d   Julia threads: %d\n\n",
        length(ll.lines), grid.n, nthreads)

N = 50

# ── CPU Weideman (KernelAbstractions CPU backend, uses all Julia threads) ─────
compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=Weideman)  # warm-up
times_w = [(@elapsed compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=Weideman)) for _ in 1:N]

# ── CPU FullFaddeeva (Threads.@threads) ───────────────────────────────────────
compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=FullFaddeeva)  # warm-up
times_f = [(@elapsed compute_voigt_cross_sections(grid, ll, T_atm, p_atm; method=FullFaddeeva)) for _ in 1:N]

# ── Metal GPU Weideman (Float32) ──────────────────────────────────────────────
metal_ok = false
times_m  = Float64[]
try
    compute_voigt_cross_sections(grid, ll, T_atm, p_atm; backend=Metal.MetalBackend())  # warm-up
    compute_voigt_cross_sections(grid, ll, T_atm, p_atm; backend=Metal.MetalBackend())
    times_m = [(@elapsed compute_voigt_cross_sections(grid, ll, T_atm, p_atm; backend=Metal.MetalBackend())) for _ in 1:N]
    metal_ok = true
catch e
    println("Metal GPU: $(typeof(e))")
end

# ── Results ───────────────────────────────────────────────────────────────────
med_w = median(times_w) * 1e3
med_f = median(times_f) * 1e3

println("=" ^ 62)
@printf("%-32s  %8s  %8s  %10s\n", "Method", "median ms", "min ms", "vs Faddeeva")
println("-" ^ 62)
@printf("%-32s  %8.3f  %8.3f  %10s\n",
        "CPU Weideman ($nthreads threads, F64)",
        med_w, minimum(times_w)*1e3, @sprintf("%.2fx faster", med_f/med_w))
@printf("%-32s  %8.3f  %8.3f  %10s\n",
        "CPU FullFaddeeva ($nthreads threads)",
        med_f, minimum(times_f)*1e3, "reference")
if metal_ok
    med_m = median(times_m) * 1e3
    @printf("%-32s  %8.3f  %8.3f  %10s\n",
            "Metal GPU Weideman (F32)",
            med_m, minimum(times_m)*1e3, @sprintf("%.2fx faster", med_f/med_m))
    println()
    println("Note: Metal result is Float32 (M1 GPU has no Float64).")
    println("      Includes CPU↔GPU transfer for $(grid.n) grid pts + $(length(ll.lines)) lines.")
else
    @printf("%-32s  %8s  %8s  %10s\n", "Metal GPU Weideman (F32)", "N/A", "N/A",
            "compiler crash")
    println()
    println("Metal GPU: shader compiler daemon (AGXMetalG13X) crashes on this")
    println("kernel — dynamic binary search + 24-iter inner loop is too complex")
    println("for the current Metal.jl version. Float32 support is wired up and")
    println("the type compilation succeeds; only the final GPU link step fails.")
end
println("=" ^ 62)
