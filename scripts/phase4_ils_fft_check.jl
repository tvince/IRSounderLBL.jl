# Phase 4 — validate apply_ils_fft ≡ apply_ils and benchmark the speedup.
using IRSounderLBL
using Printf

# Correctness on a small grid with a structured spectrum.
let
    g = wavenumber_grid(699.0, 705.0, 0.002)
    δν, kern = ils_kernel(g.Δν, 2.0, 0.5; apodization=:gaussian)
    spec = [1.0 + 0.5sin(3.0*ν) + (abs(ν-702.0) < 0.05 ? -0.8 : 0.0) for ν in g.ν]
    direct = apply_ils(g, spec, δν, kern)
    fft    = apply_ils_fft(g, spec, δν, kern)
    @printf("small grid (n=%d, n_ils=%d): max|fft-direct|=%.3e\n",
            g.n, length(kern), maximum(abs.(fft .- direct)))
end

# Production-resolution timing on a ~150 cm⁻¹ band at internal_dnu = 0.001.
let
    g = wavenumber_grid(700.0, 850.0, 0.001)
    δν, kern = ils_kernel(g.Δν, 2.0, 0.5; apodization=:gaussian)
    spec = [1.0 + 0.3sin(0.7*ν) for ν in g.ν]
    @printf("\nproduction grid: n_ν=%d, n_ils=%d\n", g.n, length(kern))

    d = apply_ils(g, spec, δν, kern)
    conv = ILSConvolver(g, δν, kern)
    f = ils_apply!(conv, similar(spec), spec)
    @printf("max|fft-direct|=%.3e (relative %.2e)\n",
            maximum(abs.(f .- d)), maximum(abs.(f .- d))/maximum(abs.(d)))

    # Time both (the convolver is prebuilt, mirroring per-column reuse in the Jacobian).
    apply_ils(g, spec, δν, kern); ils_apply!(conv, f, spec)   # warm up
    t_dir = @elapsed for _ in 1:2;  apply_ils(g, spec, δν, kern); end
    t_fft = @elapsed for _ in 1:2;  ils_apply!(conv, f, spec); end
    @printf("direct: %.3f s/call   fft: %.4f s/call   speedup ×%.0f\n",
            t_dir/2, t_fft/2, (t_dir/2)/(t_fft/2))
end

# End-to-end analytic_jacobian at a realistic band/resolution.
let
    mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                              Float32(0.07), Float32(0.08), E, Float32(0.75), Float32(-0.003))
    lines = [mk(700.0 + 1.3*j, 1.0e-21*(1+0.1j), 200.0 + 5j) for j in 0:110]
    ll = HITRANLinelist(lines); linelists = Dict(CO2 => ll)
    iasi = IASIInstrument(700.0, 850.0, 0.25, 601, 2.0, 0.5)   # 150 cm⁻¹, internal 0.001
    nlev = 10
    p = collect(range(1000.0, 100.0; length=nlev))
    T = 288.0 .+ (220.0-288.0).*(1000.0.-p)./900.0
    z = (1000.0.-p)./66.0
    prof = AtmosphericProfile(p, T, z, Dict(CO2=>fill(4.0e-4,nlev)))
    spec = StateVectorSpec(nlev, [CO2]; include_temperature=true, include_tsfc=true, include_emissivity=true)

    jac = analytic_jacobian(prof, linelists, spec; iasi=iasi, apply_continuum=false, with_ils=true)  # warm up
    t = @elapsed analytic_jacobian(prof, linelists, spec; iasi=iasi, apply_continuum=false, with_ils=true)
    @printf("\nanalytic_jacobian: n_state=%d, n_ν_hi=%d, %d channels → %.2f s\n",
            spec.n, 150_000, iasi.n_channels, t)
    @printf("  (direct-ILS tail alone would be ≈ n_state × 4.4 s ≈ %.0f s)\n", spec.n*4.4)
end
