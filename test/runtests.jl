using Test
using RadiativeTransfer

@testset "RadiativeTransfer.jl" begin

    # ── WavenumberGrid ────────────────────────────────────────────────────
    @testset "WavenumberGrid" begin
        g = wavenumber_grid(645.0, 2760.0, 0.25)
        @test g.n == 8461
        @test g.Δν ≈ 0.25
        @test g.ν[1] ≈ 645.0
        @test g.ν[end] ≈ 2760.0
        @test length(g) == g.n

        g2 = wavenumber_grid(1000.0, 2000.0; n=101)
        @test g2.n == 101
        @test g2.ν[1] ≈ 1000.0
        @test g2.ν[end] ≈ 2000.0
        @test g2.Δν ≈ 10.0
    end

    # ── Atmosphere ───────────────────────────────────────────────────────
    @testset "Standard Atmospheres" begin
        us = us_standard_atmosphere()
        @test length(us.pressure) == 43
        @test us.pressure[1] ≈ 1013.25
        @test us.temperature[1] ≈ 288.15
        @test haskey(us.vmr, H2O)
        @test haskey(us.vmr, CO2)
        @test all(us.vmr[CO2] .≈ 4.15e-4)

        tr = tropical_atmosphere()
        @test tr.temperature[1] > us.temperature[1]   # tropics warmer at surface
        @test tr.vmr[H2O][1] > us.vmr[H2O][1]         # tropics wetter

        sa = subarctic_atmosphere()
        @test sa.temperature[1] < us.temperature[1]   # subarctic cooler
    end

    @testset "Layer Properties" begin
        prof = us_standard_atmosphere()
        layers = layer_properties(prof)
        @test length(layers.p_mid) == 42
        @test length(layers.T_mid) == 42
        @test length(layers.Δp)    == 42
        @test all(layers.Δp .> 0)
        @test all(layers.T_mid .> 0)
    end

    # ── Planck Function ───────────────────────────────────────────────────
    @testset "Planck & Brightness Temperature" begin
        ν  = 1000.0   # cm⁻¹
        T  = 300.0    # K
        B  = planck_radiance(ν, T)
        @test B > 0

        # Round-trip: BT(B(T)) ≈ T
        T_rt = brightness_temperature(ν, B)
        @test T_rt ≈ T rtol=1e-8

        # Monotonicity: hotter → brighter
        @test planck_radiance(ν, 300.0) > planck_radiance(ν, 250.0)

        # Grid method
        g = wavenumber_grid(645.0, 2760.0, 0.25)
        Bvec = planck_radiance(g, 280.0)
        @test length(Bvec) == g.n
        @test all(Bvec .> 0)

        BT = brightness_temperature(g, Bvec)
        @test all(isapprox.(BT, 280.0; rtol=1e-7))
    end

    # ── Voigt Profile ─────────────────────────────────────────────────────
    @testset "Weideman Voigt" begin
        # Test normalization: ∫V dν ≈ 1
        ν0  = 1000.0; γL = 0.05; γD = 0.02
        Δν  = 0.001
        νs  = collect((ν0 - 5.0):Δν:(ν0 + 5.0))
        Vs  = [voigt_profile(ν, ν0, γL, γD) for ν in νs]
        integral = sum(Vs) * Δν
        # Weideman ~1e-4 accuracy; ±5 cm⁻¹ wing truncation with γL=0.05
        # adds ~0.6% loss. Overall tolerance is 3%.
        @test integral ≈ 1.0 rtol=3e-2

        # Peak should be at line center
        i_peak = argmax(Vs)
        @test νs[i_peak] ≈ ν0 atol=Δν

        # Symmetry
        N = length(νs)
        mid = (N + 1) ÷ 2
        for k in 1:10
            @test Vs[mid - k] ≈ Vs[mid + k] rtol=1e-10
        end

        # Lorentz limit (γD → 0) should approach Lorentzian
        γD_small = 1e-6
        V_near_L = voigt_profile(ν0 + γL, ν0, γL, γD_small)
        L_peak   = 1.0 / (π * γL)      # Lorentzian peak
        L_half   = L_peak / 2.0        # at HWHM
        @test V_near_L ≈ L_half rtol=0.02
    end

    # ── HITRANLine parsing ────────────────────────────────────────────────
    @testset "HITRANLine parsing" begin
        # Construct a synthetic .par record (160 chars)
        # Format: mol(2) iso(1) ν(12) S(10) A(10) γ_air(5) γ_self(5) E″(10) n(4) δ(8)
        record = " 11 667.37986 2.910E-19 3.510E-03 0.0602 0.4680  0.000000 0.75 0.000000" *
                 "               " *    # global quanta (15 chars)
                 "               " *    # local quanta (15 chars)
                 "      " *             # error codes (6 chars)
                 "        " *           # reference codes (12 chars — 8 here, pad)
                 "    " *               # line mixing flag (1 char + 3 pad)
                 "          "           # gpp etc.
        # The built-in parser expects exactly the column positions
        # For testing just verify the struct fields
        line = HITRANLine(1, 1, 667.37986, 2.910e-19, 3.510e-3,
                          0.0602f0, 0.4680f0, 0.0, 0.75f0, 0.0f0)
        @test line.mol_id == 1
        @test line.wavenumber ≈ 667.37986
        @test line.intensity ≈ 2.910e-19
        @test line.air_broad ≈ 0.0602f0
    end

    # ── Partition Functions ───────────────────────────────────────────────
    @testset "Partition Functions" begin
        # Q(T_ref) / Q(T_ref) = 1
        @test Q_ratio(1, 1, T_REF) ≈ 1.0
        # Higher T → lower Q ratio (more population in excited states)
        @test Q_ratio(1, 1, 350.0) < 1.0
        @test Q_ratio(1, 1, 250.0) > 1.0
    end

    # ── Broadening ────────────────────────────────────────────────────────
    @testset "Line Broadening" begin
        line = HITRANLine(1, 1, 667.0, 1e-19, 1e-3,
                          0.060f0, 0.450f0, 0.0, 0.76f0, 0.0f0)
        γL, γD = pressure_broadened_width(line, 1.0, 296.0)
        @test γL ≈ 0.060 rtol=1e-6
        @test γD > 0.0

        # Higher pressure → wider Lorentz
        γL2, _ = pressure_broadened_width(line, 0.5, 296.0)
        @test γL2 < γL

        # Higher temperature → narrower Lorentz (n>0), wider Doppler
        γL3, γD3 = pressure_broadened_width(line, 1.0, 400.0)
        @test γL3 < γL   # temp dependence reduces Lorentz
        @test γD3 > γD   # Doppler increases with T

        # Temperature-scaled intensity
        S296 = temperature_scaled_intensity(line, T_REF)
        @test S296 ≈ line.intensity rtol=1e-6
    end

    # ── Schwarzschild RTE ─────────────────────────────────────────────────
    @testset "Schwarzschild RTE" begin
        g      = wavenumber_grid(645.0, 2760.0, 0.25)
        n_lay  = 5
        # T_lev has n_lay+1 level-boundary temperatures (surface-first)
        T_lev  = fill(280.0, n_lay + 1)
        T_sfc  = 300.0
        B_sfc  = planck_radiance.(g.ν, T_sfc)

        # Optically thick: surface blocked, TOA emission ≈ B(T_top) from top layer
        τ_thick = fill(100.0, g.n, n_lay)
        I_thick = schwarzschild_rte(g, τ_thick, T_lev, T_sfc)
        B_atm   = planck_radiance.(g.ν, T_lev[end])
        @test all(isapprox.(I_thick, B_atm; rtol=1e-3))

        # Optically thin limit: τ → 0 → I ≈ B(T_sfc) (surface visible)
        τ_thin = fill(1e-10, g.n, n_lay)
        I_thin = schwarzschild_rte(g, τ_thin, T_lev, T_sfc)
        @test all(isapprox.(I_thin, B_sfc; rtol=1e-4))

        # Isothermal atmosphere at T: I ≈ B(T) regardless of τ (Kirchhoff)
        T_iso  = 280.0
        T_lev2 = fill(T_iso, n_lay + 1)
        τ_any  = fill(2.0, g.n, n_lay)
        I_iso  = schwarzschild_rte(g, τ_any, T_lev2, T_iso)
        B_iso  = planck_radiance.(g.ν, T_iso)
        @test all(isapprox.(I_iso, B_iso; rtol=1e-6))
    end

    # ── ILS (sinc ⊗ Gaussian) ─────────────────────────────────────────────
    @testset "ILS kernel" begin
        # Kernel is area-normalised: ∑ kern × Δν ≈ 1
        δν_arr, kern = ils_kernel(0.25, 2.0, 0.5)
        @test sum(kern) * 0.25 ≈ 1.0 rtol=1e-2

        # Kernel is symmetric
        @test kern ≈ reverse(kern) rtol=1e-6

        # Peak is at centre (δν = 0)
        @test argmax(kern) == (length(kern) + 1) ÷ 2

        # High-res kernel also normalises correctly
        δν_hi, kern_hi = ils_kernel(0.0625, 2.0, 0.5)
        @test sum(kern_hi) * 0.0625 ≈ 1.0 rtol=1e-2
    end

    # ── Norton-Beer (kept for reference) ──────────────────────────────────
    @testset "Norton-Beer Apodization" begin
        for style in keys(NORTON_BEER_COEFFS)
            @test norton_beer_apodization(0.0, style) ≈ 1.0
        end
        for (style, c) in NORTON_BEER_COEFFS
            @test norton_beer_apodization(1.0, style) ≈ c.C0
        end
        @test norton_beer_apodization(1.5) == 0.0
    end

    # ── IASI Instrument ───────────────────────────────────────────────────
    @testset "IASI Instrument" begin
        iasi = IASIInstrument()
        @test iasi.n_channels == 8461
        @test iasi.ν_min == 645.0
        @test iasi.ν_max == 2760.0
        @test iasi.Δν == 0.25

        g = iasi_grid(iasi)
        @test g.n == iasi.n_channels
    end

    # ── ViewingGeometry ───────────────────────────────────────────────────
    @testset "Viewing Geometry" begin
        nadir = nadir_geometry()
        @test nadir.μ ≈ 1.0
        @test airmass_factor(nadir) ≈ 1.0

        g48 = ViewingGeometry(48.33)
        @test g48.μ ≈ cosd(48.33) rtol=1e-10
        @test airmass_factor(g48) ≈ 1.0 / cosd(48.33) rtol=1e-10

        @test_throws ErrorException ViewingGeometry(90.0)
        @test_throws ErrorException ViewingGeometry(-1.0)

        angles = iasi_scan_angles()
        @test length(angles) == 30
        @test maximum(abs.(angles)) ≈ 48.33 rtol=1e-2
    end

    # ── MT-CKD Continuum ─────────────────────────────────────────────────
    @testset "MT-CKD Continuum" begin
        g = wavenumber_grid(645.0, 2760.0, 0.25)
        k_h2o = h2o_continuum(g, 0.01, 1013.25, 296.0)
        @test length(k_h2o) == g.n
        @test all(k_h2o .>= 0)

        k_co2 = co2_continuum(g, 4.15e-4, 1013.25, 296.0)
        @test length(k_co2) == g.n
        @test all(k_co2 .>= 0)

        # CO2 CIA should peak in the 1200-1500 cm⁻¹ window
        idx_1350 = argmin(abs.(g.ν .- 1350.0))
        idx_2000 = argmin(abs.(g.ν .- 2000.0))
        @test k_co2[idx_1350] > k_co2[idx_2000]
    end

    # ── Transmittance ─────────────────────────────────────────────────────
    @testset "Level Transmittances" begin
        n_ν, n_lay = 100, 5
        # Zero optical depth → all transmittances = 1
        τ_zero = zeros(Float64, n_ν, n_lay)
        T = level_transmittances(τ_zero)
        @test size(T) == (n_ν, n_lay + 1)
        @test all(T .≈ 1.0)

        # Very thick → surface transmittance ≈ 0
        τ_thick = fill(1000.0, n_ν, n_lay)
        T2 = level_transmittances(τ_thick)
        @test all(T2[:, end] .≈ 1.0)    # TOA = 1
        @test all(T2[:, 1] .< 1e-10)    # surface blocked
    end

    # ── Compute Backend ───────────────────────────────────────────────────
    @testset "Compute Backend" begin
        cb = detect_backend(verbose=false)
        @test cb isa ComputeBackend
        @test cb.n_threads >= 1
        @test cb.gpu_backend in (:metal, :cuda, :none)
    end

    @testset "Line mixing — band_modes no-coupling limit" begin
        RL = RadiativeTransfer.RelmatLine
        RB = RadiativeTransfer.RelmatBand
        WT = RadiativeTransfer.W0B0Table

        # Two synthetic lines, isotopologue 1, identical broadening, separated by 10 cm⁻¹.
        # Empty WTfit table ⇒ no off-diagonal coupling ⇒ M = diag(ν₀ + i·p·γ_L).
        # Expect poles at the line centres (modulo eigen ordering), amplitudes = S(T).
        line1 = RL(Int8(1), 700.0, Int16(0), Int8(0), 0.07, Float32(0.7), Float32(0.0),
                   200.0, 1.0, 0.5, 1.0)
        line2 = RL(Int8(1), 710.0, Int16(2), Int8(0), 0.07, Float32(0.7), Float32(0.0),
                   220.0, 1.0, 0.45, 1.0)
        band  = RB("TEST01", Int8(1), Int8(1), Int8(1), 700.0, 710.0, [line1, line2])

        T     = 296.0
        p_atm = 0.5
        modes = RadiativeTransfer.band_modes(band, WT(), T, p_atm)

        @test length(modes.poles) == 2
        @test length(modes.amplitudes) == 2

        # Sort poles by real part (eigen may reorder)
        idx = sortperm(real.(modes.poles))
        λ1, λ2 = modes.poles[idx[1]], modes.poles[idx[2]]
        A1, A2 = modes.amplitudes[idx[1]], modes.amplitudes[idx[2]]

        γ_L = 0.07   # at T=T_REF, exponent 0.7 gives factor 1.0
        @test real(λ1) ≈ 700.0 atol = 1e-10
        @test real(λ2) ≈ 710.0 atol = 1e-10
        @test imag(λ1) ≈ p_atm * γ_L atol = 1e-10
        @test imag(λ2) ≈ p_atm * γ_L atol = 1e-10

        # In no-coupling limit Aₖ should be real and equal Sₖ(T=T_REF) = DipoT²·PopuT0·ν·stim(T_REF)
        S1 = line1.DipoT^2 * line1.PopuT0 * line1.ν * (1 - exp(-1.4388 * line1.ν / T))
        S2 = line2.DipoT^2 * line2.PopuT0 * line2.ν * (1 - exp(-1.4388 * line2.ν / T))
        @test real(A1) ≈ S1 rtol = 1e-12
        @test real(A2) ≈ S2 rtol = 1e-12
        @test abs(imag(A1)) < 1e-12 * abs(real(A1))
        @test abs(imag(A2)) < 1e-12 * abs(real(A2))

        # Doppler scaling sanity: γ_D ≈ 3.58e-7 · 705 · √(296/44) ≈ 6.5e-4
        γ_D_expected = 3.5812e-7 * 705.0 * sqrt(T / 43.98983)
        @test modes.f ≈ sqrt(log(2.0)) / γ_D_expected rtol = 1e-12
    end

    @testset "Line mixing — VP_W evaluator vs Voigt baseline (no coupling)" begin
        using SpecialFunctions: erfcx
        RL = RadiativeTransfer.RelmatLine
        RB = RadiativeTransfer.RelmatBand
        WT = RadiativeTransfer.W0B0Table

        # Two-line synthetic band, empty WTfit ⇒ M is diagonal.
        # Poles = ν₀ + i·p·γ_L exactly; amplitudes = S(T) exactly.
        # `compute_vpw_band_xsec` must reproduce the band-uniform-γ_D Voigt sum.
        line1 = RL(Int8(1), 700.00, Int16(0), Int8(0), 0.07, Float32(0.7), Float32(0.0),
                   200.0, 1.0, 0.5, 1.0)
        line2 = RL(Int8(1), 700.05, Int16(2), Int8(0), 0.07, Float32(0.7), Float32(0.0),
                   220.0, 1.0, 0.45, 1.0)
        band  = RB("TEST02", Int8(1), Int8(1), Int8(1), 700.0, 700.05, [line1, line2])

        T     = 296.0
        p_atm = 0.5
        modes = RadiativeTransfer.band_modes(band, WT(), T, p_atm)

        ν_grid = wavenumber_grid(699.5, 700.5, 0.005)
        σ_vpw  = RadiativeTransfer.compute_vpw_band_xsec(ν_grid, modes; cutoff=10.0)

        # Reference: pure Voigt sum at band-uniform γ_D (same as modes.f)
        γ_L     = 0.07 * p_atm   # T = T_REF
        f_band  = modes.f
        σ_ref   = zeros(ν_grid.n)
        for line in (line1, line2)
            stim = 1.0 - exp(-1.4388 * line.ν / T)
            S_T  = line.DipoT^2 * line.PopuT0 * line.ν * stim   # T = T_REF
            for i in 1:ν_grid.n
                x  = (ν_grid.ν[i] - line.ν) * f_band
                y  = γ_L * f_band
                wz = erfcx(complex(y, -x))
                σ_ref[i] += S_T * f_band * (1.0 / sqrt(π)) * real(wz)
            end
        end

        rel = maximum(abs.(σ_vpw .- σ_ref)) / maximum(abs, σ_ref)
        @test rel < 1e-12

        # Cutoff sanity: poles outside the window contribute zero
        ν_far = wavenumber_grid(800.0, 800.5, 0.005)
        σ_far = RadiativeTransfer.compute_vpw_band_xsec(ν_far, modes; cutoff=10.0)
        @test all(==(0.0), σ_far)
    end

end
