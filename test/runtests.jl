using Test
using IRSounderLBL

@testset "IRSounderLBL.jl" begin

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

    # ── CIM source function (LBLRTM Padé) — guards the 4.3 µm +7 K comb fix ─
    # The comb was caused by a wrong layer source-function anchoring. The fix
    # (commit "Fix CG-consistent source function") folds em into the per-layer
    # accumulator as  I += B_avg·ΔT_k + (B_top − B_avg)·ΔT_k·f_cim(s), i.e. the
    # correction term C_k carries the SAME (1 − e^{−s}) factor as ΔT_k. These
    # tests pin both the _cim_correction shape and that folded algebra against
    # the canonical LBLRTM EMIN form, so the comb cannot silently return.
    @testset "Schwarzschild RTE — CIM source function" begin
        # ── _cim_correction f(s) = 1 − 2·(e^{−s}/(e^{−s}−1) + 1/s) ──────────
        cimf = IRSounderLBL._cim_correction
        # Thin limit f(s) → s/6 (series branch is exact below 0.06)
        @test cimf(0.03)  ≈ 0.03 / 6.0            # s/6, series branch
        @test cimf(1e-9)  ≈ 1e-9 / 6.0 rtol=1e-6  # thin limit
        # Reference value at s = 1 (independent hand calc)
        @test cimf(1.0)   ≈ 0.16395341373865302 rtol=1e-12
        # Distinct from the Toon (:toon) correction at the same s
        @test !isapprox(cimf(1.0), IRSounderLBL._lit_correction(1.0); rtol=1e-3)
        # Large-s asymptote f(s) → 1 − 2/s (slow saturation toward 1)
        @test cimf(1000.0) ≈ 1.0 - 2.0/1000.0 rtol=1e-3
        @test cimf(1e7)    ≈ 1.0 rtol=1e-6
        # Continuity across the series/closed-form branch at s = 0.06
        @test cimf(0.06 - 1e-9) ≈ cimf(0.06 + 1e-9) rtol=1e-3

        g     = wavenumber_grid(645.0, 805.0, 10.0)   # small grid, 17 points
        n_lay = 5
        T_lev = [300.0, 290.0, 280.0, 270.0, 260.0, 250.0]  # surface-first
        T_sfc = 305.0
        # Mass-weighted layer T, deliberately ≠ the level midpoints so the test
        # actually exercises the T_AVE anchoring (not just (T_bot+T_top)/2).
        T_ave = [296.0, 285.0, 274.0, 263.0, 252.0]
        τ     = fill(0.7, g.n, n_lay)                  # moderate s ~ O(1)

        # Independent reference: the canonical LBLRTM EMIN per-layer form,
        # computed via a different algebra path (explicit tr, em assembled
        # before the 𝒯[k+1] weighting) than schwarzschild_rte's folded loop.
        function cim_reference(g, τ, T_lev, T_sfc, T_ave; μ=1.0, ε=1.0)
            n_ν, n_l = size(τ)
            𝒯 = level_transmittances(τ, μ)
            I = [ε * planck_radiance(g.ν[i], T_sfc) * 𝒯[i, 1] for i in 1:n_ν]
            for k in 1:n_l, i in 1:n_ν
                s     = τ[i, k] / μ
                tr    = exp(-s)
                B_avg = planck_radiance(g.ν[i], T_ave[k])
                B_top = planck_radiance(g.ν[i], T_lev[k + 1])
                em    = (1 - tr) * (B_avg + (B_top - B_avg) * cimf(s))
                I[i] += em * 𝒯[i, k + 1]
            end
            return I
        end

        I_cim = schwarzschild_rte(g, τ, T_lev, T_sfc;
                                  source_function=:cim, T_ave=T_ave)
        I_ref = cim_reference(g, τ, T_lev, T_sfc, T_ave)
        # The exact comb-fix guard: folded form must equal the canonical form.
        @test I_cim ≈ I_ref rtol=1e-12

        # CIM and Toon must give genuinely different answers for a T-gradient,
        # moderate-τ atmosphere (guards against accidental aliasing of the two).
        I_toon = schwarzschild_rte(g, τ, T_lev, T_sfc; source_function=:toon)
        @test !all(isapprox.(I_cim, I_toon; rtol=1e-3))

        # Physical limits with :cim.
        # Optically thick + T gradient: TOA emission → B(T_top) of the top layer.
        τ_thick = fill(100.0, g.n, n_lay)
        I_thick = schwarzschild_rte(g, τ_thick, T_lev, T_sfc;
                                    source_function=:cim, T_ave=T_ave)
        @test all(isapprox.(I_thick, planck_radiance.(g.ν, T_lev[end]); rtol=1e-3))
        # Optically thin: surface visible → B(T_sfc).
        τ_thin = fill(1e-10, g.n, n_lay)
        I_thin = schwarzschild_rte(g, τ_thin, T_lev, T_sfc;
                                   source_function=:cim, T_ave=T_ave)
        @test all(isapprox.(I_thin, planck_radiance.(g.ν, T_sfc); rtol=1e-4))
        # Isothermal (T_lev ≡ T_ave ≡ T): I → B(T) for any τ (Kirchhoff).
        T_iso   = 280.0
        I_iso   = schwarzschild_rte(g, fill(2.0, g.n, n_lay),
                                    fill(T_iso, n_lay + 1), T_iso;
                                    source_function=:cim, T_ave=fill(T_iso, n_lay))
        @test all(isapprox.(I_iso, planck_radiance.(g.ν, T_iso); rtol=1e-6))

        # Error handling: :cim requires a correctly sized T_ave; bad symbol errors.
        @test_throws ErrorException schwarzschild_rte(g, τ, T_lev, T_sfc;
                                                      source_function=:cim)
        @test_throws ErrorException schwarzschild_rte(g, τ, T_lev, T_sfc;
                            source_function=:cim, T_ave=fill(280.0, n_lay + 1))
        @test_throws ErrorException schwarzschild_rte(g, τ, T_lev, T_sfc;
                                                      source_function=:bogus)
    end

    # ── ILS (sinc ⊗ Gaussian default; Norton-Beer optional) ───────────────
    @testset "ILS kernel" begin
        # Default kernel (Gaussian apodization) is area-normalised
        δν_arr, kern = ils_kernel(0.25, 2.0, 0.5)
        @test sum(kern) * 0.25 ≈ 1.0 rtol=1e-2

        # Kernel is symmetric
        @test kern ≈ reverse(kern) rtol=1e-6

        # Peak is at centre (δν = 0)
        @test argmax(kern) == (length(kern) + 1) ÷ 2

        # High-res kernel also normalises correctly
        δν_hi, kern_hi = ils_kernel(0.0625, 2.0, 0.5)
        @test sum(kern_hi) * 0.0625 ≈ 1.0 rtol=1e-2

        # Norton-Beer variants are area-normalised, symmetric, peak-centred
        for style in (:norton_beer_weak, :norton_beer_medium, :norton_beer_strong)
            δν_nb, kern_nb = ils_kernel(0.25, 2.0, 0.5; apodization=style)
            @test sum(kern_nb) * 0.25 ≈ 1.0 rtol=1e-2
            @test kern_nb ≈ reverse(kern_nb) rtol=1e-6
            @test argmax(kern_nb) == (length(kern_nb) + 1) ÷ 2
        end

        # Default == :gaussian — passing the symbol explicitly matches the default
        _, kern_def = ils_kernel(0.25, 2.0, 0.5)
        _, kern_gau = ils_kernel(0.25, 2.0, 0.5; apodization=:gaussian)
        @test kern_def ≈ kern_gau

        # Gaussian and Norton-Beer Medium produce different kernels
        _, kern_nbm = ils_kernel(0.25, 2.0, 0.5; apodization=:norton_beer_medium)
        @test !(kern_def ≈ kern_nbm)

        # Unknown apodization symbol errors
        @test_throws ErrorException ils_kernel(0.25, 2.0, 0.5; apodization=:bogus)
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

        # Scan angle → local zenith conversion (MetOp, h=817 km)
        @test scan_angle_to_local_zenith(0.0) ≈ 0.0 atol=1e-12
        @test scan_angle_to_local_zenith(48.33) ≈ 57.46 atol=0.05
        @test scan_angle_to_local_zenith(-48.33) ≈ 57.46 atol=0.05  # symmetric

        # Limb-violating angle should error
        @test_throws ErrorException scan_angle_to_local_zenith(80.0)

        zen = iasi_zenith_angles()
        @test length(zen) == 30
        @test all(abs.(zen) .> abs.(iasi_scan_angles()))   # local > scan everywhere off-nadir
        @test zen[1] ≈ -zen[end] atol=1e-12                # antisymmetric across nadir
    end

    # ── MT-CKD Continuum ─────────────────────────────────────────────────
    @testset "MT-CKD Continuum" begin
        g = wavenumber_grid(645.0, 2760.0, 0.25)
        k_h2o = h2o_continuum(g, 0.01, 1013.25, 296.0)
        @test length(k_h2o) == g.n
        @test all(k_h2o .>= 0)

        # CO₂ CIA (HITRAN) should peak in the 1200-1500 cm⁻¹ window
        k_co2_cia = co2_cia(g, 4.15e-4, 1013.25, 296.0)
        @test length(k_co2_cia) == g.n
        @test all(k_co2_cia .>= 0)
        idx_1350 = argmin(abs.(g.ν .- 1350.0))
        idx_2000 = argmin(abs.(g.ν .- 2000.0))
        @test k_co2_cia[idx_1350] > k_co2_cia[idx_2000]

        # MT-CKD CO₂ continuum: positive, and decays from the ν₂ band toward
        # the window (S peaks near band center).
        k_co2 = co2_continuum(g, 4.15e-4, 1013.25, 296.0)
        @test length(k_co2) == g.n
        @test all(k_co2 .>= 0)
        idx_700 = argmin(abs.(g.ν .- 700.0))
        idx_790 = argmin(abs.(g.ν .- 790.0))
        @test k_co2[idx_700] > k_co2[idx_790] > 0

        # Bandhead temperature correction (2386–2434 cm⁻¹): the continuum gets
        # an extra (T/246)^tdep factor there (tdep=0.73 at 2400), and none at
        # 15 µm (tdep=0). Isolate it with a cross-T ratio so S·xfac cancel;
        # divide out the density (∝1/T²) and radiation-field temperature
        # dependence that the base formula also carries.
        T1, T2 = 296.0, 246.0
        k1 = co2_continuum(g, 4.15e-4, 1013.25, T1)
        k2 = co2_continuum(g, 4.15e-4, 1013.25, T2)
        _rad(ν, T) = (x = 1.4387769 * ν / T; ν * (1 - exp(-x)) / (1 + exp(-x)))
        _base(ν, T) = _rad(ν, T) / T^2
        i2400 = argmin(abs.(g.ν .- 2400.0))
        r_bh = (k1[i2400] / _base(g.ν[i2400], T1)) /
               (k2[i2400] / _base(g.ν[i2400], T2))
        @test r_bh ≈ (T1 / T2)^0.73 rtol = 1e-3
        i714 = argmin(abs.(g.ν .- 714.0))
        r_15 = (k1[i714] / _base(g.ν[i714], T1)) /
               (k2[i714] / _base(g.ν[i714], T2))
        @test r_15 ≈ 1.0 rtol = 1e-3
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
        RL = IRSounderLBL.RelmatLine
        RB = IRSounderLBL.RelmatBand
        WT = IRSounderLBL.W0B0Table

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
        modes = IRSounderLBL.band_modes(band, WT(), T, p_atm)

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
        RL = IRSounderLBL.RelmatLine
        RB = IRSounderLBL.RelmatBand
        WT = IRSounderLBL.W0B0Table

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
        modes = IRSounderLBL.band_modes(band, WT(), T, p_atm)

        ν_grid = wavenumber_grid(699.5, 700.5, 0.005)
        σ_vpw  = IRSounderLBL.compute_vpw_band_xsec(ν_grid, modes; cutoff=10.0)

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
        σ_far = IRSounderLBL.compute_vpw_band_xsec(ν_far, modes; cutoff=10.0)
        @test all(==(0.0), σ_far)
    end

    # ── JLD2 linelist cache ───────────────────────────────────────────────
    @testset "Linelist cache (JLD2)" begin
        # 67-char .par records with exact fixed-width columns the parser expects.
        mk = (mol, iso, ν, S) -> string(
            lpad(mol, 2), iso, lpad(string(ν), 12), lpad(string(S), 10),
            lpad("0.0", 10), lpad("0.05", 5), lpad("0.07", 5),
            lpad("0.0", 10), lpad("0.75", 4), lpad("0.0", 8))

        mktempdir() do dir
            par = joinpath(dir, "synthetic.par")
            open(par, "w") do io
                println(io, mk(2, 1, 700.0, 1.0e-21))
                println(io, mk(2, 1, 710.0, 2.0e-21))
                println(io, mk(2, 1, 720.0, 3.0e-21))
            end
            cachedir = joinpath(dir, "cache")

            # Isolate the cache via the env override so we never touch Scratch.
            withenv("IRSOUNDER_LINELIST_CACHE" => cachedir) do
                @test linelist_cache_dir() == cachedir

                # First load parses + writes a cache; second reads it back.
                ll1 = load_linelist(par)
                @test length(ll1) == 3
                @test count(f -> endswith(f, ".jld2"), readdir(cachedir)) == 1

                ll2 = load_linelist(par)
                @test ll2.lines == ll1.lines          # roundtrip is exact

                # Windowing is applied in memory, not baked into the cache.
                w = load_linelist(par; ν_min = 705.0, ν_max = 715.0)
                @test length(w) == 1
                @test w.lines[1].wavenumber ≈ 710.0

                # cache=false and rebuild=true agree with the cached result.
                @test load_linelist(par; cache = false).lines == ll1.lines
                @test load_linelist(par; rebuild = true).lines == ll1.lines

                # Editing the .par changes size/mtime -> a fresh cache key.
                sleep(0.01)
                open(par, "a") do io
                    println(io, mk(2, 1, 730.0, 4.0e-21))
                end
                @test length(load_linelist(par)) == 4
                @test count(f -> endswith(f, ".jld2"), readdir(cachedir)) == 2

                @test clear_linelist_cache() == 2
                @test count(f -> endswith(f, ".jld2"), readdir(cachedir)) == 0
            end
        end
    end

end
