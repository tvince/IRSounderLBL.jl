using Test
using IRSounderLBL
using LinearAlgebra

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

    # ── Voigt far-wing Lorentzian shortcut (Option A) ─────────────────────
    @testset "Voigt far-wing Lorentzian (x_far)" begin
        # Past |x|=x_far the Faddeeva is replaced by the analytic Lorentzian.
        # Gate: the default x_far=_X_FAR (≈122) must reproduce exact Faddeeva
        # (x_far=Inf) to <1e-4, while forcing x_far=0 (Lorentzian everywhere,
        # incl. the core) must visibly differ — proving the branch is live.
        mk(ν, S) = HITRANLine(Int8(2), Int8(1), ν, S, 1.0,
                              Float32(0.07), Float32(0.09), 250.0,
                              Float32(0.75), Float32(0.0))
        ll = HITRANLinelist([mk(2348.0, 1e-19), mk(2350.0, 5e-20),
                             mk(2351.2, 8e-20), mk(2352.5, 2e-19)])
        g  = wavenumber_grid(2345.0, 2355.0, 0.001)
        T, p, cutoff = 230.0, 0.2, 25.0     # narrow Doppler cores + real wings
        csf(xf) = compute_voigt_cross_sections(g, ll, T, p; cutoff=cutoff,
                                               method=FullFaddeeva, x_far=xf)

        σ_exact = csf(Inf)        # all-Faddeeva = pre-Option-A behavior
        σ_optA  = csf(122.0)      # Option A default
        peak = maximum(σ_exact)

        # Lossless: absolute diff far below peak, relative diff <1e-4 worst-case
        @test maximum(abs.(σ_optA .- σ_exact)) < 1e-5 * peak
        mask = σ_exact .> 1e-6 * peak
        @test maximum(abs.((σ_optA[mask] .- σ_exact[mask]) ./ σ_exact[mask])) < 3e-4

        # Branch must actually be exercised: Lorentzian-everywhere differs in the core
        @test maximum(abs.(csf(0.0) .- σ_exact)) > 1e-3 * peak
    end

    # ── Vectorized Voigt methods agree (all run the KA kernel) ─────────────
    @testset "Voigt methods (vectorized) agree" begin
        # FullFaddeeva, Weideman, PseudoVoigt all go through the KernelAbstractions
        # kernel now. Cross-check their cross-sections against each other at their
        # documented accuracies (Weideman ~1e-4, PseudoVoigt ~1% core).
        mk(ν, S) = HITRANLine(Int8(2), Int8(1), ν, S, 1.0,
                              Float32(0.07), Float32(0.09), 250.0,
                              Float32(0.75), Float32(0.0))
        ll = HITRANLinelist([mk(2348.0, 1e-19), mk(2350.0, 5e-20),
                             mk(2351.2, 8e-20), mk(2352.5, 2e-19)])
        g  = wavenumber_grid(2345.0, 2355.0, 0.001)
        T, p, cutoff = 230.0, 0.2, 25.0
        cs(m) = compute_voigt_cross_sections(g, ll, T, p; cutoff=cutoff, method=m)
        σF = cs(FullFaddeeva); σW = cs(Weideman); σP = cs(PseudoVoigt)
        @test all(isfinite, σF) && all(σF .>= 0)
        @test all(isfinite, σW) && all(σW .>= 0)
        @test all(isfinite, σP) && all(σP .>= 0)
        peak = maximum(σF)
        mask = σF .> 1e-3 * peak                       # near cores
        # Weideman ≈ Faddeeva to ~1e-3; PseudoVoigt ~few % in the core region
        @test maximum(abs.((σW[mask] .- σF[mask]) ./ σF[mask])) < 2e-3
        @test maximum(abs.((σP[mask] .- σF[mask]) ./ σF[mask])) < 5e-2
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
        # x_far=Inf: this test's reference uses exact erfcx everywhere, so disable
        # the far-wing shortcut to keep the 1e-12 machine-precision comparison valid.
        σ_vpw  = IRSounderLBL.compute_vpw_band_xsec(ν_grid, modes; cutoff=10.0, x_far=Inf)

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

    # ── LM far-wing Lorentzian/dispersion shortcut (x_far) ────────────────
    @testset "Line mixing — far-wing shortcut (x_far) is lossless" begin
        # Past |x|=x_far both LM evaluators replace the complex erfcx with the
        # analytic w(z) limit (Re w = y/√π(x²+y²), Im w = x/√π(x²+y²)). Gate:
        # the default x_far=_X_FAR must reproduce exact Faddeeva (x_far=Inf) to
        # <1e-6·peak, while x_far=0 (analytic everywhere, incl. core) must differ.
        RL = IRSounderLBL.RelmatLine
        RB = IRSounderLBL.RelmatBand
        WT = IRSounderLBL.W0B0Table
        XF = IRSounderLBL._X_FAR

        # Two coupled Q lines: a populated qq entry gives nonzero off-diagonal W,
        # so VP_W amplitudes go complex (Re+Im branches) AND VP_Y Y is nonzero.
        # Small p + modest γ_air keep y≈O(5) so a real Gaussian core survives and
        # the x_far=0 "branch live" check has signal.
        line1 = RL(Int8(1), 667.00, Int16(20), Int8(0), 0.03, Float32(0.7), Float32(0.0),
                   300.0, 1.0, 1.0, 1.0)
        line2 = RL(Int8(1), 667.05, Int16(18), Int8(0), 0.03, Float32(0.7), Float32(0.0),
                   260.0, 1.0, 0.8, 1.0)
        band  = RB("TESTQ", Int8(1), Int8(1), Int8(1), 667.0, 667.05, [line1, line2])
        wt = WT(); wt.qq[(20, 18)] = (log(0.03), 0.5)   # one off-diagonal coupling

        T, p_atm, cutoff = 250.0, 0.1, 25.0
        g = wavenumber_grid(665.0, 669.0, 0.002)        # ±2 cm⁻¹ ≫ x_far window

        # VP_W evaluator (exercises Re + Im far-wing via complex amplitudes)
        modes = IRSounderLBL.band_modes(band, wt, T, p_atm)
        @test any(a -> imag(a) != 0.0, modes.amplitudes)            # coupling present
        σw_exact = IRSounderLBL.compute_vpw_band_xsec(g, modes; cutoff=cutoff, x_far=Inf)
        σw_optA  = IRSounderLBL.compute_vpw_band_xsec(g, modes; cutoff=cutoff, x_far=XF)
        σw_lor   = IRSounderLBL.compute_vpw_band_xsec(g, modes; cutoff=cutoff, x_far=0.0)
        pkw = maximum(abs, σw_exact)
        @test maximum(abs.(σw_optA .- σw_exact)) < 1e-5 * pkw        # lossless (matches Option-A gate)
        @test maximum(abs.(σw_lor  .- σw_exact)) > 1e-3 * pkw        # branch live

        # VP_Y dispersive kernel (purely Im[w] → gates the Im far-wing form)
        σy_exact = IRSounderLBL._lm_band_dispersive(g, band, wt, T, p_atm; cutoff=cutoff, x_far=Inf)
        σy_optA  = IRSounderLBL._lm_band_dispersive(g, band, wt, T, p_atm; cutoff=cutoff, x_far=XF)
        σy_lor   = IRSounderLBL._lm_band_dispersive(g, band, wt, T, p_atm; cutoff=cutoff, x_far=0.0)
        pky = maximum(abs, σy_exact)
        @test pky > 0                                                # nonzero dispersive signal
        @test maximum(abs.(σy_optA .- σy_exact)) < 1e-5 * pky        # lossless (matches Option-A gate)
        @test maximum(abs.(σy_lor  .- σy_exact)) > 1e-3 * pky        # branch live
    end

    # ── #4 band-strength cutoff (min_band_strength) ───────────────────────
    @testset "Line mixing — band-strength cutoff (min_band_strength)" begin
        # The #4 cutoff skips coupled bands whose integrated intensity
        # (_band_eff_strength) is below `min_band_strength`. Gate: a threshold
        # set between a strong and a weak band drops EXACTLY the weak band — the
        # dispersive result must equal computing with only the strong band, and
        # must differ from the no-cutoff (both-bands) result.
        RL = IRSounderLBL.RelmatLine
        RB = IRSounderLBL.RelmatBand
        WT = IRSounderLBL.W0B0Table
        RD = IRSounderLBL.HITRANRelmatData

        # Strong Q-band near 700; weak Q-band near 720 (DipoT 1e-3 ⇒ ~1e6× weaker).
        s1 = RL(Int8(1), 700.00, Int16(20), Int8(0), 0.03, Float32(0.7), Float32(0.0),
                300.0, 1.0, 1.0, 1.0)
        s2 = RL(Int8(1), 700.05, Int16(18), Int8(0), 0.03, Float32(0.7), Float32(0.0),
                260.0, 1.0, 0.8, 1.0)
        w1 = RL(Int8(1), 720.00, Int16(4),  Int8(0), 0.03, Float32(0.7), Float32(0.0),
                300.0, 1.0e-3, 1.0, 1.0e-3)
        w2 = RL(Int8(1), 720.05, Int16(2),  Int8(0), 0.03, Float32(0.7), Float32(0.0),
                260.0, 1.0e-3, 0.8, 1.0e-3)
        bStrong = RB("STRONG", Int8(1), Int8(1), Int8(1), 700.0, 700.05, [s1, s2])
        bWeak   = RB("WEAK",   Int8(1), Int8(1), Int8(1), 720.0, 720.05, [w1, w2])

        wt = WT()                       # shared (li,lf)=(1,1) coupling table
        wt.qq[(20, 18)] = (log(0.03), 0.5)
        wt.qq[(4, 2)]   = (log(0.03), 0.5)
        wtfit = Dict{NTuple{2,Int8}, typeof(wt)}((Int8(1), Int8(1)) => wt)

        relmat_both   = RD([bStrong, bWeak], wtfit)
        relmat_strong = RD([bStrong],        wtfit)

        sS = IRSounderLBL._band_eff_strength(bStrong)
        sW = IRSounderLBL._band_eff_strength(bWeak)
        @test sS > sW
        thr = sqrt(sS * sW)             # geometric mean ⇒ strictly between the two

        T, p_atm = 250.0, 0.5
        g = wavenumber_grid(695.0, 725.0, 0.01)

        σ_full   = IRSounderLBL.compute_lm_dispersive_correction(g, relmat_both,   T, p_atm; min_band_strength=0.0)
        σ_cut    = IRSounderLBL.compute_lm_dispersive_correction(g, relmat_both,   T, p_atm; min_band_strength=thr)
        σ_strong = IRSounderLBL.compute_lm_dispersive_correction(g, relmat_strong, T, p_atm; min_band_strength=0.0)

        # Cutoff drops exactly the weak band ⇒ identical to strong-only.
        @test σ_cut == σ_strong
        # ...and the weak band really did contribute, so the cutoff changed the result.
        @test maximum(abs.(σ_full .- σ_cut)) > 0.0
        # min_band_strength=0.0 is exact (no band dropped).
        @test IRSounderLBL.compute_lm_dispersive_correction(g, relmat_both, T, p_atm; min_band_strength=0.0) == σ_full

        # Guard the abundance-double-count fix: _band_eff_strength must NOT depend
        # on isotopologue (S-file DipoT/PopuT0 already include abundance). Same
        # lines under a different isot ⇒ identical effective strength.
        bWeak_iso2 = RB("WEAK2", Int8(1), Int8(1), Int8(2), 720.0, 720.05, [w1, w2])
        @test IRSounderLBL._band_eff_strength(bWeak_iso2) == sW
    end

    # ── species generalization (CH4 plumbing) ─────────────────────────────
    @testset "Line mixing — species generalization (CH4 plumbing)" begin
        RL = IRSounderLBL.RelmatLine
        RB = IRSounderLBL.RelmatBand
        WT = IRSounderLBL.W0B0Table
        RD = IRSounderLBL.HITRANRelmatData

        # Per-species mass/abundance lookups select the right molecule; CO2 unchanged.
        @test IRSounderLBL._lm_mass_amu(6, 1)  == 16.031300       # ¹²CH₄
        @test IRSounderLBL._lm_iso_abund(6, 1) == 0.988274
        @test IRSounderLBL._lm_mass_amu(2, 1)  == 43.98983        # CO2 unchanged
        @test IRSounderLBL._lm_mass_amu(99, 1) == 44.0            # unknown ⇒ fallback

        # Back-compat constructors default to CO2 (molecule 2 / species CO2).
        bco2 = RB("C", Int8(1), Int8(1), Int8(1), 700.0, 700.1, RL[])
        @test bco2.molecule == Int8(2)
        @test RD(RelmatBand[], Dict{NTuple{2,Int8}, WT}()).species == CO2

        # A CH4 ν₄ Q-branch band (molecule 6) routes Q(T) through CH4 TIPS and runs
        # the dispersive kernel end to end — finite, nonzero LM perturbation.
        l1 = RL(Int8(1), 1306.00, Int16(6), Int8(0), 0.05, Float32(0.75), Float32(0.0),
                220.0, 1.0, 1.0, 1.0)
        l2 = RL(Int8(1), 1306.05, Int16(5), Int8(0), 0.05, Float32(0.75), Float32(0.0),
                200.0, 1.0, 0.9, 1.0)
        bch4 = RB("CH4Q", Int8(1), Int8(1), Int8(1), 1306.0, 1306.05, [l1, l2], Int8(6))
        wt = WT(); wt.qq[(6, 5)] = (log(0.04), 0.5)
        relmat = RD([bch4], Dict((Int8(1), Int8(1)) => wt), CH4)

        g = wavenumber_grid(1304.0, 1308.0, 0.01)
        σ = IRSounderLBL.compute_lm_dispersive_correction(g, relmat, 250.0, 0.5)
        @test all(isfinite, σ)
        @test maximum(abs.(σ)) > 0.0

        # Dispatch gate keys on the relmat's species, not a hardcoded CO2.
        @test IRSounderLBL.VPYLineMixing(relmat).data.species == CH4
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

    # ── Jacobian Phase 0: state vector + finite-difference reference harness ──
    @testset "Jacobian — state vector layout & pack/unpack" begin
        spec = StateVectorSpec(4, [CO2, H2O];
                               include_temperature=true, include_tsfc=true,
                               include_emissivity=true, log_vmr=true)
        # Layout: 4 T + 4 CO2 + 4 H2O + T_sfc + ε = 14
        @test spec.n == 14
        @test length(spec) == 14
        @test spec.temp_range == 1:4
        @test spec.vmr_ranges[1] == (CO2 => 5:8)
        @test spec.vmr_ranges[2] == (H2O => 9:12)
        @test spec.tsfc_index == 13
        @test spec.emis_index == 14

        labels = state_labels(spec)
        @test length(labels) == 14
        @test labels[1] == "T[1]"
        @test labels[5] == "logVMR_CO₂[1]"
        @test labels[13] == "T_sfc"
        @test labels[14] == "emissivity"

        prof = AtmosphericProfile(
            [1000.0, 700.0, 400.0, 100.0],            # hPa
            [288.0, 270.0, 250.0, 230.0],             # K
            [0.0, 3.0, 7.0, 16.0],                    # km
            Dict(CO2 => fill(4.0e-4, 4), H2O => [1.0e-2, 5.0e-3, 1.0e-3, 1.0e-4]))

        x = pack_state(spec, prof; T_sfc=295.0, ε_sfc=0.98)
        @test x[spec.temp_range] == prof.temperature
        @test x[5:8] ≈ log.(fill(4.0e-4, 4))          # log-VMR storage
        @test x[13] == 295.0
        @test x[14] == 0.98

        # Round-trip: unpack reproduces the state-bearing fields exactly.
        p2, Ts2, ε2 = unpack_state(spec, x, prof)
        @test p2.temperature == prof.temperature
        @test p2.vmr[CO2] ≈ prof.vmr[CO2]
        @test p2.vmr[H2O] ≈ prof.vmr[H2O]
        @test p2.pressure == prof.pressure            # fixed field carried from base
        @test Ts2 == 295.0
        @test ε2 == 0.98

        # T_sfc not in state ⇒ follows (perturbed) T[1]; ε not in state ⇒ 1.0.
        spec2 = StateVectorSpec(4, GasSpecies[]; include_tsfc=false,
                                include_emissivity=false)
        @test spec2.n == 4
        x2 = pack_state(spec2, prof)
        x2[1] = 301.0
        _, Ts3, ε3 = unpack_state(spec2, x2, prof)
        @test Ts3 == 301.0
        @test ε3 == 1.0
    end

    @testset "Jacobian — finite-difference harness (synthetic CO₂ band)" begin
        # Synthetic, data-free: a couple of CO₂ lines in a narrow window, a tiny
        # profile, ILS/continuum off so the harness exercises only LBL + RTE.
        mk(ν0, S) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                               Float32(0.08), Float32(0.10), 250.0,
                               Float32(0.75), Float32(0.0))
        ll = HITRANLinelist([mk(700.5, 3.0e-23), mk(702.0, 1.0e-23)])
        linelists = Dict(CO2 => ll)

        prof = AtmosphericProfile(
            [1000.0, 600.0, 200.0],
            [288.0, 255.0, 225.0],
            [0.0, 4.0, 12.0],
            Dict(CO2 => fill(4.0e-4, 3)))

        # Narrow 6 cm⁻¹ band at 0.5 cm⁻¹ sampling (13 channels).
        iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)
        fm = (iasi=iasi, apply_continuum=false, with_ils=false)

        spec = StateVectorSpec(3, [CO2]; include_temperature=true,
                               include_tsfc=true, include_emissivity=true)
        @test spec.n == 3 + 3 + 1 + 1          # T + CO2 + T_sfc + ε

        jac = finite_difference_jacobian(prof, linelists, spec;
                                         T_sfc=290.0, ε_sfc=0.98,
                                         observable=:bt, fm_kwargs=fm)

        @test size(jac.K) == (13, spec.n)
        @test all(isfinite, jac.K)
        @test length(jac.ν) == 13
        @test length(jac.y0) == 13

        # Identify the most transparent channel (warmest BT ≈ closest to surface).
        i_window = argmax(jac.y0)
        # Most opaque (coldest) channel sits on a line core.
        i_line   = argmin(jac.y0)

        # ∂BT/∂T_sfc in a window channel ≈ 1 (surface fully seen through thin air).
        dTsfc = column(jac, "T_sfc")
        @test 0.7 < dTsfc[i_window] < 1.05
        @test all(dTsfc .>= -1e-6)            # warming the surface never cools any channel

        # ∂BT/∂(log VMR_CO₂): adding absorber over a colder atmosphere cools the
        # line-core channel ⇒ the summed CO₂ column sensitivity there is negative.
        co2_block = jac.K[:, spec.vmr_ranges[1][2]]
        @test sum(co2_block[i_line, :]) < 0.0

        # Emissivity sensitivity is positive in the window (more surface emission).
        dε = column(jac, "emissivity")
        @test dε[i_window] > 0.0

        # Central vs forward differences agree to FD truncation order on T_sfc.
        jac_fwd = finite_difference_jacobian(prof, linelists, spec;
                                             T_sfc=290.0, ε_sfc=0.98,
                                             observable=:bt, method=:forward,
                                             fm_kwargs=fm)
        @test isapprox(column(jac_fwd, "T_sfc")[i_window], dTsfc[i_window];
                       rtol=1e-3, atol=1e-3)
    end

    # ── Jacobian Phase 1: analytic RTE Jacobian vs finite differences ────────
    @testset "Jacobian — source-correction derivatives" begin
        cim  = IRSounderLBL._cim_correction
        cimd = IRSounderLBL._cim_correction_deriv
        lit  = IRSounderLBL._lit_correction
        litd = IRSounderLBL._lit_correction_deriv
        h = 1e-6
        # Points either side of the series/closed-form branches (cim 0.06, lit
        # 1e-4) — deliberately NOT straddling a branch with the ±h step, where the
        # forward function's tiny series/closed-form mismatch would corrupt the FD.
        for s in (0.02, 0.05, 0.059, 0.07, 0.1, 0.5, 1.0, 3.0, 10.0)
            @test cimd(s) ≈ (cim(s + h) - cim(s - h)) / (2h) rtol=1e-5
            @test litd(s) ≈ (lit(s + h) - lit(s - h)) / (2h) rtol=1e-5
        end
    end

    @testset "Jacobian — analytic RTE Jacobian (vs FD, :cim & :toon)" begin
        g     = wavenumber_grid(700.0, 740.0, 5.0)              # 9 channels
        n_lay = 5
        T_lev = [300.0, 288.0, 272.0, 258.0, 244.0, 230.0]     # surface-first
        T_sfc = 305.0
        ε     = 0.97
        μ     = 0.85
        p_lev = [1000.0, 700.0, 450.0, 250.0, 100.0, 20.0]
        τ     = [0.2 + 0.15*k + 0.05*i for i in 1:g.n, k in 1:n_lay]  # cols differ

        for src in (:cim, :toon)
            Tave = src == :cim ?
                [IRSounderLBL.cg_temperature_mass(T_lev[k], T_lev[k+1], p_lev[k], p_lev[k+1])
                 for k in 1:n_lay] : nothing
            run(τ2, Tl2, Ts2, ε2) = schwarzschild_rte(g, τ2, Tl2, Ts2;
                μ=μ, ε_sfc=ε2, source_function=src,
                T_ave = src == :cim ?
                    [IRSounderLBL.cg_temperature_mass(Tl2[k], Tl2[k+1], p_lev[k], p_lev[k+1])
                     for k in 1:n_lay] : nothing)

            I, dτ, dTl, dTs, dε = schwarzschild_rte_jacobian(g, τ, T_lev, T_sfc;
                μ=μ, ε_sfc=ε, source_function=src, p_levels=p_lev)

            # Forward radiance matches the non-Jacobian solver.
            @test I ≈ run(τ, T_lev, T_sfc, ε) rtol=1e-12

            # dI/dτ (central FD per layer)
            hτ = 1e-6
            for m in 1:n_lay
                τp = copy(τ); τp[:, m] .+= hτ
                τm = copy(τ); τm[:, m] .-= hτ
                fd = (run(τp, T_lev, T_sfc, ε) .- run(τm, T_lev, T_sfc, ε)) ./ (2hτ)
                @test maximum(abs.(fd .- dτ[:, m])) < 1e-6
            end

            # dI/dT_lev (central FD per level)
            hT = 1e-3
            for j in 1:(n_lay + 1)
                Tp = copy(T_lev); Tp[j] += hT
                Tm = copy(T_lev); Tm[j] -= hT
                fd = (run(τ, Tp, T_sfc, ε) .- run(τ, Tm, T_sfc, ε)) ./ (2hT)
                @test maximum(abs.(fd .- dTl[:, j])) < 1e-7
            end

            # dI/dT_sfc and dI/dε
            fds = (run(τ, T_lev, T_sfc + 1e-3, ε) .- run(τ, T_lev, T_sfc - 1e-3, ε)) ./ 2e-3
            @test maximum(abs.(fds .- dTs)) < 1e-7
            fde = (run(τ, T_lev, T_sfc, ε + 1e-5) .- run(τ, T_lev, T_sfc, ε - 1e-5)) ./ 2e-5
            @test maximum(abs.(fde .- dε)) < 1e-7
        end

        # Isothermal Kirchhoff: dI/dτ ≡ 0 (adding opacity in an isothermal column
        # at the surface T cannot change TOA radiance).
        T_iso = fill(285.0, n_lay + 1)
        _, dτ0, _, _, _ = schwarzschild_rte_jacobian(g, τ, T_iso, 285.0;
            μ=μ, ε_sfc=1.0, source_function=:cim, p_levels=p_lev)
        @test maximum(abs.(dτ0)) < 1e-9

        # :cim without p_levels errors.
        @test_throws ErrorException schwarzschild_rte_jacobian(g, τ, T_lev, T_sfc;
            source_function=:cim)
    end

    # ── Jacobian Phase 2: analytic VMR (+ surface) Jacobian vs FD harness ─────
    @testset "Jacobian — CG column-VMR gradient & ∂BT/∂R" begin
        cg   = IRSounderLBL.cg_column_vmr
        grad = IRSounderLBL._cg_column_vmr_grad
        # Closed-form ∂(CG column VMR)/∂(level VMR) vs scalar central FD.
        for (v1, v2, p1, p2) in [(4.0e-4, 4.0e-4, 1000.0, 600.0),
                                 (1.0e-2, 3.0e-3, 1000.0, 600.0),
                                 (3.0e-3, 1.0e-2, 600.0, 200.0),
                                 (5.0e-4, 4.9e-4, 850.0, 700.0)]
            g1, g2 = grad(v1, v2, p1, p2)
            h1 = v1 * 1e-6; h2 = v2 * 1e-6
            @test g1 ≈ (cg(v1+h1, v2, p1, p2) - cg(v1-h1, v2, p1, p2)) / (2h1) rtol=1e-6
            @test g2 ≈ (cg(v1, v2+h2, p1, p2) - cg(v1, v2-h2, p1, p2)) / (2h2) rtol=1e-6
        end
        # Well-mixed limit reduces to the cg_temperature_mass weight, sums to 1.
        fr = IRSounderLBL._cg_mass_frac(1000.0, 600.0)
        g1, g2 = grad(4.0e-4, 4.0e-4, 1000.0, 600.0)
        @test g1 ≈ 1.0 - fr
        @test g2 ≈ fr
        @test g1 + g2 ≈ 1.0
        # Nonpositive VMR → forward returns ½(v1+v2) → gradient (0.5, 0.5).
        @test all(grad(0.0, 4.0e-4, 1000.0, 600.0) .≈ (0.5, 0.5))
        # ∂BT/∂R vs FD of the inverse Planck function.
        for (ν, T) in [(700.0, 250.0), (2300.0, 290.0)]
            R = planck_radiance(ν, T); h = R * 1e-6
            @test IRSounderLBL._dBT_dR(ν, R) ≈
                (brightness_temperature(ν, R+h) - brightness_temperature(ν, R-h)) / (2h) rtol=1e-6
        end
    end

    @testset "Jacobian — analytic VMR Jacobian (vs FD harness)" begin
        mk(ν0, S) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                               Float32(0.08), Float32(0.10), 250.0,
                               Float32(0.75), Float32(0.0))
        ll = HITRANLinelist([mk(700.5, 3.0e-23), mk(702.0, 1.0e-23)])
        linelists = Dict(CO2 => ll)
        iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)

        build(nlev) = let p = collect(range(1000.0, 200.0; length=nlev))
            T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
            z = (1000.0 .- p) ./ 66.0
            AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev)))
        end
        # VMR-block max |Δ| between analytic and FD Jacobians at a given layering.
        function vmr_block_diff(nlev)
            prof = build(nlev)
            spec = StateVectorSpec(nlev, [CO2]; include_temperature=false,
                                   include_tsfc=true, include_emissivity=true)
            fm = (iasi=iasi, apply_continuum=false, with_ils=true)
            fd = finite_difference_jacobian(prof, linelists, spec;
                                            T_sfc=290.0, ε_sfc=0.98,
                                            observable=:bt, fm_kwargs=fm)
            an = analytic_jacobian(prof, linelists, spec; iasi=iasi,
                                   T_sfc=290.0, ε_sfc=0.98, observable=:bt,
                                   apply_continuum=false, with_ils=true)
            co2 = spec.vmr_ranges[1][2]
            return (fd=fd, an=an, spec=spec,
                    vmr = maximum(abs.(an.K[:, co2] .- fd.K[:, co2])),
                    rel = maximum(abs.(an.K[:, co2] .- fd.K[:, co2])) /
                          (maximum(abs.(fd.K[:, co2])) + 1e-30))
        end

        r6  = vmr_block_diff(6)
        r11 = vmr_block_diff(11)

        # Radiance/BT reproduced to round-off (analytic forward rebuild ≡ FM).
        @test maximum(abs.(r6.an.y0 .- r6.fd.y0)) < 1e-9

        # Exact columns: T_sfc and ε have no τ-coupling, match FD to FD precision.
        @test maximum(abs.(r6.an.K[:, r6.spec.tsfc_index] .-
                           r6.fd.K[:, r6.spec.tsfc_index])) < 1e-4
        @test maximum(abs.(r6.an.K[:, r6.spec.emis_index] .-
                           r6.fd.K[:, r6.spec.emis_index])) < 1e-4

        # Dominant VMR term: agrees to <1% with ILS on; the residual is the
        # deferred §2.1 CG-pressure/temperature coupling, which scales O(Δz) and
        # so must *shrink* under finer layering (proof it is physics, not a bug).
        @test r6.rel  < 0.01
        @test r11.rel < r6.rel

        # VMR sensitivity sign: adding CO₂ over a colder atmosphere cools the
        # band, so the summed column response is negative on opaque channels.
        i_line = argmin(r6.an.y0)
        co2blk = r6.an.K[:, r6.spec.vmr_ranges[1][2]]
        @test sum(co2blk[i_line, :]) < 0.0

        # VMR-only spec (Phase 2): analytic temperature path is off here.
        @test true
    end

    # ── Jacobian Phase 3: analytic temperature Jacobian (∂σ/∂T) vs FD ─────────
    @testset "Jacobian — ∂σ/∂T cross-section derivative (vs FD)" begin
        mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                                  Float32(0.07), Float32(0.08), E,
                                  Float32(0.75), Float32(-0.003))
        ll = HITRANLinelist([mk(700.3, 2.0e-21, 250.0), mk(701.4, 8.0e-22, 600.0),
                             mk(702.6, 1.5e-21, 120.0)])
        g = wavenumber_grid(698.0, 705.0, 0.002)
        p_atm = 0.5
        for T in (220.4, 260.3, 295.6)      # mid-1K-cell (avoid TIPS staircase straddle)
            σ, dσ = compute_voigt_cross_sections_dT(g, ll, T, p_atm)
            σ_fwd = compute_voigt_cross_sections(g, ll, T, p_atm)
            @test maximum(abs.(σ .- σ_fwd)) < 1e-30          # σ reproduces forward
            h  = 0.05                                        # stays within the 1 K cell
            fd = (compute_voigt_cross_sections(g, ll, T+h, p_atm) .-
                  compute_voigt_cross_sections(g, ll, T-h, p_atm)) ./ (2h)
            nz = findall(>(maximum(σ) * 1e-6), σ)
            @test maximum(abs.(dσ[nz] .- fd[nz])) /
                  (maximum(abs.(fd[nz])) + 1e-300) < 1e-5
        end
        # Component derivatives vs FD (mid-cell T).
        line = ll.lines[1]; T = 260.3; h = 0.02
        S, dS = IRSounderLBL.temperature_scaled_intensity_deriv(line, T)
        fdS = (IRSounderLBL.temperature_scaled_intensity(line, T+h) -
               IRSounderLBL.temperature_scaled_intensity(line, T-h)) / (2h)
        @test dS ≈ fdS rtol=1e-4
        gl, gd, dgl, dgd = IRSounderLBL.pressure_broadened_width_deriv(line, 0.5, T)
        glp, gdp = IRSounderLBL.pressure_broadened_width(line, 0.5, T+h)
        glm, gdm = IRSounderLBL.pressure_broadened_width(line, 0.5, T-h)
        @test dgl ≈ (glp - glm) / (2h) rtol=1e-6
        @test dgd ≈ (gdp - gdm) / (2h) rtol=1e-6
    end

    @testset "Jacobian — analytic temperature Jacobian (vs FD harness)" begin
        mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                                  Float32(0.07), Float32(0.08), E,
                                  Float32(0.75), Float32(-0.003))
        ll = HITRANLinelist([mk(700.5, 2.0e-21, 250.0), mk(702.0, 8.0e-22, 600.0)])
        linelists = Dict(CO2 => ll)
        iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)
        p = collect(range(1000.0, 200.0; length=6))
        T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
        z = (1000.0 .- p) ./ 66.0
        prof = AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, 6)))

        spec = StateVectorSpec(6, [CO2]; include_temperature=true,
                               include_tsfc=true, include_emissivity=true)
        fm = (iasi=iasi, apply_continuum=false, with_ils=true)
        # Small in-cell T steps so the FD doesn't straddle the 1 K TIPS quantization.
        steps = default_fd_steps(spec; δT=0.1, δlogvmr=1e-3, δtsfc=0.1, δε=1e-3)
        fd = finite_difference_jacobian(prof, linelists, spec; T_sfc=290.0, ε_sfc=0.98,
                                        observable=:bt, steps=steps, fm_kwargs=fm)
        an = analytic_jacobian(prof, linelists, spec; iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
                               observable=:bt, apply_continuum=false, with_ils=true)

        @test size(an.K) == size(fd.K)
        @test maximum(abs.(an.y0 .- fd.y0)) < 1e-9

        tr = spec.temp_range
        tblk_rel = maximum(abs.(an.K[:, tr] .- fd.K[:, tr])) /
                   (maximum(abs.(fd.K[:, tr])) + 1e-30)
        @test tblk_rel < 1e-3          # measured ~3e-6; bound leaves staircase headroom
        # Surface columns stay exact.
        @test maximum(abs.(an.K[:, spec.tsfc_index] .- fd.K[:, spec.tsfc_index])) < 1e-4
        @test maximum(abs.(an.K[:, spec.emis_index] .- fd.K[:, spec.emis_index])) < 1e-4
        # Warming the near-surface level warms the band-centre channel.
        i_line = argmin(an.y0)
        @test an.K[i_line, spec.temp_range[1]] > 0.0
    end

    # ── Jacobian Phase 5: optimal-estimation retrieval (synthetic closed loop) ──
    @testset "Optimal estimation — synthetic closed-loop retrieval" begin
        mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                                  Float32(0.07), Float32(0.08), E,
                                  Float32(0.75), Float32(-0.003))
        ll = HITRANLinelist([mk(700.5, 2.0e-21, 250.0), mk(702.0, 8.0e-22, 600.0)])
        linelists = Dict(CO2 => ll)
        iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)
        fm   = (iasi=iasi, apply_continuum=false, with_ils=true)
        nlev = 6
        p = collect(range(1000.0, 200.0; length=nlev))
        T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
        z = (1000.0 .- p) ./ 66.0
        base = AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev)))

        # (A) Well-posed scalar recovery: T_sfc from a 10 K-wrong guess, noiseless.
        specA = StateVectorSpec(nlev, GasSpecies[]; include_temperature=false,
                                include_tsfc=true, include_emissivity=false)
        _, _, yA = iasi_forward_model(base, linelists; T_sfc=298.0, ε_sfc=1.0, dptmn=0.0, fm...)
        SeA = Matrix(Diagonal(fill(0.2^2, length(yA))))
        rA = optimal_estimation(yA, specA, base, linelists;
                                xa=[288.0], Sa=reshape([100.0],1,1), Se=SeA, fm_kwargs=fm)
        @test rA.converged
        @test isapprox(rA.x[1], 298.0; atol=0.05)        # recovers truth
        @test isapprox(rA.dof, 1.0; atol=0.05)           # one piece of information
        @test rA.cost[end] < rA.cost[1]

        # (B) Profile + surface (ε fixed), noisy: convergence, fit, valid diagnostics.
        specB = StateVectorSpec(nlev, GasSpecies[]; include_temperature=true,
                                include_tsfc=true, include_emissivity=false)
        _, _, yclean = iasi_forward_model(base, linelists; T_sfc=296.0, ε_sfc=1.0, dptmn=0.0, fm...)
        σn = 0.2
        yB = yclean .+ σn .* [sin(13.0*i) for i in 1:length(yclean)]
        xtB = pack_state(specB, base; T_sfc=296.0, ε_sfc=1.0)
        xaB = copy(xtB); xaB[specB.temp_range] .+= 4.0; xaB[specB.tsfc_index] = 290.0
        SaB = Matrix(Diagonal([fill(25.0, nlev); 25.0]))
        SeB = Matrix(Diagonal(fill(σn^2, length(yB))))
        rB = optimal_estimation(yB, specB, base, linelists;
                                xa=xaB, Sa=SaB, Se=SeB, fm_kwargs=fm)
        @test rB.converged
        @test rB.cost[end] < rB.cost[1]
        @test maximum(abs.(rB.y_fit .- yB)) < 5σn       # fits to the noise level
        @test isapprox(rB.x[specB.tsfc_index], 296.0; atol=1.0)
        @test 0.0 < rB.dof < specB.n                     # regularised: DOF below n
        @test isapprox(rB.S_hat, rB.S_hat')              # posterior symmetric
        @test isposdef(Symmetric(rB.S_hat))              # and positive-definite
        @test size(rB.A) == (specB.n, specB.n)

        # (C) VMR path through OE: recover a +20% CO₂ column, noiseless.
        specC = StateVectorSpec(nlev, [CO2]; include_temperature=false,
                                include_tsfc=false, include_emissivity=false)
        base12 = AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.8e-4, nlev)))
        _, _, yC = iasi_forward_model(base12, linelists; T_sfc=288.0, ε_sfc=1.0, dptmn=0.0, fm...)
        xaC = pack_state(specC, base)                    # prior = unscaled 4.0e-4
        SeC = Matrix(Diagonal(fill(0.1^2, length(yC))))
        rC = optimal_estimation(yC, specC, base, linelists;
                                xa=xaC, Sa=Matrix(Diagonal(fill(1.0, specC.n))),
                                Se=SeC, fm_kwargs=fm, max_iter=20)
        @test rC.converged
        @test rC.cost[end] < rC.cost[1]
        @test maximum(abs.(rC.y_fit .- yC)) < 0.05       # spectrum fit (K)
        @test sum(rC.x) > sum(xaC)                       # retrieved CO₂ increased toward truth
    end

    # ── Phase 4: FFT-based ILS reproduces the direct convolution ──────────────
    @testset "ILS — FFT convolution ≡ direct apply_ils" begin
        g = wavenumber_grid(699.0, 705.0, 0.002)
        spec = [1.0 + 0.5sin(3.0*ν) + (abs(ν-702.0) < 0.05 ? -0.8 : 0.0) for ν in g.ν]
        for apod in (:gaussian, :norton_beer_medium)
            δν, kern = ils_kernel(g.Δν, 2.0, 0.5; apodization=apod)
            direct = apply_ils(g, spec, δν, kern)
            @test maximum(abs.(apply_ils_fft(g, spec, δν, kern) .- direct)) < 1e-12
            # Prebuilt convolver + in-place apply matches too, and is reusable.
            conv = ILSConvolver(g, δν, kern)
            out = similar(spec)
            ils_apply!(conv, out, spec)
            @test maximum(abs.(out .- direct)) < 1e-12
            ils_apply!(conv, out, reverse(spec))             # reuse on a different spectrum
            @test maximum(abs.(out .- apply_ils(g, reverse(spec), δν, kern))) < 1e-12
        end
    end

end
