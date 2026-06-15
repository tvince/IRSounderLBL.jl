"""
Analytic VMR (and surface) Jacobian of the IASI forward model (roadmap Phase 2).

Composes the Phase-1 analytic RTE Jacobian `∂I/∂τ` with the **dominant** layer
optical-depth response `∂τ/∂VMR`, then pushes each monochromatic derivative column
through the *linear* channel-space tail — ILS convolution, resample to the sensor
grid, and the diagonal inverse-Planck `∂BT/∂R` — to produce `K = ∂y/∂x` columns in
the same units and ordering as the Phase-0 finite-difference harness.

## What is exact here vs. deferred

`τ_layers[ν,k] = Σ_s σ_s(ν;T_cg,p_cg,vmr_self)·vmr_cg_s[k]·Δp_k·N_air + continuum(vmr_mid)`.
The retrieval state perturbs VMR on the **levels**; a level VMR enters a layer's CG
column amount, and *also* its CG pressure / CG temperature (which shift `σ`), its
self-broadening (H₂O), and the continuum (quadratic in H₂O/CO₂ VMR).

Phase 2 implements the **dominant number-density term** for line-by-line absorbers:

    ∂τ_layers[ν,k]/∂vmr_cg_s[k] = σ_s(ν,k)·Δp_k·N_air = τ_s(ν,k)/vmr_cg_s[k]

chained to the level state through `∂vmr_cg_s[k]/∂VMR_level_j` (`_cg_column_vmr_grad`),
which is exact (it reduces to the `cg_temperature_mass` weight `frac` at the well-mixed
point). The **`T_sfc` and `ε` columns are fully exact** (Phase-1 `∂I/∂T_sfc`, `∂I/∂ε`
have no τ-coupling).

DEFERRED to "Phase 2b" (needs `∂σ/∂p` and the H₂O self term — see roadmap §2.1):
the CG-*pressure* coupling `∂σ/∂p·∂p_cg/∂VMR`, H₂O self-broadening `∂σ/∂vmr_self`,
and the quadratic continuum/CIA term. For a clean well-mixed LBL absorber with
continuum off these are second-order; the FD harness validates how much they matter
(`test/runtests.jl`).

## Phase 3: temperature columns

`T_lev[j]` reaches the radiance through two paths, both filled when
`spec.include_temperature`:

  - the **source** `∂B/∂T` — Phase-1 `dI_dTlev`, CIM-consistent (already exact);
  - the **opacity** `∂σ/∂T_cg → ∂τ → dI/dτ` — `compute_voigt_cross_sections_dT`
    chained through `∂T_cg/∂T_lev` (the log-p interpolation weight, or the mass
    `frac` for `:mass_weighted`). `T_cg` enters only the two bounding levels, so
    level `j` collects from layers `j` (weight `1−wT`) and `j−1` (weight `wT`).

There is no `∂n/∂T` term: the LBL column amount `coef = vmr·Δp·N_air` is set by mass
(Δp), T-independent in pressure coordinates. `∂σ/∂T` with line mixing is deferred
(roadmap §6.4), so `include_temperature` + `line_mixing` is rejected.
"""

# ── ∂(CG column VMR)/∂(level VMR) ──────────────────────────────────────────────

"""
    _cg_column_vmr_grad(v1, v2, p1, p2) -> (∂c/∂v1, ∂c/∂v2)

Gradient of `cg_column_vmr` w.r.t. its two bounding level VMRs, holding the
pressures fixed. Matches the *implemented* forward branches (roadmap Phase-1
discipline):

- nonpositive VMR → forward returns `½(v1+v2)` → gradient `(0.5, 0.5)`;
- `v1 ≈ v2` (well-mixed) → smooth limit `(1−frac, frac)` with `frac =
  _cg_mass_frac(p1,p2)` (the column integral of the log-p weight — identical to the
  `cg_temperature_mass` weight). NB the forward's degenerate `return v1` branch has
  naive gradient `(1,0)`; the smooth limit is the physically correct one and what a
  central finite difference recovers, so we use it;
- the `α+1 → 0` (VMR ∝ 1/p) corner → central difference of `cg_column_vmr` (rare,
  measure-zero gradient set);
- otherwise the closed form below, with `c = cg_column_vmr` and
  `J = ∫ φ(p)·VMR(p) dp`, `φ = ln(p/p1)/ln(p2/p1)`:
      `∂c/∂v1 = c/v1 − J/(v1·Δp)`,  `∂c/∂v2 = J/(v2·Δp)`.
"""
@inline function _cg_column_vmr_grad(v1::Float64, v2::Float64,
                                     p1::Float64, p2::Float64)
    (v1 <= 0.0 || v2 <= 0.0) && return (0.5, 0.5)
    if v1 ≈ v2
        fr = _cg_mass_frac(p1, p2)
        return (1.0 - fr, fr)
    end
    L  = log(p2 / p1)          # < 0  (p2 is the upper / lower-pressure boundary)
    α  = log(v2 / v1) / L
    β  = α + 1.0
    Δp = abs(p1 - p2)
    if abs(β) < 1e-6
        # VMR ∝ 1/p corner: fall back to a symmetric difference of the forward map.
        h1 = max(abs(v1), 1e-300) * 1e-6
        h2 = max(abs(v2), 1e-300) * 1e-6
        d1 = (cg_column_vmr(v1 + h1, v2, p1, p2) - cg_column_vmr(v1 - h1, v2, p1, p2)) / (2h1)
        d2 = (cg_column_vmr(v1, v2 + h2, p1, p2) - cg_column_vmr(v1, v2 - h2, p1, p2)) / (2h2)
        return (d1, d2)
    end
    r  = p2 / p1
    rβ = r^β
    c  = v1 * p1 * (1.0 - rβ) / (β * Δp)
    J  = (v1 * p1 / L) * ((rβ - 1.0) / β^2 - rβ * L / β)
    dc_dv1 = c / v1 - J / (v1 * Δp)
    dc_dv2 = J / (v2 * Δp)
    return (dc_dv1, dc_dv2)
end

# ── Inverse-Planck derivative (channel-space diagonal) ──────────────────────────

"""
    _dBT_dR(ν, R) -> Float64

`∂T_B/∂R` of the inverse Planck function `T_B = C2·ν / ln(1 + C1·ν³/R)`:

    ∂T_B/∂R = C2·ν·A / (L²·R·(R+A)),   A = C1·ν³,  L = ln(1 + A/R).

The diagonal that maps a radiance Jacobian column to a brightness-temperature one.
"""
@inline function _dBT_dR(ν::Float64, R::Float64)::Float64
    R <= 0.0 && return 0.0
    A = C1 * ν^3
    L = log1p(A / R)
    return C2 * ν * A / (L^2 * R * (R + A))
end

# ── Main entry ─────────────────────────────────────────────────────────────────

"""
    analytic_jacobian(prof, linelists, spec;
                      iasi=IASIInstrument(), geom=nadir_geometry(),
                      T_sfc=nothing, ε_sfc=1.0, observable=:bt,
                      internal_dnu=0.001, high_res_factor=nothing,
                      cutoff=25.0, apply_continuum=true,
                      continua=(:h2o,:co2,:co2_cia,:n2,:o2),
                      with_ils=true, apodization=:gaussian,
                      line_mixing=nothing, T_method=:logp_at_pcg,
                      source_function=:cim, backend=CPU()) -> Jacobian

Analytic Jacobian `K = ∂y/∂x` of the IASI forward model about the state packed from
`prof` (+ `T_sfc`, `ε_sfc`), returned as the same `Jacobian` struct as the FD harness
so the two are directly comparable. `observable` is `:bt` (default) or `:radiance`.

Populates the **VMR columns** (dominant number-density term; see file docstring),
the **temperature-level columns** (`∂B/∂T` source + `∂σ/∂T` opacity; requires
`line_mixing=nothing`), the **`T_sfc` column**, and the **`ε` column**. The forward
configuration mirrors `iasi_forward_model` and is run with the cutoff-freezing policy
(`dptmn=0.0`, full line set) so it matches `finite_difference_jacobian`.
"""
function analytic_jacobian(prof::AtmosphericProfile,
                           linelists::Dict{GasSpecies, HITRANLinelist},
                           spec::StateVectorSpec;
                           iasi::IASIInstrument  = IASIInstrument(),
                           geom::ViewingGeometry = nadir_geometry(),
                           T_sfc::Union{Float64, Nothing} = nothing,
                           ε_sfc::Float64        = 1.0,
                           observable::Symbol    = :bt,
                           internal_dnu::Union{Float64, Nothing} = 0.001,
                           high_res_factor::Union{Int, Nothing}  = nothing,
                           cutoff::Float64       = 25.0,
                           apply_continuum::Bool = true,
                           continua              = (:h2o, :co2, :co2_cia, :n2, :o2),
                           with_ils::Bool        = true,
                           apodization::Symbol   = :gaussian,
                           line_mixing::Union{Nothing, AbstractLineMixing} = nothing,
                           T_method::Symbol      = :logp_at_pcg,
                           source_function::Symbol = :cim,
                           backend               = CPU())::Jacobian
    observable in (:bt, :radiance) ||
        error("observable must be :bt or :radiance, got :$observable")
    need_T = spec.include_temperature
    (need_T && line_mixing !== nothing) &&
        error("analytic_jacobian: ∂σ/∂T with line mixing is deferred (roadmap §6.4); " *
              "pass line_mixing=nothing or build the spec with include_temperature=false.")
    length(prof.temperature) == spec.n_levels ||
        error("profile has $(length(prof.temperature)) levels, spec expects $(spec.n_levels)")

    # ── 1. Internal grid (mirror iasi_forward_model) ─────────────────────────
    hrf = if high_res_factor !== nothing
        high_res_factor
    elseif internal_dnu !== nothing
        max(1, ceil(Int, iasi.Δν / internal_dnu))
    else
        4
    end
    Δν_hi = iasi.Δν / hrf
    ν_grid_hi = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)
    n_ν_hi = ν_grid_hi.n

    # ── 2. Layer properties ──────────────────────────────────────────────────
    layers   = layer_properties(prof; T_method=T_method)
    n_layers = length(layers.p_mid)
    Nair_per_vmr = 2.1209e22   # molec/(cm²·hPa)

    retrieved = [s for (s, _) in spec.vmr_ranges]

    # ── 3. Optical-depth cube + per-species linearization quantities ──────────
    # τ_sp:    LBL τ contributed by each RETRIEVED gas (for ∂τ/∂VMR).
    # dτdTcg:  ∂τ_sp/∂T_cg = (∂σ/∂T)·coef per layer for EVERY gas (for ∂τ/∂T_lev).
    # wT:      ∂T_cg_sp[k]/∂T_lev[k+1] (weight on the upper level); the lower-level
    #          weight is 1−wT. log-p interpolation weight (:logp_at_pcg) or the
    #          mass frac (:mass_weighted) — matches layer_properties exactly.
    τ_layers = zeros(Float64, n_ν_hi, n_layers)
    τ_sp = Dict{GasSpecies, Matrix{Float64}}()
    for s in retrieved
        haskey(linelists, s) && (τ_sp[s] = zeros(Float64, n_ν_hi, n_layers))
    end
    dτdTcg = Dict{GasSpecies, Matrix{Float64}}()
    wT     = Dict{GasSpecies, Vector{Float64}}()
    if need_T
        for s in keys(linelists)
            dτdTcg[s] = zeros(Float64, n_ν_hi, n_layers)
            wT[s]     = zeros(Float64, n_layers)
        end
    end

    for k in 1:n_layers
        for (sp, ll) in linelists
            vmr = haskey(layers.vmr_cg, sp) ? layers.vmr_cg[sp][k] : 0.0
            vmr == 0.0 && continue
            p_cg_atm = layers.p_cg[sp][k] / 1013.25
            T_cg_sp  = layers.T_cg[sp][k]
            vmr_self = (sp == H2O) ? vmr : 0.0
            coef     = vmr * layers.Δp[k] * Nair_per_vmr
            if need_T
                # Plain-Voigt path (LM-∂T deferred): σ and ∂σ/∂T together.
                σ_sp, dσ_sp = compute_voigt_cross_sections_dT(ν_grid_hi, ll, T_cg_sp,
                                                              p_cg_atm; vmr_self=vmr_self,
                                                              cutoff=cutoff)
                dτdTcg[sp][:, k] .= dσ_sp .* coef
                p1, p2  = prof.pressure[k], prof.pressure[k+1]
                wT[sp][k] = T_method == :mass_weighted ? _cg_mass_frac(p1, p2) :
                            (log(layers.p_cg[sp][k] / p1) / log(p2 / p1))
            else
                σ_sp = _species_cross_section(line_mixing, sp, ν_grid_hi, ll,
                                              T_cg_sp, p_cg_atm;
                                              vmr_self=vmr_self, cutoff=cutoff, backend=backend)
            end
            contrib = σ_sp .* coef
            τ_layers[:, k] .+= contrib
            haskey(τ_sp, sp) && (τ_sp[sp][:, k] .= contrib)
        end
    end

    if apply_continuum
        for k in 1:n_layers
            vmr_h2o = haskey(layers.vmr_mid, H2O) ? layers.vmr_mid[H2O][k] : 0.0
            vmr_co2 = haskey(layers.vmr_mid, CO2) ? layers.vmr_mid[CO2][k] : 4.15e-4
            dz_cm   = _dp_to_dz_cm(layers.Δp[k], layers.p_mid[k], layers.T_mid[k])
            vmr_dry = 1.0 - vmr_h2o
            (:h2o     in continua) && (τ_layers[:, k] .+= h2o_continuum(ν_grid_hi, vmr_h2o, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:co2     in continua) && (τ_layers[:, k] .+= co2_continuum(ν_grid_hi, vmr_co2, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:co2_cia in continua) && (τ_layers[:, k] .+= co2_cia(ν_grid_hi, vmr_co2, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:n2      in continua) && (τ_layers[:, k] .+= n2_cia(ν_grid_hi, 0.78084 * vmr_dry, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:o2      in continua) && (τ_layers[:, k] .+= o2_cia(ν_grid_hi, 0.20946 * vmr_dry, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
        end
    end

    # ── 4. Analytic RTE Jacobian (Phase 1) ───────────────────────────────────
    Tsfc = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    p_lev = source_function == :cim ? prof.pressure : nothing
    I_hi, dI_dτ, dI_dTlev, dI_dTsfc, dI_dε =
        schwarzschild_rte_jacobian(ν_grid_hi, τ_layers, prof.temperature, Tsfc;
                                   μ=geom.μ, ε_sfc=ε_sfc,
                                   source_function=source_function, p_levels=p_lev)

    # ── 5. Channel-space linear tail, precomputed once ───────────────────────
    ν_iasi = wavenumber_grid(iasi.ν_min, iasi.ν_max, iasi.Δν)
    ils_δν = nothing; ils_kern = nothing
    if with_ils
        ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss;
                                      apodization=apodization)
    end
    # I → R on channel grid (same path as the forward model) for the ∂BT/∂R diagonal.
    R_apod_I = with_ils ? apply_ils(ν_grid_hi, I_hi, ils_δν, ils_kern) : I_hi
    R_iasi   = _resample_to_iasi(ν_grid_hi, R_apod_I, ν_iasi)
    n_ch     = ν_iasi.n
    dBTdR    = observable === :bt ?
               [_dBT_dR(ν_iasi.ν[c], R_iasi[c]) for c in 1:n_ch] :
               ones(Float64, n_ch)

    # Map a monochromatic derivative column (length n_ν_hi) to channel space.
    to_channel = function (dI_hi::Vector{Float64})
        a = with_ils ? apply_ils(ν_grid_hi, dI_hi, ils_δν, ils_kern) : dI_hi
        r = _resample_to_iasi(ν_grid_hi, a, ν_iasi)
        observable === :bt ? (r .* dBTdR) : r
    end

    # ── 6. Assemble K ─────────────────────────────────────────────────────────
    x0 = pack_state(spec, prof; T_sfc=T_sfc, ε_sfc=ε_sfc)
    y0 = observable === :bt ? brightness_temperature(ν_iasi, R_iasi) : R_iasi
    K  = zeros(Float64, n_ch, spec.n)

    # VMR blocks.  ∂vmr_cg_s[k]/∂VMR_level_j: layer k=j contributes via v1 (=level j),
    # layer k=j−1 via v2 (=level j).  Precompute the (g1[k], g2[k]) weights per layer.
    dI_hi = Vector{Float64}(undef, n_ν_hi)
    for (s, r) in spec.vmr_ranges
        haskey(τ_sp, s) || continue   # continuum-only species: dominant LBL term is 0
        τs   = τ_sp[s]
        vcg  = layers.vmr_cg[s]
        vlev = prof.vmr[s]
        g1 = Vector{Float64}(undef, n_layers)   # ∂vmr_cg[k]/∂v_level[k]
        g2 = Vector{Float64}(undef, n_layers)   # ∂vmr_cg[k]/∂v_level[k+1]
        for k in 1:n_layers
            g1[k], g2[k] = _cg_column_vmr_grad(Float64(vlev[k]), Float64(vlev[k+1]),
                                               Float64(prof.pressure[k]),
                                               Float64(prof.pressure[k+1]))
        end
        for j in 1:spec.n_levels
            col = r[j]
            fill!(dI_hi, 0.0)
            # layer k = j (level j is its lower boundary → weight g1[j])
            if j <= n_layers && vcg[j] != 0.0
                w = g1[j] / vcg[j]
                @inbounds @views dI_hi .+= dI_dτ[:, j] .* τs[:, j] .* w
            end
            # layer k = j−1 (level j is its upper boundary → weight g2[j−1])
            if j >= 2 && vcg[j-1] != 0.0
                w = g2[j-1] / vcg[j-1]
                @inbounds @views dI_hi .+= dI_dτ[:, j-1] .* τs[:, j-1] .* w
            end
            colvec = to_channel(dI_hi)
            # log-VMR: ∂y/∂(log v) = v·∂y/∂v
            spec.log_vmr && (colvec .*= Float64(vlev[j]))
            @inbounds K[:, col] .= colvec
        end
    end

    # Temperature levels: source part (Phase 1 dI_dTlev) + opacity part
    # (∂σ/∂T_cg chained through ∂T_cg/∂T_lev). Level j touches layers j (lower
    # boundary, weight 1−wT[j]) and j−1 (upper boundary, weight wT[j−1]).
    if need_T
        for j in 1:spec.n_levels
            @inbounds dI_hi .= @view dI_dTlev[:, j]
            for (sp, M) in dτdTcg
                w = wT[sp]
                if j <= n_layers
                    @inbounds @views dI_hi .+= dI_dτ[:, j] .* M[:, j] .* (1.0 - w[j])
                end
                if j >= 2
                    @inbounds @views dI_hi .+= dI_dτ[:, j-1] .* M[:, j-1] .* w[j-1]
                end
            end
            @inbounds K[:, spec.temp_range[j]] .= to_channel(dI_hi)
        end
    end

    # T_sfc and ε columns (exact, no τ-coupling).
    spec.include_tsfc       && (@inbounds K[:, spec.tsfc_index] .= to_channel(copy(dI_dTsfc)))
    spec.include_emissivity && (@inbounds K[:, spec.emis_index] .= to_channel(copy(dI_dε)))

    return Jacobian(K, spec, collect(Float64, ν_iasi.ν), x0, y0, observable)
end
