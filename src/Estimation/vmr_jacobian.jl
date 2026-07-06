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

## Phase 2b: VMR coupling (`vmr_coupling=true`, default)

Beyond the dominant term, a level VMR also moves the layer's CG *pressure* and CG
*temperature* (which shift `σ`) and, for H₂O, its self-broadening fraction. These are
added analytically:

    ∂σ/∂p_cg·∂p_cg/∂VMR  +  ∂σ/∂T_cg·∂T_cg/∂VMR  +  ∂σ/∂vmr_self·∂vmr_self/∂VMR

via `compute_voigt_cross_sections_grad` (`σ`, `∂σ/∂T`, `∂σ/∂p`, `∂σ/∂vmr_self` in one
pass). `∂p_cg/∂VMR` is a central difference of the scalar `cg_pressure` (robust through
its `v1≈v2` branch); `∂T_cg/∂VMR = ∂T_cg/∂p_cg·∂p_cg/∂VMR` (zero for `:mass_weighted`,
whose `T_cg` is VMR-independent); `∂vmr_self/∂VMR = ∂vmr_cg/∂VMR` for H₂O (else 0).
This brings the VMR columns to FD precision (~1e-7), matching the T columns. Set
`vmr_coupling=false` for the faster dominant-only term (the residual is then the
coupling, ~few-% on coarse grids, O(Δz)). With line mixing active, `∂σ/∂{p,T}`
includes the LM perturbation (roadmap §6.4; see `_species_cross_section_grad`).

## Phase 3: temperature columns

`T_lev[j]` reaches the radiance through two paths, both filled when
`spec.include_temperature`:

  - the **source** `∂B/∂T` — Phase-1 `dI_dTlev`, CIM-consistent (already exact);
  - the **opacity** `∂σ/∂T_cg → ∂τ → dI/dτ` — `compute_voigt_cross_sections_dT`
    chained through `∂T_cg/∂T_lev` (the log-p interpolation weight, or the mass
    `frac` for `:mass_weighted`). `T_cg` enters only the two bounding levels, so
    level `j` collects from layers `j` (weight `1−wT`) and `j−1` (weight `wT`).

There is no `∂n/∂T` term: the LBL column amount `coef = vmr·Δp·N_air` is set by mass
(Δp), T-independent in pressure coordinates. With `line_mixing` active, `∂σ/∂T` adds
the central-differenced LM perturbation on top of the analytic Voigt baseline
(roadmap §6.4; `_species_cross_section_grad`).
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
                      source_function=:cim, vmr_coupling=true, backend=CPU()) -> Jacobian

Analytic Jacobian `K = ∂y/∂x` of the IASI forward model about the state packed from
`prof` (+ `T_sfc`, `ε_sfc`), returned as the same `Jacobian` struct as the FD harness
so the two are directly comparable. `observable` is `:bt` (default) or `:radiance`.

Populates the **VMR columns** (dominant number-density term + full CG-pressure/
temperature/self coupling when `vmr_coupling`; see file docstring), the
**temperature-level columns** (`∂B/∂T` source + `∂σ/∂T` opacity), the **`T_sfc`
column**, and the **`ε` column**. `line_mixing` (VP_Y/VP_W) is supported on every
column via `_species_cross_section_grad` (§6.4). The forward
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
                           lm_cutoff::Float64    = cutoff,
                           apply_continuum::Bool = true,
                           continua              = (:h2o, :co2, :co2_cia, :n2, :o2),
                           with_ils::Bool        = true,
                           apodization::Symbol   = :gaussian,
                           line_mixing::Union{Nothing, AbstractLineMixing} = nothing,
                           T_method::Symbol      = :logp_at_pcg,
                           source_function::Symbol = :cim,
                           vmr_coupling::Bool    = true,
                           backend               = CPU())::Jacobian
    observable in (:bt, :radiance) ||
        error("observable must be :bt or :radiance, got :$observable")
    need_T = spec.include_temperature
    # VMR coupling (∂σ/∂{p,T,vmr_self}·∂{p_cg,T_cg,vmr_self}/∂VMR) and the temperature
    # opacity term both go through `_species_cross_section_grad`, which is line-mixing
    # aware (roadmap §6.4): the Voigt baseline keeps its analytic grad and the additive
    # LM perturbation (VP_Y/VP_W) is central-differenced in (T, p).
    do_coupling = vmr_coupling
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
    ν_grid_hi = _internal_grid(iasi, Δν_hi, with_ils)
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
    retr_set = Set{GasSpecies}()
    for s in retrieved
        if haskey(linelists, s)
            τ_sp[s] = zeros(Float64, n_ν_hi, n_layers)
            push!(retr_set, s)
        end
    end
    dτdTcg = Dict{GasSpecies, Matrix{Float64}}()        # ∂τ/∂T_cg (all gases, T columns)
    wT     = Dict{GasSpecies, Vector{Float64}}()
    if need_T
        for s in keys(linelists)
            dτdTcg[s] = zeros(Float64, n_ν_hi, n_layers)
            wT[s]     = zeros(Float64, n_layers)
        end
    end
    # Phase-2b coupling: ∂τ/∂{p_cg,T_cg,vmr_self} for the RETRIEVED gases.
    dτdp  = Dict{GasSpecies, Matrix{Float64}}()
    dτdTc = Dict{GasSpecies, Matrix{Float64}}()
    dτdvs = Dict{GasSpecies, Matrix{Float64}}()
    if do_coupling
        for s in retr_set
            dτdp[s]  = zeros(Float64, n_ν_hi, n_layers)
            dτdTc[s] = zeros(Float64, n_ν_hi, n_layers)
            dτdvs[s] = zeros(Float64, n_ν_hi, n_layers)
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
            is_retr  = sp in retr_set
            if need_T || (do_coupling && is_retr)
                # σ and the needed derivatives. `_species_cross_section_grad` is the
                # LM-aware grad: plain Voigt (analytic) for non-LM species/models, and
                # Voigt baseline + central-FD'd LM perturbation when `line_mixing`
                # handles this species (roadmap §6.4). σ matches the forward exactly.
                # need_p: gr.dp is consumed only in the coupling block below, so
                # for e.g. a T(p)+T_sfc state the LM grad skips its 2 ∂p FD calls.
                gr = _species_cross_section_grad(line_mixing, sp, ν_grid_hi, ll,
                                                 T_cg_sp, p_cg_atm;
                                                 vmr_self=vmr_self, cutoff=cutoff, lm_cutoff=lm_cutoff,
                                                 backend=backend,
                                                 need_p=(do_coupling && is_retr))
                σ_sp = gr.σ
                if need_T
                    dτdTcg[sp][:, k] .= gr.dT .* coef
                    p1, p2 = prof.pressure[k], prof.pressure[k+1]
                    wT[sp][k] = T_method == :mass_weighted ? _cg_mass_frac(p1, p2) :
                                (log(layers.p_cg[sp][k] / p1) / log(p2 / p1))
                end
                if do_coupling && is_retr
                    # gr.dp is ∂σ/∂p in atm⁻¹; the p_cg weight below is in hPa, so
                    # convert to per-hPa (1 atm = 1013.25 hPa) for a consistent chain.
                    dτdp[sp][:, k]  .= gr.dp    .* (coef / 1013.25)
                    dτdTc[sp][:, k] .= gr.dT    .* coef
                    dτdvs[sp][:, k] .= gr.dself .* coef
                end
            else
                σ_sp = _species_cross_section(line_mixing, sp, ν_grid_hi, ll,
                                              T_cg_sp, p_cg_atm;
                                              vmr_self=vmr_self, cutoff=cutoff, lm_cutoff=lm_cutoff,
                                              backend=backend)
            end
            contrib = σ_sp .* coef
            τ_layers[:, k] .+= contrib
            is_retr && (τ_sp[sp][:, k] .= contrib)
        end
    end

    # Continuum/CIA layer optical depth: τ_cont = Σ(continua)·dz(T_mid), evaluated at
    # mid-layer VMR/T. Factored into a closure so the Phase-2c derivatives below
    # finite-difference the *same* expression that builds the forward τ.
    vmr_co2_default = 4.15e-4
    cont_layer(k, vh2o, vco2, Tm) = begin
        pm  = layers.p_mid[k]
        dz  = _dp_to_dz_cm(layers.Δp[k], pm, Tm)
        vdry = 1.0 - vh2o
        τc  = zeros(Float64, n_ν_hi)
        (:h2o     in continua) && (τc .+= h2o_continuum(ν_grid_hi, vh2o, pm, Tm) .* dz)
        (:co2     in continua) && (τc .+= co2_continuum(ν_grid_hi, vco2, pm, Tm) .* dz)
        (:co2_cia in continua) && (τc .+= co2_cia(ν_grid_hi, vco2, pm, Tm) .* dz)
        (:n2      in continua) && (τc .+= n2_cia(ν_grid_hi, 0.78084 * vdry, pm, Tm) .* dz)
        (:o2      in continua) && (τc .+= o2_cia(ν_grid_hi, 0.20946 * vdry, pm, Tm) .* dz)
        return τc
    end
    vmrmid(sp, k, default) = haskey(layers.vmr_mid, sp) ? layers.vmr_mid[sp][k] : default

    if apply_continuum
        for k in 1:n_layers
            τ_layers[:, k] .+= cont_layer(k, vmrmid(H2O, k, 0.0),
                                          vmrmid(CO2, k, vmr_co2_default), layers.T_mid[k])
        end
    end

    # ── Phase 2c: continuum/CIA derivatives (central FD of the cheap continuum) ──
    # The continuum depends on VMR (H₂O/CO₂, incl. n₂/o₂ via vmr_dry) and on T (its
    # own T-dependence + dz∝T). Central-difference cont_layer and chain through the
    # p_mid interpolation weight wM (∂{vmr,T}_mid/∂level). Only meaningful when
    # apply_continuum; VMR part is gated by do_coupling, T part by need_T.
    cont_vmr_h2o = apply_continuum && do_coupling && (H2O in retr_set)
    cont_vmr_co2 = apply_continuum && do_coupling && (CO2 in retr_set)
    cont_T       = apply_continuum && need_T
    dτc_dvh2o = cont_vmr_h2o ? zeros(Float64, n_ν_hi, n_layers) : nothing
    dτc_dvco2 = cont_vmr_co2 ? zeros(Float64, n_ν_hi, n_layers) : nothing
    dτc_dTm   = cont_T       ? zeros(Float64, n_ν_hi, n_layers) : nothing
    wM        = zeros(Float64, n_layers)        # ∂(mid quantity)/∂level[k+1]
    if cont_vmr_h2o || cont_vmr_co2 || cont_T
        for k in 1:n_layers
            wM[k] = log(layers.p_mid[k] / prof.pressure[k]) /
                    log(prof.pressure[k+1] / prof.pressure[k])
            vh = vmrmid(H2O, k, 0.0)
            vc = vmrmid(CO2, k, vmr_co2_default)
            Tm = layers.T_mid[k]
            if cont_vmr_h2o
                h = max(abs(vh), 1e-30) * 1e-6
                dτc_dvh2o[:, k] .= (cont_layer(k, vh+h, vc, Tm) .- cont_layer(k, vh-h, vc, Tm)) ./ (2h)
            end
            if cont_vmr_co2
                h = max(abs(vc), 1e-30) * 1e-6
                dτc_dvco2[:, k] .= (cont_layer(k, vh, vc+h, Tm) .- cont_layer(k, vh, vc-h, Tm)) ./ (2h)
            end
            if cont_T
                hT = 0.05
                dτc_dTm[:, k] .= (cont_layer(k, vh, vc, Tm+hT) .- cont_layer(k, vh, vc, Tm-hT)) ./ (2hT)
            end
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
    # The ILS convolution is the same fixed linear operator for every column, so we
    # build one FFT convolver (Phase 4) and reuse it — ~3000× faster than the direct
    # per-column convolution at the production grid, reproducing it to round-off.
    ν_iasi = wavenumber_grid(iasi.ν_min, iasi.ν_max, iasi.Δν)
    conv   = nothing
    if with_ils
        ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss;
                                      apodization=apodization)
        conv = ILSConvolver(ν_grid_hi, ils_δν, ils_kern)
    end
    apod_buf = Vector{Float64}(undef, n_ν_hi)   # reused ILS output buffer
    # I → R on channel grid (same path as the forward model) for the ∂BT/∂R diagonal.
    R_apod_I = with_ils ? ils_apply!(conv, apod_buf, I_hi) : I_hi
    R_iasi   = _resample_to_iasi(ν_grid_hi, R_apod_I, ν_iasi)
    n_ch     = ν_iasi.n
    dBTdR    = observable === :bt ?
               [_dBT_dR(ν_iasi.ν[c], R_iasi[c]) for c in 1:n_ch] :
               ones(Float64, n_ch)

    # Map a monochromatic derivative column (length n_ν_hi) to channel space.
    to_channel = function (dI_hi::Vector{Float64})
        a = with_ils ? ils_apply!(conv, apod_buf, dI_hi) : dI_hi
        r = _resample_to_iasi(ν_grid_hi, a, ν_iasi)
        observable === :bt ? (r .* dBTdR) : r
    end

    # ── 6. Assemble K ─────────────────────────────────────────────────────────
    x0 = pack_state(spec, prof; T_sfc=T_sfc, ε_sfc=ε_sfc)
    y0 = observable === :bt ? brightness_temperature(ν_iasi, R_iasi) : R_iasi
    K  = zeros(Float64, n_ch, spec.n)

    # VMR blocks. Level j enters layer k=j through its lower boundary and layer k=j−1
    # through its upper boundary. The dominant term is σ·∂vmr_cg/∂v (Phase 2); when
    # do_coupling, add ∂σ/∂{p_cg,T_cg,vmr_self}·∂{p_cg,T_cg,vmr_self}/∂v (Phase 2b).
    # Per layer we precompute the level-weight pairs of each CG quantity: vmr_cg
    # (analytic), p_cg (central difference of the scalar cg_pressure — robust through
    # its v1≈v2 branch), and T_cg = ∂T_cg/∂p_cg·∂p_cg/∂v (0 for :mass_weighted, whose
    # T_cg is VMR-independent). vmr_self moves only for H₂O (vmr_self = vmr_cg).
    dI_hi = Vector{Float64}(undef, n_ν_hi)
    for blk in spec.vmr_blocks
        s = blk.species
        haskey(τ_sp, s) || continue   # continuum-only species: dominant LBL term is 0
        r    = blk.range
        B    = blk.basis              # nothing ⇒ identity (per-level fast path)
        τs   = τ_sp[s]
        vcg  = layers.vmr_cg[s]
        vlev = prof.vmr[s]
        is_h2o = s == H2O
        # Reduced parameterization: accumulate ∂I/∂θ_m = Σ_j B[j,m]·∂I/∂(log v_j) in the
        # hi-res domain, then convolve M times (vs n_levels). ILS linearity makes the
        # projected-then-convolved column identical to convolving each level and summing.
        Hbuf = B === nothing ? nothing : [zeros(Float64, n_ν_hi) for _ in 1:blk.m]
        gV1 = Vector{Float64}(undef, n_layers); gV2 = similar(gV1)
        gP1 = zeros(Float64, n_layers); gP2 = zeros(Float64, n_layers)
        gT1 = zeros(Float64, n_layers); gT2 = zeros(Float64, n_layers)
        for k in 1:n_layers
            v1, v2 = Float64(vlev[k]), Float64(vlev[k+1])
            p1, p2 = Float64(prof.pressure[k]), Float64(prof.pressure[k+1])
            gV1[k], gV2[k] = _cg_column_vmr_grad(v1, v2, p1, p2)
            if do_coupling
                h1 = max(abs(v1), 1e-300) * 1e-6
                h2 = max(abs(v2), 1e-300) * 1e-6
                gP1[k] = (cg_pressure(v1+h1, v2, p1, p2) - cg_pressure(v1-h1, v2, p1, p2)) / (2h1)
                gP2[k] = (cg_pressure(v1, v2+h2, p1, p2) - cg_pressure(v1, v2-h2, p1, p2)) / (2h2)
                if T_method != :mass_weighted     # T_cg = T(p_cg); ∂T_cg/∂p_cg·∂p_cg/∂v
                    dTcg_dpcg = (Float64(prof.temperature[k+1]) - Float64(prof.temperature[k])) /
                                (layers.p_cg[s][k] * log(p2 / p1))
                    gT1[k] = dTcg_dpcg * gP1[k]
                    gT2[k] = dTcg_dpcg * gP2[k]
                end
            end
        end
        # Continuum/CIA VMR derivative for this species (Phase 2c; H₂O/CO₂ only),
        # chained through the p_mid interpolation weight wM.
        dτc_v = s == H2O ? dτc_dvh2o : (s == CO2 ? dτc_dvco2 : nothing)
        for j in 1:spec.n_levels
            fill!(dI_hi, 0.0)
            # layer k = j (level j is the lower boundary)
            if j <= n_layers && vcg[j] != 0.0
                @inbounds @views dI_hi .+= dI_dτ[:, j] .* τs[:, j] .* (gV1[j] / vcg[j])
                if do_coupling
                    @inbounds @views dI_hi .+= dI_dτ[:, j] .*
                        (dτdp[s][:, j] .* gP1[j] .+ dτdTc[s][:, j] .* gT1[j])
                    is_h2o && (@inbounds @views dI_hi .+= dI_dτ[:, j] .* dτdvs[s][:, j] .* gV1[j])
                end
            end
            # layer k = j−1 (level j is the upper boundary)
            if j >= 2 && vcg[j-1] != 0.0
                @inbounds @views dI_hi .+= dI_dτ[:, j-1] .* τs[:, j-1] .* (gV2[j-1] / vcg[j-1])
                if do_coupling
                    @inbounds @views dI_hi .+= dI_dτ[:, j-1] .*
                        (dτdp[s][:, j-1] .* gP2[j-1] .+ dτdTc[s][:, j-1] .* gT2[j-1])
                    is_h2o && (@inbounds @views dI_hi .+= dI_dτ[:, j-1] .* dτdvs[s][:, j-1] .* gV2[j-1])
                end
            end
            if dτc_v !== nothing      # continuum/CIA contribution (not gated by vcg)
                j <= n_layers && (@inbounds @views dI_hi .+= dI_dτ[:, j]   .* dτc_v[:, j]   .* (1.0 - wM[j]))
                j >= 2        && (@inbounds @views dI_hi .+= dI_dτ[:, j-1] .* dτc_v[:, j-1] .* wM[j-1])
            end
            if B === nothing
                colvec = to_channel(dI_hi)
                # log-VMR: ∂y/∂(log v) = v·∂y/∂v
                spec.log_vmr && (colvec .*= Float64(vlev[j]))
                @inbounds K[:, r[j]] .= colvec
            else
                # ∂I/∂(log v_j) = v_j·∂I/∂v_j (reduced params are always multiplicative-log).
                vj = Float64(vlev[j])
                @inbounds for m in 1:blk.m
                    w = B[j, m]
                    w == 0.0 && continue
                    @views Hbuf[m] .+= (w * vj) .* dI_hi
                end
            end
        end
        if B !== nothing
            @inbounds for m in 1:blk.m
                K[:, r[m]] .= to_channel(Hbuf[m])
            end
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
            if cont_T          # Phase 2c: continuum/CIA T-dependence (incl. dz∝T)
                j <= n_layers && (@inbounds @views dI_hi .+= dI_dτ[:, j]   .* dτc_dTm[:, j]   .* (1.0 - wM[j]))
                j >= 2        && (@inbounds @views dI_hi .+= dI_dτ[:, j-1] .* dτc_dTm[:, j-1] .* wM[j-1])
            end
            @inbounds K[:, spec.temp_range[j]] .= to_channel(dI_hi)
        end
    end

    # T_sfc and ε columns (exact, no τ-coupling).
    spec.include_tsfc       && (@inbounds K[:, spec.tsfc_index] .= to_channel(copy(dI_dTsfc)))
    spec.include_emissivity && (@inbounds K[:, spec.emis_index] .= to_channel(copy(dI_dε)))

    return Jacobian(K, spec, collect(Float64, ν_iasi.ν), x0, y0, observable)
end
