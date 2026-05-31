"""
Layer-by-layer Schwarzschild radiative transfer equation solver with
linear-in-τ (LIT) source function.

For a plane-parallel, non-scattering atmosphere the exact solution is:

    I_TOA = ε B(T_sfc) 𝒯_sfc
          + Σ_k ∫ B(T(t)) exp(−t) dt × 𝒯_top_k

where 𝒯_top_k is the transmittance from the top of layer k to TOA,
t is the path optical depth measured downward from the top of the layer,
and B(T(t)) varies linearly from B(T_top) to B(T_bot) within the layer.

Evaluating the integral analytically under the linear-B assumption gives the
LIT source function correction (Toon et al. 1989, J. Geophys. Res. 94, 16287):

    ΔI_k = B_top × ΔT_k + (B_bot − B_top) × 𝒯_top_k × f(s_k)

where s_k = τ_k/μ, ΔT_k = 𝒯_top_k − 𝒯_bot_k, and
    f(s) = (1 − exp(−s))/s − exp(−s)

In the thin-layer limit s → 0: f(s) → s/2, recovering the mid-point Planck
approximation B_mid × ΔT_k. In the thick-layer limit s → ∞: f(s) → 0, giving
B_top × 𝒯_top_k — emission from the top boundary only, which is correct since
optically thick layers radiate only from their upper surface.

Layers are numbered 1 (surface) to N_layers (near-TOA).
Levels are numbered 1 (surface) to N_levels = N_layers + 1 (TOA).
T_levels[k] is the temperature at the bottom of layer k (= top of layer k−1).
"""

# ── Source-function correction factors ───────────────────────────────────────

"""
    _lit_correction(s) -> Float64

Compute f(s) = (1 − exp(−s))/s − exp(−s), the Toon (1989) linear-in-τ
correction. Used by `source_function=:toon`: anchors emission at the two
level boundary temperatures.

Taylor for s < 1e-4: f(s) = s/2 − s²/3 + s³/8 − …
"""
@inline function _lit_correction(s::Float64)::Float64
    s < 1e-4 && return s * (0.5 - s * (1.0/3.0 - s * 0.125))
    ems = exp(-s)
    return (1.0 - ems) / s - ems
end

"""
    _cim_correction(s) -> Float64

Compute f(s) = 1 − 2·(exp(−s)/(exp(−s)−1) + 1/s), the Clough–Iacono–Moncet
Padé-rational correction used by LBLRTM (`source_function=:cim`).  Anchors
emission at the layer mass-weighted T_AVE for thin τ and at the upper-level
T_top for saturated τ.  Same form as LBLRTM/src/xmerge.f90 EMIN, line 1867.

Taylor for s < `od_lo` ≈ 0.06: f(s) ≈ s/6 − s³/360 + …
"""
@inline function _cim_correction(s::Float64)::Float64
    s < 0.06 && return s / 6.0   # Series; matches LBLRTM's rec_6 * odvi exactly
    ems = exp(-s)
    return 1.0 - 2.0 * (ems / (ems - 1.0) + 1.0 / s)
end

# ── Primary solver ────────────────────────────────────────────────────────────

"""
    schwarzschild_rte(ν_grid, τ_layers, T_levels, T_sfc;
                      μ=1.0, ε_sfc=1.0,
                      source_function=:toon, T_ave=nothing) -> Vector{Float64}

Solve the Schwarzschild RTE with a per-layer source-function approximation.

# Arguments
- `ν_grid`:   `WavenumberGrid`
- `τ_layers`: (n_ν × n_layers) matrix of layer optical depths
- `T_levels`: temperatures at the **n_layers + 1 level boundaries** (K),
              ordered surface-first. T_levels[k] is the bottom boundary of
              layer k, T_levels[k+1] is its top boundary.
- `T_sfc`:    surface skin temperature (K); may differ from T_levels[1]

# Keyword arguments
- `μ`:        cos(viewing zenith angle) in (0, 1], default 1.0 (nadir)
- `ε_sfc`:    surface emissivity (0–1), default 1.0
- `source_function`:
    - `:toon` (default): Toon (1989) linear-in-τ. Per-layer emission anchored
      at the two level boundary T's `(T_bot, T_top)`:
            em = B(T_top)·(1 − e^{−s}) + (B(T_bot) − B(T_top))·C_toon(s)
    - `:cim`: Clough–Iacono–Moncet Padé as used in LBLRTM (xmerge.f90 EMIN).
      Per-layer emission anchored at the Curtis–Godson layer-mean `T_AVE` and
      the upper-level `T_top`:
            em = (1 − e^{−s})·(B(T_AVE) + (B(T_top) − B(T_AVE))·C_cim(s))
      Saturates to B(T_top) and reduces to s·B(T_AVE) in the thin limit, so
      it is consistent with using `T_AVE = ∫T dp/Δp` for both opacity and
      source on the same layer (CG-consistent). Requires `T_ave`.
- `T_ave`:    Length-n_layers vector of per-layer mass-weighted T (K), needed
              when `source_function=:cim`. See `cg_temperature_mass` in
              `src/Atmosphere/layers.jl`.

# Returns
Top-of-atmosphere spectral radiance (mW/m²/sr/cm⁻¹), length n_ν.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            T_levels::AbstractVector{<:Real},
                            T_sfc::Float64;
                            μ::Float64    = 1.0,
                            ε_sfc::Float64 = 1.0,
                            source_function::Symbol = :toon,
                            T_ave::Union{AbstractVector{<:Real}, Nothing} = nothing
                            )::Vector{Float64}

    n_ν, n_layers = size(τ_layers)
    n_levels = n_layers + 1
    n_ν == ν_grid.n              || error("τ_layers rows $n_ν ≠ ν_grid.n $(ν_grid.n)")
    length(T_levels) == n_levels || error("T_levels must have n_layers+1 = $n_levels entries")
    0.0 < μ <= 1.0               || error("μ must be in (0, 1]")
    source_function in (:toon, :cim) ||
        error("source_function must be :toon or :cim, got :$source_function")
    if source_function == :cim
        isnothing(T_ave) && error(":cim source_function requires T_ave keyword")
        length(T_ave) == n_layers ||
            error("T_ave must have n_layers = $n_layers entries, got $(length(T_ave))")
    end

    𝒯 = level_transmittances(τ_layers, μ)   # (n_ν × n_levels)

    # Surface contribution
    I = Vector{Float64}(undef, n_ν)
    @inbounds @simd for i in 1:n_ν
        I[i] = ε_sfc * planck_radiance(ν_grid.ν[i], T_sfc) * 𝒯[i, 1]
    end

    if source_function == :toon
        @inbounds for k in 1:n_layers
            T_bot = Float64(T_levels[k])
            T_top = Float64(T_levels[k + 1])
            @simd for i in 1:n_ν
                B_bot = planck_radiance(ν_grid.ν[i], T_bot)
                B_top = planck_radiance(ν_grid.ν[i], T_top)
                s     = τ_layers[i, k] / μ
                ΔT_k  = 𝒯[i, k + 1] - 𝒯[i, k]
                C_k   = 𝒯[i, k + 1] * _lit_correction(s)
                I[i] += B_top * ΔT_k + (B_bot - B_top) * C_k
            end
        end
    else  # :cim — LBLRTM Padé using T_AVE for opacity-consistent emission
        # LBLRTM EMIN line 1871:  em = (1−tr)·(B_avg + (B_top−B_avg)·f_i)
        # so BOTH the B_avg term AND the (B_top−B_avg) term carry (1−tr).
        # When folded into the per-layer accumulator I += em · 𝒯[k+1] :
        #   I += 𝒯[k+1]·(1−tr)·B_avg  +  𝒯[k+1]·(1−tr)·f_i·(B_top−B_avg)
        # The first term is just B_avg·ΔT_k (since ΔT_k = 𝒯[k+1]·(1−tr) already).
        # The second term needs the explicit (1−tr) factor in C_k.
        @inbounds for k in 1:n_layers
            T_avg = Float64(T_ave[k])
            T_top = Float64(T_levels[k + 1])
            @simd for i in 1:n_ν
                B_avg = planck_radiance(ν_grid.ν[i], T_avg)
                B_top = planck_radiance(ν_grid.ν[i], T_top)
                s     = τ_layers[i, k] / μ
                ΔT_k  = 𝒯[i, k + 1] - 𝒯[i, k]                      # = 𝒯[k+1]·(1−e^{−s})
                C_k   = ΔT_k * _cim_correction(s)                   # = 𝒯[k+1]·(1−tr)·f_cim
                I[i] += B_avg * ΔT_k + (B_top - B_avg) * C_k
            end
        end
    end

    return I
end

# ── Convenience method ────────────────────────────────────────────────────────

"""
    schwarzschild_rte(ν_grid, τ_layers, prof;
                      T_sfc=nothing, μ=1.0, ε_sfc=1.0,
                      source_function=:toon, T_ave=nothing) -> Vector{Float64}

Convenience overload accepting an `AtmosphericProfile`. Level-boundary
temperatures are taken directly from `prof.temperature`. The surface skin
temperature defaults to `prof.temperature[1]`.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            prof::AtmosphericProfile;
                            T_sfc::Union{Float64, Nothing} = nothing,
                            μ::Float64    = 1.0,
                            ε_sfc::Float64 = 1.0,
                            source_function::Symbol = :toon,
                            T_ave::Union{AbstractVector{<:Real}, Nothing} = nothing)
    Tsfc = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    return schwarzschild_rte(ν_grid, τ_layers, prof.temperature, Tsfc;
                             μ=μ, ε_sfc=ε_sfc,
                             source_function=source_function, T_ave=T_ave)
end
