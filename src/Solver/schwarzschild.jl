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

# ── LIT correction factor ─────────────────────────────────────────────────────

"""
    _lit_correction(s) -> Float64

Compute f(s) = (1 − exp(−s))/s − exp(−s), the linear-in-τ source function
correction factor. Uses a Taylor series for s < 1e-4 to avoid catastrophic
cancellation between the two terms (each ≈ 1 for small s).

Taylor: f(s) = s/2 − s²/3 + s³/8 − ...
"""
@inline function _lit_correction(s::Float64)::Float64
    s < 1e-4 && return s * (0.5 - s * (1.0/3.0 - s * 0.125))
    ems = exp(-s)
    return (1.0 - ems) / s - ems
end

# ── Primary solver ────────────────────────────────────────────────────────────

"""
    schwarzschild_rte(ν_grid, τ_layers, T_levels, T_sfc;
                      μ=1.0, ε_sfc=1.0) -> Vector{Float64}

Solve the Schwarzschild RTE with a linear-in-τ source function.

# Arguments
- `ν_grid`:   `WavenumberGrid`
- `τ_layers`: (n_ν × n_layers) matrix of layer optical depths
- `T_levels`: temperatures at the **n_layers + 1 level boundaries** (K),
              ordered surface-first. T_levels[k] is the bottom boundary of
              layer k, T_levels[k+1] is its top boundary.
- `T_sfc`:    surface skin temperature (K); may differ from T_levels[1]
- `μ`:        cos(viewing zenith angle) in (0, 1], default 1.0 (nadir)
- `ε_sfc`:    surface emissivity (0–1), default 1.0

# Returns
Top-of-atmosphere spectral radiance (mW/m²/sr/cm⁻¹), length n_ν.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            T_levels::AbstractVector{<:Real},
                            T_sfc::Float64;
                            μ::Float64    = 1.0,
                            ε_sfc::Float64 = 1.0)::Vector{Float64}

    n_ν, n_layers = size(τ_layers)
    n_levels = n_layers + 1
    n_ν == ν_grid.n              || error("τ_layers rows $n_ν ≠ ν_grid.n $(ν_grid.n)")
    length(T_levels) == n_levels || error("T_levels must have n_layers+1 = $n_levels entries")
    0.0 < μ <= 1.0               || error("μ must be in (0, 1]")

    𝒯 = level_transmittances(τ_layers, μ)   # (n_ν × n_levels)

    # Surface contribution
    I = Vector{Float64}(undef, n_ν)
    @inbounds @simd for i in 1:n_ν
        I[i] = ε_sfc * planck_radiance(ν_grid.ν[i], T_sfc) * 𝒯[i, 1]
    end

    # Atmospheric emission with linear-in-τ source function.
    # Layer k: bottom boundary at level k (T_levels[k]), top at level k+1 (T_levels[k+1]).
    @inbounds for k in 1:n_layers
        T_bot = Float64(T_levels[k])
        T_top = Float64(T_levels[k + 1])
        @simd for i in 1:n_ν
            B_bot = planck_radiance(ν_grid.ν[i], T_bot)
            B_top = planck_radiance(ν_grid.ν[i], T_top)
            s     = τ_layers[i, k] / μ
            ΔT_k  = 𝒯[i, k + 1] - 𝒯[i, k]          # fraction absorbed by layer k
            C_k   = 𝒯[i, k + 1] * _lit_correction(s) # LIT weight
            I[i] += B_top * ΔT_k + (B_bot - B_top) * C_k
        end
    end

    return I
end

# ── Convenience method ────────────────────────────────────────────────────────

"""
    schwarzschild_rte(ν_grid, τ_layers, prof;
                      T_sfc=nothing, μ=1.0, ε_sfc=1.0) -> Vector{Float64}

Convenience overload accepting an `AtmosphericProfile`. Level-boundary
temperatures are taken directly from `prof.temperature`. The surface skin
temperature defaults to `prof.temperature[1]`.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            prof::AtmosphericProfile;
                            T_sfc::Union{Float64, Nothing} = nothing,
                            μ::Float64    = 1.0,
                            ε_sfc::Float64 = 1.0)
    Tsfc = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    return schwarzschild_rte(ν_grid, τ_layers, prof.temperature, Tsfc;
                             μ=μ, ε_sfc=ε_sfc)
end
