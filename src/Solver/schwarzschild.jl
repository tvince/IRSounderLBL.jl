"""
Layer-by-layer Schwarzschild radiative transfer equation solver.

For a plane-parallel, non-scattering atmosphere the exact solution is:

    I_TOA = B(ν, T_sfc) × T_sfc_to_TOA
           + Σ_k B(ν, T_k) × [T_{k+1}(ν) − T_k(ν)]

where T_k(ν) is the transmittance from level k to TOA.

Layers are numbered 1 (surface) to N_layers (near-TOA).
Levels are numbered 1 (surface) to N_levels = N_layers + 1 (TOA).

Reference: Clough, Iacono & Moncet (1992), JGR 97, 15761–15785.
"""

"""
    schwarzschild_rte(ν_grid, τ_layers, T_layers, T_sfc;
                      μ=1.0, ε_sfc=1.0) -> Vector{Float64}

Solve the Schwarzschild radiative transfer equation for a single column.

# Arguments
- `ν_grid`:   `WavenumberGrid`
- `τ_layers`: (n_ν × n_layers) matrix of layer optical depths
- `T_layers`: temperature at mid-layer (K), length n_layers
- `T_sfc`:    surface temperature (K)
- `μ`:        cos(viewing zenith angle) in [0,1], default 1.0 (nadir)
- `ε_sfc`:    surface emissivity (0–1), default 1.0 (blackbody)

# Returns
Top-of-atmosphere spectral radiance I_TOA (mW/m²/sr/cm⁻¹), length n_ν.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            T_layers::AbstractVector{<:Real},
                            T_sfc::Float64;
                            μ::Float64   = 1.0,
                            ε_sfc::Float64 = 1.0)::Vector{Float64}

    n_ν, n_layers = size(τ_layers)
    n_ν == ν_grid.n    || error("τ_layers rows $(n_ν) ≠ ν_grid.n $(ν_grid.n)")
    length(T_layers) == n_layers || error("T_layers length mismatch")
    0.0 < μ <= 1.0     || error("μ must be in (0, 1]")

    # Level transmittances: T_level[i, k] = transmittance from level k to TOA
    T_level = level_transmittances(τ_layers, μ)  # n_ν × (n_layers+1)

    # Surface contribution: ε B(sfc) × transmittance-from-surface-to-TOA
    I = Vector{Float64}(undef, n_ν)
    @inbounds @simd for i in 1:n_ν
        I[i] = ε_sfc * planck_radiance(ν_grid.ν[i], T_sfc) * T_level[i, 1]
    end

    # Atmospheric emission: layer k contributes B_k × ΔT_level_k
    # ΔT_level_k = T_level[:, k+1] − T_level[:, k]  (positive, layers absorb)
    @inbounds for k in 1:n_layers
        @simd for i in 1:n_ν
            Bk_i = planck_radiance(ν_grid.ν[i], T_layers[k])
            ΔT_k = T_level[i, k+1] - T_level[i, k]
            I[i] += Bk_i * ΔT_k
        end
    end

    return I
end

"""
    schwarzschild_rte(ν_grid, τ_layers, atm_profile;
                      T_sfc=nothing, μ=1.0, ε_sfc=1.0) -> Vector{Float64}

Convenience method accepting an `AtmosphericProfile`.  The surface temperature
defaults to the lowest atmospheric level.
"""
function schwarzschild_rte(ν_grid::WavenumberGrid,
                            τ_layers::AbstractMatrix{<:Real},
                            prof::AtmosphericProfile;
                            T_sfc::Union{Float64, Nothing} = nothing,
                            μ::Float64    = 1.0,
                            ε_sfc::Float64 = 1.0)
    layers = layer_properties(prof)
    Tsfc   = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    return schwarzschild_rte(ν_grid, τ_layers, layers.T_mid, Tsfc;
                             μ=μ, ε_sfc=ε_sfc)
end
