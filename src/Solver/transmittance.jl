"""
Level transmittances from layer optical depths.

Transmittance from level k to the top of atmosphere (TOA) is:
    T_k = exp(−Σ_{j=k}^{N} τ_j / μ)

where μ = cos(θ) is the cosine of the satellite viewing angle.
"""

"""
    level_transmittances(τ_layers, μ=1.0) -> Matrix{Float64}

Compute cumulative transmittances from each pressure level to TOA.

# Arguments
- `τ_layers`: Matrix of shape (n_ν, n_layers) giving optical depth of each layer
- `μ`:        cos(viewing zenith angle), default 1.0 (nadir)

# Returns
Matrix of shape (n_ν, n_levels) where `T[i, k]` is the transmittance
from level k to TOA at wavenumber index i.
Level 1 is surface, level n_levels is TOA.

Convention: level k is the *top* of layer k, so T[:, end] = 1 (TOA).
"""
function level_transmittances(τ_layers::AbstractMatrix{<:Real},
                               μ::Float64 = 1.0)::Matrix{Float64}
    n_ν, n_layers = size(τ_layers)
    n_levels = n_layers + 1
    T = Matrix{Float64}(undef, n_ν, n_levels)

    # TOA transmittance = 1
    T[:, n_levels] .= 1.0

    # Integrate downward from TOA
    @inbounds for k in n_layers:-1:1
        @simd for i in 1:n_ν
            T[i, k] = T[i, k+1] * exp(-τ_layers[i, k] / μ)
        end
    end

    return T
end

"""
    layer_transmittances(τ_layers, μ=1.0) -> Matrix{Float64}

Transmittance through each individual layer: t_layer = exp(−τ/μ).
Returns Matrix of shape (n_ν, n_layers).
"""
function layer_transmittances(τ_layers::AbstractMatrix{<:Real},
                               μ::Float64 = 1.0)::Matrix{Float64}
    return exp.(-τ_layers ./ μ)
end
