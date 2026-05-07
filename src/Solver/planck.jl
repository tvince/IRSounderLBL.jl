"""
Planck function and inverse Planck (brightness temperature).

Spectral radiance units: mW / (m² · sr · cm⁻¹)

Physical constants following HITRAN convention:
    C1 = 2hc² = 1.191042953e-5  mW/(m²·sr·cm⁻⁴)
    C2 = hc/k  = 1.4387769      cm·K
"""

const C1 = 1.191042953e-5   # mW/(m²·sr·cm⁻⁴)
const C2 = 1.4387769        # cm·K

"""
    planck_radiance(ν, T) -> Float64

Evaluate the Planck function B(ν, T) at wavenumber ν (cm⁻¹) and
temperature T (K).

Returns spectral radiance in mW / (m² · sr · cm⁻¹).
"""
@inline function planck_radiance(ν::Float64, T::Float64)::Float64
    return C1 * ν^3 / expm1(C2 * ν / T)
end

"""
    planck_radiance(ν_grid, T) -> Vector{Float64}

Evaluate B(ν, T) for all wavenumbers in `ν_grid` at temperature T (K).
"""
function planck_radiance(ν_grid::WavenumberGrid, T::Float64)::Vector{Float64}
    return [planck_radiance(ν, T) for ν in ν_grid.ν]
end

"""
    brightness_temperature(ν, R) -> Float64

Inverse Planck function: given spectral radiance R (mW/m²/sr/cm⁻¹) at
wavenumber ν (cm⁻¹), return the brightness temperature (K).

    T_B(ν, R) = C2 ν / ln(1 + C1 ν³ / R)
"""
@inline function brightness_temperature(ν::Float64, R::Float64)::Float64
    R <= 0.0 && return 0.0
    return C2 * ν / log1p(C1 * ν^3 / R)
end

"""
    brightness_temperature(ν_grid, R) -> Vector{Float64}

Convert radiance spectrum R [mW/m²/sr/cm⁻¹] to brightness temperature [K].
"""
function brightness_temperature(ν_grid::WavenumberGrid,
                                 R::AbstractVector{<:Real})::Vector{Float64}
    length(R) == ν_grid.n || error("R length $(length(R)) ≠ grid size $(ν_grid.n)")
    return [brightness_temperature(ν_grid.ν[i], R[i]) for i in 1:ν_grid.n]
end

"""
    dB_dT(ν, T) -> Float64

Jacobian of the Planck function with respect to temperature:
    ∂B/∂T = C1 C2 ν⁴ exp(C2 ν / T) / [T² (exp(C2 ν / T) − 1)²]

Useful for Jacobian calculations in retrieval algorithms.
"""
@inline function dB_dT(ν::Float64, T::Float64)::Float64
    x  = C2 * ν / T
    ex = exp(x)
    return C1 * C2 * ν^4 * ex / (T^2 * (ex - 1.0)^2)
end
