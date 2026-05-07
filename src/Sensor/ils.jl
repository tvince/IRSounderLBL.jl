"""
IASI Instrument Line Shape (ILS) — Norton-Beer apodization.

IASI is a Fourier Transform Spectrometer.  The raw (unapodized) ILS is a
sinc function.  The Norton-Beer apodization functions (Strong, Medium, Weak)
reduce side-lobes at the cost of slightly lower spectral resolution.

Reference:
  Norton & Beer (1976), JOSA 66, 259–264.
  Niro et al. (2005), J. Mol. Spectrosc. 230, 214–233.
"""

"""
Norton-Beer apodization coefficients for three standard strengths.

Each set {C0, C2, C4} satisfies C0 + C2 + C4 = 1.
The apodization function is:
    A(x) = C0 + C2 × (1 − x²)² + C4 × (1 − x²)⁴    |x| ≤ 1
"""
const NORTON_BEER_COEFFS = Dict(
    :norton_beer_weak   => (C0=0.384093, C2=0.087577, C4=0.528330),
    :norton_beer_medium => (C0=0.152442, C2=0.136176, C4=0.711382),
    :norton_beer_strong => (C0=0.045335, C2=0.554883, C4=0.399782),
)

"""
    norton_beer_apodization(x, style=:norton_beer_medium) -> Float64

Evaluate the Norton-Beer apodization function at normalized OPD position x
in [−1, 1].
"""
function norton_beer_apodization(x::Float64,
                                  style::Symbol = :norton_beer_medium)::Float64
    abs(x) > 1.0 && return 0.0
    c = NORTON_BEER_COEFFS[style]
    t = 1.0 - x^2
    return c.C0 + c.C2 * t^2 + c.C4 * t^4
end

"""
    ils_kernel(Δν, opd_max, fwhm_gauss; n_pts=2048) -> (δν_arr, ils)

Compute the ILS as the Fourier transform of a Gaussian-tapered rectangular
OPD window — i.e. the convolution of the natural sinc ILS with a Gaussian.

**Physical model**

The unapodized FTS ILS is sinc(2π opd_max δν), whose Fourier transform is
rect(L / opd_max).  Convolving with a Gaussian of width `fwhm_gauss` (cm⁻¹)
is equivalent in the OPD domain to multiplying by a Gaussian envelope:

    A_eff(L) = rect(L / opd_max) × exp(−2π² σ² L²)
    σ = fwhm_gauss / (2 √(2 ln 2))

The ILS is then recovered by numerical integration:

    ILS(δν) = 2 ∫₀^L_max A_eff(L) cos(2π L δν) dL

# Arguments
- `Δν`:         spectral grid spacing of the high-res internal grid (cm⁻¹)
- `opd_max`:    maximum OPD (cm); for IASI = 2.0
- `fwhm_gauss`: FWHM of the Gaussian broadening kernel (cm⁻¹); for IASI = 0.5
- `n_pts`:      number of OPD quadrature points
"""
function ils_kernel(Δν::Float64,
                    opd_max::Float64,
                    fwhm_gauss::Float64;
                    n_pts::Int = 2048)
    σ = fwhm_gauss / (2.0 * sqrt(2.0 * log(2.0)))   # Gaussian σ (cm⁻¹)

    # One-sided OPD quadrature grid
    opd = LinRange(0.0, opd_max, n_pts)
    dL  = opd_max / (n_pts - 1)

    # Effective OPD apodization: Gaussian envelope × rectangular window
    A = [exp(-2π^2 * σ^2 * L^2) for L in opd]

    # Spectral offset grid: cover ±16 cm⁻¹ at Δν spacing
    n_half = ceil(Int, 16.0 / Δν)
    δν_arr = collect((-n_half:n_half) .* Δν)
    n_ils  = length(δν_arr)

    # Numerical FT: ILS(δν) = 2 ∫₀^L_max A(L) cos(2π L δν) dL
    ils = Vector{Float64}(undef, n_ils)
    for (k, δν) in enumerate(δν_arr)
        ils[k] = 2.0 * sum(A .* cos.(2π .* opd .* δν)) * dL
    end

    # Area-normalise so that ∑ ILS × Δν = 1
    ils ./= (sum(ils) * Δν)

    return δν_arr, ils
end

"""
    apply_ils(ν_grid, spectrum, ils_δν, ils_kernel) -> Vector{Float64}

Convolve a monochromatic spectrum with the instrument line shape kernel
via direct convolution.

# Arguments
- `ν_grid`:     source `WavenumberGrid`
- `spectrum`:   high-resolution spectrum on `ν_grid`
- `ils_δν`:     wavenumber offset array of the ILS kernel (cm⁻¹)
- `ils_kernel`: ILS kernel values (normalized, sums to 1/Δν)

# Returns
Apodized spectrum on the same `ν_grid`.
"""
function apply_ils(ν_grid::WavenumberGrid,
                   spectrum::AbstractVector{<:Real},
                   ils_δν::AbstractVector{<:Real},
                   ils_kern::AbstractVector{<:Real})::Vector{Float64}
    length(spectrum) == ν_grid.n || error("spectrum length mismatch")
    Δν    = ν_grid.Δν
    n_ν   = ν_grid.n
    n_ils = length(ils_δν)
    half  = (n_ils - 1) ÷ 2

    out = zeros(Float64, n_ν)
    @inbounds for i in 1:n_ν
        acc = 0.0
        for k in 1:n_ils
            j = i + (k - 1 - half)
            (j < 1 || j > n_ν) && continue
            acc += spectrum[j] * ils_kern[k]
        end
        out[i] = acc * Δν   # integrate (kernel is normalized per cm⁻¹)
    end

    return out
end
