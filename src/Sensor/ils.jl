"""
Instrument Line Shape (ILS) for Fourier transform spectrometers.

The raw (unapodized) ILS is a sinc function.  Apodization tapers the
interferogram before transforming, trading spectral resolution for side-lobe
suppression.  Four tapers are available: Gaussian (the IASI L1C convention),
the three Norton-Beer recipes, and Hamming (the CrIS convention).

References:
  Norton & Beer (1976), JOSA 66, 259–264.
  Niro et al. (2005), J. Mol. Spectrosc. 230, 214–233.
  Blackman & Tukey (1958), *The Measurement of Power Spectra* — the 0.54/0.46
  raised cosine, universally called the Hamming window.
"""

"""
Norton-Beer apodization coefficients for three standard strengths.

Each set {C0, C2, C4} satisfies C0 + C2 + C4 = 1.
The apodization function is:
    A(x) = C0 + C2 × (1 − x²)² + C4 × (1 − x²)⁴    |x| ≤ 1
"""
# Half-width (cm⁻¹) of the truncated ILS kernel built by `ils_kernel`. The kernel
# is tabulated on ±ceil(ILS_HALFWIDTH_CM/Δν)·Δν; the same value sizes the internal-
# grid padding (`_internal_grid`) so every requested channel sees full kernel
# support after convolution. Keep the two in lock-step by referencing this const.
const ILS_HALFWIDTH_CM = 16.0

# Hamming taper coefficients, A(L) = A0 + (1−A0)·cos(π L / L_max).  The classic
# 0.54/0.46 raised cosine: unity at zero OPD, 0.08 at the end of the sweep.
const HAMMING_A0 = 0.54

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
    ils_kernel(Δν, opd_max, fwhm_gauss;
               apodization=:gaussian, n_pts=2048) -> (δν_arr, ils)

Compute the ILS as the Fourier transform of an apodized rectangular OPD
window.  Supports IASI-style Gaussian apodization (default), the three
Norton-Beer recipes, and CrIS-style Hamming.

**Physical model**

The unapodized FTS ILS is sinc(2π opd_max δν).  In the OPD domain this
corresponds to a rectangular window rect(L / opd_max).  Apodization is an
additional taper A(L) applied inside the rectangle; the ILS is then

    ILS(δν) = 2 ∫₀^L_max A(L) cos(2π L δν) dL.

Apodization options:

- `:gaussian` (default) — IASI L1C convention.  A(L) = exp(−2π² σ² L²)
  with σ = `fwhm_gauss` / (2 √(2 ln 2)).  Resulting ILS is sinc ⊗ Gaussian
  with main-lobe FWHM ≈ `fwhm_gauss`.
- `:norton_beer_weak` / `:norton_beer_medium` / `:norton_beer_strong` —
  Norton & Beer (1976) polynomial taper.  A(L) = C0 + C2 (1−x²)² + C4 (1−x²)⁴
  with x = L / `opd_max`.  Side-lobe-optimised; `fwhm_gauss` is ignored.
- `:hamming` — the CrIS convention.  A(L) = 0.54 + 0.46 cos(π L / `opd_max`),
  the 0.54/0.46 raised cosine.  It suppresses the peak side lobe from −13.3 dB
  (the bare sinc) to about −42 dB, at the cost of doubling the null-to-null
  main-lobe width — the FWHM grows by roughly half.  `fwhm_gauss` is ignored.

  Equivalently — and this is how CrIS SDR users usually meet it — Hamming
  apodization is a three-point convolution `[0.23, 0.54, 0.23]` applied to the
  unapodized channel radiances.  The two are the same operation: the transform
  of that three-tap kernel is `0.54 + 0.46 cos(2π L Δν)`, which equals the taper
  above whenever `Δν = 1/(2 opd_max)`, i.e. at full spectral resolution.  CrIS
  FSR satisfies this exactly (Δν = 0.625 cm⁻¹, OPD 0.8 cm).  Apply it here, in
  the OPD domain, rather than convolving the output spectrum — doing both
  apodizes twice.

# Arguments
- `Δν`:           spectral grid spacing of the high-res internal grid (cm⁻¹)
- `opd_max`:      maximum OPD (cm); for IASI = 2.0
- `fwhm_gauss`:   FWHM of the Gaussian broadening kernel (cm⁻¹); for IASI = 0.5.
                  Only used when `apodization=:gaussian`; pass any value
                  for the Norton-Beer modes.
- `apodization`:  apodization style (Symbol); default `:gaussian`
- `n_pts`:        number of OPD quadrature points
"""
function ils_kernel(Δν::Float64,
                    opd_max::Float64,
                    fwhm_gauss::Float64;
                    apodization::Symbol = :gaussian,
                    n_pts::Int = 2048)
    # One-sided OPD quadrature grid
    opd = LinRange(0.0, opd_max, n_pts)
    dL  = opd_max / (n_pts - 1)

    # OPD-domain apodization taper A(L)
    A = if apodization === :gaussian
        σ = fwhm_gauss / (2.0 * sqrt(2.0 * log(2.0)))   # Gaussian σ (cm⁻¹)
        [exp(-2π^2 * σ^2 * L^2) for L in opd]
    elseif haskey(NORTON_BEER_COEFFS, apodization)
        [norton_beer_apodization(L / opd_max, apodization) for L in opd]
    elseif apodization === :hamming
        [HAMMING_A0 + (1.0 - HAMMING_A0) * cos(π * L / opd_max) for L in opd]
    else
        error("Unknown apodization $(apodization); expected :gaussian, :hamming, or one of $(collect(keys(NORTON_BEER_COEFFS)))")
    end

    # Spectral offset grid: cover ±ILS_HALFWIDTH_CM at Δν spacing
    n_half = ceil(Int, ILS_HALFWIDTH_CM / Δν)
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
