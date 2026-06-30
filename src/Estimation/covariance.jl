"""
Measurement-error covariance construction for optimal estimation.

The headline routine is [`apodized_measurement_covariance`], which builds the
**inter-channel-correlated** `Sₑ` that IASI L1C actually has. Raw FTS samples are
spectrally uncorrelated (the sinc ILS sampled at the channel spacing is a delta),
but IASI L1C is Gaussian-apodized (FWHM 0.5 cm⁻¹), and that self-apodization mixes
neighbouring channels — so the L1C noise covariance is banded, not diagonal
(Vincent thesis §3.6; Amato et al. 1998). Using a diagonal `Sₑ` on contiguous L1C
channels mis-weights the retrieval; the correlation is only negligible if the
selected channels are ≥4 indices apart (lag 1≈0.70, 2≈0.25, 3≈0.04, 4≈0.004).
"""

using LinearAlgebra: Symmetric

"""
    _ils_noise_autocorr(Δν_sample, fwhm_gauss; max_lag=4) -> Vector{Float64}

Normalized channel-lag autocorrelation ρ[0…max_lag] of the Gaussian apodization
self-weights. The L1B→L1C apodization convolves the spectrum by a Gaussian of the
given FWHM; sampled on the channel grid the weights are `wₖ ∝ exp(−(k·Δν)²/2σ²)`
with `σ = fwhm/(2√(2ln2))`, and the noise correlation is their discrete
autocorrelation `ρ(d) = Σₖ wₖ wₖ₊d / Σₖ wₖ²` (= the `O·Oᵀ` of `Sₑ = O·Sₑ_white·Oᵀ`).
For IASI (Δν=0.25, fwhm=0.5) this gives ρ ≈ [1, 0.705, 0.250, 0.044, 0.0039, …] —
matching the thesis statement that channels >3 indices apart correlate <0.4%
(lag 4 ≈ 0.39%). `fwhm_gauss ≤ 0` ⇒ no apodization ⇒ diagonal (ρ = [1, 0, …]).
"""
function _ils_noise_autocorr(Δν_sample::Float64, fwhm_gauss::Float64; max_lag::Int = 4)
    fwhm_gauss <= 0.0 && return Float64[k == 0 ? 1.0 : 0.0 for k in 0:max_lag]
    σ = fwhm_gauss / (2.0 * sqrt(2.0 * log(2.0)))
    K = max_lag + 6                                   # pad so the autocorr tail is captured
    w = [exp(-(k * Δν_sample)^2 / (2σ^2)) for k in -K:K]
    R0 = sum(abs2, w)
    return [k == 0 ? 1.0 : sum(w[i] * w[i + k] for i in 1:(length(w) - k)) / R0
            for k in 0:max_lag]
end

"""
    apodized_measurement_covariance(ν, σ;
        Δν_sample=0.25, fwhm_gauss=0.5, max_lag=4) -> Symmetric{Float64}

Build the IASI L1C measurement-error covariance `Sₑ` for the channels at
wavenumbers `ν` (cm⁻¹), accounting for the inter-channel noise correlation
introduced by Gaussian apodization (Vincent thesis §3.6):

    Sₑ[i,j] = σᵢ · σⱼ · ρ(|round((νᵢ − νⱼ)/Δν_sample)|),

where `ρ` is the apodization autocorrelation from [`_ils_noise_autocorr`] and
correlations beyond `max_lag` channels are dropped (ρ < 0.4% past 3 for IASI). The
result is banded for a contiguous channel window and ~diagonal for a sparse
channel selection whose members are ≥`max_lag` indices apart.

# Arguments
- `ν`  : channel wavenumbers (cm⁻¹) — e.g. `granule.wno` or a selected subset.
- `σ`  : per-channel noise **standard deviation** in the units of `y` (radiance or
         BT); a scalar (uniform) or a vector matching `ν`. The diagonal is `σᵢ²`.
- `Δν_sample` : underlying L1C channel spacing (cm⁻¹), 0.25 for IASI.
- `fwhm_gauss`: apodization FWHM (cm⁻¹), 0.5 for IASI L1C; `0` ⇒ diagonal `Sₑ`.
- `max_lag`   : channel-lag band half-width to populate (default 4).

Pass the result straight to `optimal_estimation` as `Se`.
"""
function apodized_measurement_covariance(ν::AbstractVector{<:Real},
                                         σ::Union{Real, AbstractVector{<:Real}};
                                         Δν_sample::Float64 = 0.25,
                                         fwhm_gauss::Float64 = 0.5,
                                         max_lag::Int = 4)::Symmetric{Float64, Matrix{Float64}}
    n  = length(ν)
    σv = σ isa Real ? fill(Float64(σ), n) : collect(Float64, σ)
    length(σv) == n || error("σ must be a scalar or length(ν)=$n; got $(length(σv))")
    ρ  = _ils_noise_autocorr(Δν_sample, fwhm_gauss; max_lag = max_lag)
    Se = zeros(Float64, n, n)
    @inbounds for j in 1:n, i in 1:j
        lag = round(Int, abs(ν[i] - ν[j]) / Δν_sample)
        lag > max_lag && continue
        v = σv[i] * σv[j] * ρ[lag + 1]
        Se[i, j] = v; Se[j, i] = v
    end
    return Symmetric(Se)
end

"""
    scene_nedt(ν, Tb_scene; nedt_280K=0.25, T_ref=280.0) -> Vector{Float64}

Scene-specific noise-equivalent differential temperature NEΔTₛ per channel, in
brightness-temperature units (K), via Vincent thesis Eq. (2.3):

    NEΔTₛ(ν) = (∂B/∂T)|_T_ref · (∂Tb/∂L)|_scene · NEΔT_ref
             = NEΔT_ref · dB_dT(ν, T_ref) / dB_dT(ν, Tb_scene(ν)),

since ∂Tb/∂L = 1/(∂B/∂T)|_scene. Working in BT (not radiance) makes the noise
scene-dependent: the reported `NEΔT_ref` (instrument noise about a `T_ref`=280 K
blackbody, IASI ≈ 0.2–0.35 K) is only ~accurate above ~1300 cm⁻¹; below that
(e.g. the CO₂ ν₂ band) the cold, opaque channels are genuinely noisier in BT and
this conversion captures it. The Planck-radiance units cancel in the ratio.

# Arguments
- `ν`         : channel wavenumbers (cm⁻¹).
- `Tb_scene`  : per-channel scene brightness temperature (K) — pass the observed
                (or modelled) BT spectrum `y`.
- `nedt_280K` : reported NEΔT at `T_ref` — scalar or per-channel vector (K).
- `T_ref`     : reference blackbody temperature the NEΔT is quoted at (K).

Pass the result as the `σ` vector to [`apodized_measurement_covariance`].
"""
function scene_nedt(ν::AbstractVector{<:Real}, Tb_scene::AbstractVector{<:Real};
                    nedt_280K::Union{Real, AbstractVector{<:Real}} = 0.25,
                    T_ref::Real = 280.0)::Vector{Float64}
    n = length(ν)
    length(Tb_scene) == n || error("Tb_scene must match length(ν)=$n; got $(length(Tb_scene))")
    n0 = nedt_280K isa Real ? fill(Float64(nedt_280K), n) : collect(Float64, nedt_280K)
    length(n0) == n || error("nedt_280K must be scalar or length(ν)=$n; got $(length(n0))")
    Tref = Float64(T_ref)
    return [n0[i] * dB_dT(Float64(ν[i]), Tref) / dB_dT(Float64(ν[i]), Float64(Tb_scene[i]))
            for i in 1:n]
end

"""
    scene_measurement_covariance(ν, Tb_scene; nedt_280K=0.25, T_ref=280.0, kwargs...)

Convenience wrapper: scene-specific Sₑ with the Eq. (2.3) diagonal ([`scene_nedt`])
and the apodization off-diagonals ([`apodized_measurement_covariance`], thesis
§3.6). `kwargs` (`Δν_sample`, `fwhm_gauss`, `max_lag`) pass through to the latter.
"""
function scene_measurement_covariance(ν::AbstractVector{<:Real}, Tb_scene::AbstractVector{<:Real};
                                      nedt_280K::Union{Real, AbstractVector{<:Real}} = 0.25,
                                      T_ref::Real = 280.0, kwargs...)
    σ = scene_nedt(ν, Tb_scene; nedt_280K = nedt_280K, T_ref = T_ref)
    return apodized_measurement_covariance(ν, σ; kwargs...)
end
