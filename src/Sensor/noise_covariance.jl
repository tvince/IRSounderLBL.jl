"""
Host-provided sensor noise covariance.

Space agencies publish the measured **measurement-error covariance** `Sₑ` for
their sounders — for IASI, the EUMETSAT *IASI L1C Noise Covariance Matrix (NCM)*
product (EO:EUM:DAT:1099), a full 8461×8461 radiance covariance in a NetCDF-4
file. Because IASI L1C is Gaussian-apodized, this covariance is strongly banded
(neighbour correlation ≈ 0.7), so the full matrix — not a diagonal NEΔT — is what
a correlated-`Sₑ` retrieval needs. This module reads that product, unit-converts
it to the forward model's radiance units, optionally restricts it to a retrieval's
channel window, and (since the retrieval runs in brightness-temperature space)
linearizes it to a BT-space `Sₑ`.

Use it in place of the analytically-modelled [`scene_measurement_covariance`]
when the agency's measured covariance is available.
"""

using HDF5: h5open
using LinearAlgebra: Symmetric

"""
    SounderNoiseCovariance

A sounder measurement-error covariance `Sₑ` and its channel grid.

# Fields
- `ν`     : channel wavenumbers (cm⁻¹), length `n`
- `cov`   : `n×n` covariance matrix in the units implied by `space`
- `space` : `:radiance` — mW²/(m²·sr·cm⁻¹)² (the forward-model radiance units), or
            `:bt` — K² (brightness-temperature space, ready for `optimal_estimation`)

Pass [`measurement_covariance`](@ref)`(nc)` (or `nc.cov`) as `Se` to
[`optimal_estimation`](@ref) — in BT space when the retrieval observable is `:bt`
(the default), in radiance space when it is `:radiance`.
"""
struct SounderNoiseCovariance
    ν::Vector{Float64}
    cov::Matrix{Float64}
    space::Symbol
end

Base.show(io::IO, nc::SounderNoiseCovariance) =
    print(io, "SounderNoiseCovariance($(length(nc.ν)) channels, ",
          "$(round(nc.ν[1]; digits=3))–$(round(nc.ν[end]; digits=3)) cm⁻¹, ",
          "space=:$(nc.space))")

"""
    measurement_covariance(nc) -> Symmetric{Float64}

The covariance matrix wrapped as `Symmetric` (drops any round-off asymmetry),
ready to hand to [`optimal_estimation`](@ref) as `Se`.
"""
measurement_covariance(nc::SounderNoiseCovariance) = Symmetric(nc.cov)

# W²  →  mW²  (forward-model radiance is mW/(m²·sr·cm⁻¹); the NCM stores W²).
const _W2_TO_MW2 = 1.0e6

"""
    read_iasi_ncm(path; ν_window=nothing) -> SounderNoiseCovariance

Read the EUMETSAT IASI L1C Noise Covariance Matrix product (NetCDF-4/HDF5;
EO:EUM:DAT:1099). The file stores `Covariance_R` (8461×8461, radiance² in
W²/(m²·sr·cm⁻¹)²) on the `Wn` grid (m⁻¹). Returns a `:radiance`-space
[`SounderNoiseCovariance`] with wavenumbers in cm⁻¹ and the covariance converted
to the forward model's mW² radiance units.

`ν_window=(lo, hi)` restricts the read to channels in `[lo, hi]` cm⁻¹ via an HDF5
hyperslab, so a retrieval window loads only its sub-block instead of the full
~0.5 GB matrix. The IASI NCM grid is contiguous (645–2760 cm⁻¹ at 0.25 cm⁻¹), so
the window is a contiguous index range.
"""
function read_iasi_ncm(path::AbstractString;
                       ν_window::Union{Nothing, Tuple{<:Real, <:Real}} = nothing)
    return h5open(path, "r") do h
        haskey(h, "Wn") && haskey(h, "Covariance_R") ||
            error("$(path) is not an IASI NCM file (missing Wn / Covariance_R)")
        ν = read(h["Wn"]) ./ 100.0            # m⁻¹ → cm⁻¹
        covds = h["Covariance_R"]
        if ν_window === nothing
            cov = read(covds)
            idx = eachindex(ν)
        else
            lo, hi = ν_window
            keep = findall(v -> lo <= v <= hi, ν)
            isempty(keep) && error("ν_window $(ν_window) selects no NCM channels")
            i1, i2 = first(keep), last(keep)
            # Contiguous grid ⇒ hyperslab the sub-block (avoids loading the full matrix).
            cov = covds[i1:i2, i1:i2]
            idx = i1:i2
            ν = ν[idx]
        end
        return SounderNoiseCovariance(collect(Float64, ν),
                                      Matrix{Float64}(cov) .* _W2_TO_MW2,
                                      :radiance)
    end
end

"""
    load_noise_covariance(path; format=:iasi_ncm, kwargs...) -> SounderNoiseCovariance

Generic entry point for host-provided noise covariances. `format=:iasi_ncm`
dispatches to [`read_iasi_ncm`](@ref); `kwargs` (e.g. `ν_window`) pass through.
Add new agency formats here as they are supported.
"""
function load_noise_covariance(path::AbstractString; format::Symbol = :iasi_ncm, kwargs...)
    format === :iasi_ncm && return read_iasi_ncm(path; kwargs...)
    error("unknown noise-covariance format :$format (supported: :iasi_ncm)")
end

"""
    subset_channels(nc, ν_target; atol=1e-3) -> SounderNoiseCovariance

Restrict `nc` to the channels matching `ν_target` (cm⁻¹), preserving order. Each
target is matched to the nearest covariance channel within `atol` cm⁻¹ (errors if
none is that close). Use this to align a full-band NCM to the channel subset a
retrieval actually uses.
"""
function subset_channels(nc::SounderNoiseCovariance, ν_target::AbstractVector{<:Real};
                         atol::Real = 1e-3)
    idx = Vector{Int}(undef, length(ν_target))
    @inbounds for (k, νt) in enumerate(ν_target)
        j = argmin(abs.(nc.ν .- νt))
        abs(nc.ν[j] - νt) <= atol ||
            error("no NCM channel within $(atol) cm⁻¹ of target $(νt) (nearest $(nc.ν[j]))")
        idx[k] = j
    end
    return SounderNoiseCovariance(nc.ν[idx], nc.cov[idx, idx], nc.space)
end

"""
    to_bt(nc, Tb; T_floor=1e-6) -> SounderNoiseCovariance

Linearize a `:radiance`-space covariance to brightness-temperature space (K²) at
the scene, so it can be used as `Sₑ` for a BT-observable retrieval:

    Sₑ,bt = D · Sₑ,R · Dᵀ,   Dᵢ = ∂Tb/∂L|scene = 1 / (∂B/∂T)(νᵢ, Tbᵢ),

the same Planck linearization [`scene_nedt`] uses (the mW radiance units cancel
against `∂B/∂T` in mW). `Tb` is the per-channel scene brightness temperature (K),
e.g. the observed spectrum `y`; it must align with `nc.ν` (same length/order —
[`subset_channels`](@ref) first if needed). Idempotent guard: a `:bt` covariance
is returned unchanged.
"""
function to_bt(nc::SounderNoiseCovariance, Tb::AbstractVector{<:Real}; T_floor::Real = 1e-6)
    nc.space === :bt && return nc
    nc.space === :radiance || error("cannot convert space :$(nc.space) to :bt")
    length(Tb) == length(nc.ν) ||
        error("Tb length $(length(Tb)) ≠ covariance channels $(length(nc.ν)); subset_channels first")
    D = [1.0 / max(dB_dT(nc.ν[i], Float64(Tb[i])), T_floor) for i in eachindex(nc.ν)]
    cov_bt = (D * D') .* nc.cov
    return SounderNoiseCovariance(copy(nc.ν), cov_bt, :bt)
end
