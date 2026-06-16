"""
FFT-based ILS convolution (roadmap Phase 4).

`apply_ils` convolves the high-resolution spectrum with the (truncated, symmetric)
ILS kernel by direct summation — `O(n_ν · n_ils)` per spectrum. At the production
internal grid (`internal_dnu = 0.001` cm⁻¹) the kernel spans `n_ils ≈ 32000` points,
so a Jacobian, which applies the same operator to every state column, spends most of
its time here. This module does the identical linear convolution by FFT in
`O(n_ν log n_ν)`, reproducing `apply_ils` to round-off.

`out = Δν · conv(spectrum, kern)[half+1 : half+n_ν]`, the zero-padded linear
convolution (the direct loop skips out-of-range neighbours, i.e. pads with zeros). An
`ILSConvolver` precomputes the kernel transform and the FFTW plans + buffers so the
per-column cost in `analytic_jacobian` is one rFFT / multiply / irFFT with no fresh
allocation. The kernel is the same one `ils_kernel` builds, so the result matches the
forward model's `apply_ils` (no change to the validated forward numerics).
"""

using FFTW: plan_rfft, plan_irfft
using LinearAlgebra: mul!

"""
    ILSConvolver(ν_grid, ils_δν, ils_kern)

Precomputed FFT ILS operator for spectra on `ν_grid`. Holds the kernel's rFFT, the
forward/inverse FFTW plans, and work buffers; reuse it across many spectra (e.g. the
columns of a Jacobian). See `ils_apply!`.
"""
struct ILSConvolver{P,Q}
    N::Int                       # spectrum length (ν_grid.n)
    half::Int                    # (n_ils-1)÷2
    L::Int                       # padded FFT length
    Δν::Float64
    Kfft::Vector{ComplexF64}     # rFFT of the zero-padded kernel
    rbuf::Vector{Float64}        # length L (real work / output buffer)
    cbuf::Vector{ComplexF64}     # length L÷2+1 (spectral work buffer)
    pf::P                        # plan_rfft(rbuf)
    pinv::Q                      # plan_irfft(cbuf, L)
end

function ILSConvolver(ν_grid::WavenumberGrid,
                      ils_δν::AbstractVector{<:Real},
                      ils_kern::AbstractVector{<:Real})
    N     = ν_grid.n
    n_ils = length(ils_kern)
    half  = (n_ils - 1) ÷ 2
    L     = nextprod((2, 3, 5, 7), N + n_ils - 1)   # fast FFT length ≥ linear-conv size
    rbuf  = zeros(Float64, L)
    @inbounds rbuf[1:n_ils] .= ils_kern
    pf    = plan_rfft(rbuf)
    Kfft  = pf * rbuf
    cbuf  = Vector{ComplexF64}(undef, length(Kfft))
    pinv  = plan_irfft(cbuf, L)
    return ILSConvolver{typeof(pf), typeof(pinv)}(N, half, L, Float64(ν_grid.Δν),
                                                  Kfft, rbuf, cbuf, pf, pinv)
end

"""
    ils_apply!(conv, out, spectrum) -> out

Apply the ILS convolution to `spectrum` (length `conv.N`), writing the result into
`out` (length `conv.N`). Reuses the convolver's buffers — no allocation.
"""
function ils_apply!(conv::ILSConvolver, out::AbstractVector{<:Real},
                    spectrum::AbstractVector{<:Real})
    N, L = conv.N, conv.L
    length(spectrum) == N || error("spectrum length $(length(spectrum)) ≠ $(N)")
    length(out) == N      || error("out length $(length(out)) ≠ $(N)")
    @inbounds begin
        fill!(conv.rbuf, 0.0)
        copyto!(view(conv.rbuf, 1:N), spectrum)
    end
    mul!(conv.cbuf, conv.pf, conv.rbuf)        # rFFT(zero-padded spectrum)
    @inbounds conv.cbuf .*= conv.Kfft          # multiply by kernel transform
    mul!(conv.rbuf, conv.pinv, conv.cbuf)      # irFFT → linear convolution in rbuf
    @inbounds @views out .= conv.Δν .* conv.rbuf[conv.half + 1 : conv.half + N]
    return out
end

"""
    apply_ils_fft(ν_grid, spectrum, ils_δν, ils_kern) -> Vector{Float64}

Allocating convenience wrapper around `ILSConvolver`/`ils_apply!`: reproduces
`apply_ils` via FFT. For repeated application (Jacobian columns), build one
`ILSConvolver` and call `ils_apply!` instead.
"""
function apply_ils_fft(ν_grid::WavenumberGrid,
                       spectrum::AbstractVector{<:Real},
                       ils_δν::AbstractVector{<:Real},
                       ils_kern::AbstractVector{<:Real})::Vector{Float64}
    conv = ILSConvolver(ν_grid, ils_δν, ils_kern)
    return ils_apply!(conv, Vector{Float64}(undef, ν_grid.n), spectrum)
end
