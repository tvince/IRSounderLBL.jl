using KernelAbstractions
using SpecialFunctions: erfcx

const _SQRT_LN2    = sqrt(log(2.0))
const _INV_SQRT_PI = inv(sqrt(π))

# |x| (in Doppler-scaled units x = Δν·√ln2/γ_D) beyond which the pure Lorentzian
# matches the exact Voigt H-function to <1e-4 for ALL y. Verified numerically over
# y∈[1e-3,30]. Past this point the Gaussian core is dead and H(x,y) → y/(√π(x²+y²)),
# so we skip the expensive complex erfcx. Mirrors LBLRTM CONVF4's far-wing
# f4fn = S·γ_L/(π(γ_L²+Δν²)). Threshold is in x-units so it auto-scales per line
# via f=√ln2/γ_D (≈0.3 cm⁻¹ from center at 4.3µm, ≈0.08 cm⁻¹ at 15µm).
const _X_FAR = 122.0

# Floating-point type used to evaluate the Voigt profile on a given KernelAbstractions
# backend. CPU and CUDA support Float64; Apple Metal is Float32-only, so the Metal
# package extension overrides this for `Metal.MetalBackend`.
_backend_float_type(backend) = Float64

"""
    VoigtMethod

Selects the algorithm used to evaluate the Voigt profile.

- `FullFaddeeva`: Exact Faddeeva w(z) via `SpecialFunctions.erfcx`. Machine-precision
                  accuracy; ~4.5× multi-thread speedup on CPU. **Default.**
- `Weideman`    : Weideman (1994) 24-term rational approximation, ~1e-4 accuracy.
                  GPU-compatible (Float32 on Metal, Float64 on CUDA/ROCm).
- `PseudoVoigt` : Thompson et al. (1987) pseudo-Voigt mixing η·L + (1-η)·G.
                  ~1% accuracy in the core; closed-form, no complex arithmetic.
"""
@enum VoigtMethod begin
    Weideman     = 1
    PseudoVoigt  = 2
    FullFaddeeva = 3
end

# ── Method 1: Weideman (1994) 24-term rational approximation — default ────────

# Precompute quadrature nodes and weights for the Weideman (1994) formula.
# Derivation: trapezoidal rule on ∫exp(-t²)/(z-t)dt after the substitution
# t = L*tan(u), u ∈ (-π/2, π/2), with N equal steps.
#
# H(x,y) = Re[w(x+iy)] = (y / (N·L)) · Σₙ Cₙ / ((x−tₙ)² + y²)
#
# Nodes:   tₙ = L · tan((2n−1−N)·π / (2N)),  n = 1…N
# Weights: Cₙ = (L² + tₙ²) · exp(−tₙ²)
# Scale:   N · L
#
# Optimal L (minimises max error for given N): L = √(N/ln2)
# For N=24: L ≈ 5.885, max relative error ≈ 1.9×10⁻⁴ (Weideman 1994, Table 2).
# No region switching: uniform accuracy at all (x,y) with y > 0.

const _W_N = 24

function _weideman_precompute(N::Int)
    L  = sqrt(Float64(N) / log(2.0))
    ts = ntuple(n -> L * tan(Float64(2n - 1 - N) * π / Float64(2N)), N)
    ws = ntuple(n -> L^2 + ts[n]^2, N)                   # Wₙ = L² + tₙ²
    cs = ntuple(n -> ws[n] * exp(-ts[n]^2), N)            # Cₙ = Wₙ · exp(−tₙ²)
    return ts, ws, cs, Float64(N) * L
end

const (_W_T, _W_W, _W_C, _W_SCALE) = _weideman_precompute(_W_N)

# Float32 versions for Metal GPU (Apple Silicon does not support Float64)
const _W_T32     = ntuple(n -> Float32(_W_T[n]), _W_N)
const _W_W32     = ntuple(n -> Float32(_W_W[n]), _W_N)
const _W_C32     = ntuple(n -> Float32(_W_C[n]), _W_N)
const _W_SCALE32 = Float32(_W_SCALE)


"""
    weideman_voigt(x, y) -> Float64

Weideman (1994) 24-term rational approximation to the Voigt H-function
H(x,y) = Re[w(x+iy)].  Pure Float64 arithmetic — no complex numbers, no
region switching.  GPU-compatible; loop unrolled at compile time.

Uses a two-sum complementary correction to recover the exact Gaussian limit
as y → 0: A = (y/NL)·ΣCₙ/dₙ and B = (y/NL)·ΣWₙ/dₙ, so
H = exp(-x²)·(1-B) + A → exp(-x²) as y→0 and → A ≈ H_true for large y.

Max relative error ≈ 1.9×10⁻⁴ for y ≥ ~0.5; recovers Gaussian for y → 0.
For 0 < y < 0.5, B can exceed 1 near quadrature nodes — clipped to 0 by max().

Reference: Weideman J.A.C. (1994), SIAM J. Sci. Comput. 15(5), 1996–2002.
"""
@inline function weideman_voigt(x::Float64, y::Float64)::Float64
    acc_A = 0.0
    acc_B = 0.0
    @inbounds for n in 1:_W_N
        d      = (x - _W_T[n])^2 + y^2
        acc_A += _W_C[n] / d      # Cₙ = Wₙ · exp(−tₙ²)
        acc_B += _W_W[n] / d      # Wₙ = L² + tₙ²  (pure quadrature weight)
    end
    scale = y / _W_SCALE
    A = acc_A * scale             # standard Weideman sum → 0 as y → 0
    B = acc_B * scale             # approximates Lorentzian integral ≈ 1
    return max(0.0, exp(-x^2) * (1.0 - B) + A)
end

# Float32 variant — identical algorithm, uses Float32 precomputed constants.
# Required for Metal GPU (Apple Silicon does not support Float64 in shaders).
@inline function weideman_voigt(x::Float32, y::Float32)::Float32
    acc_A = 0.0f0
    acc_B = 0.0f0
    @inbounds for n in 1:_W_N
        d      = (x - _W_T32[n])^2 + y^2
        acc_A += _W_C32[n] / d
        acc_B += _W_W32[n] / d
    end
    scale = y / _W_SCALE32
    A = acc_A * scale
    B = acc_B * scale
    return max(0.0f0, exp(-x^2) * (1.0f0 - B) + A)
end

# ── Method 2: Pseudo-Voigt (Thompson et al. 1987) ────────────────────────────

"""
    pseudo_voigt_profile(ν, ν₀, γ_L, γ_D) -> Float64

Thompson et al. (1987) pseudo-Voigt approximation to the Voigt profile:

    PV(ν) = η · L(ν; γ_V) + (1 − η) · G(ν; γ_V)

where γ_V and η are derived from γ_L and γ_D via the Thompson formula.
Max error ~1% near line center; larger in the far wings.

References:
  Thompson P., Cox D.E., Hastings J.B. (1987), J. Appl. Cryst. 20, 79–83.
"""
@inline function pseudo_voigt_profile(ν::Float64, ν₀::Float64,
                                       γ_L::Float64, γ_D::Float64)::Float64
    fG  = 2.0 * γ_D
    fL  = 2.0 * γ_L
    fV  = (fG^5 + 2.69269*fG^4*fL + 2.42843*fG^3*fL^2 +
           4.47163*fG^2*fL^3 + 0.07842*fG*fL^4 + fL^5)^0.2
    γV  = 0.5 * fV
    r   = fL / fV
    η   = 1.36603*r - 0.47719*r^2 + 0.11116*r^3
    Δν  = ν - ν₀
    G   = sqrt(log(2.0) / π) / γV * exp(-log(2.0) * Δν^2 / γV^2)
    L   = γV / (π * (Δν^2 + γV^2))
    return η * L + (1.0 - η) * G
end

# ── Method 3: Full Faddeeva via SpecialFunctions.erfcx ───────────────────────

"""
    faddeeva_voigt(x, y) -> Float64

Exact Voigt H-function via the identity  Re[w(x+iy)] = Re[erfcx(y − ix)],
using `SpecialFunctions.erfcx` for machine-precision accuracy (~1e-15).
CPU-only; ~3–4× slower than Humlicek.
"""
@inline function faddeeva_voigt(x::Float64, y::Float64)::Float64
    return real(erfcx(complex(y, -x)))
end

"""
    _voigt_H(x, y, x_far) -> Float64

Voigt H-function with an analytic far-wing shortcut: for |x| > `x_far` the profile
is the pure Lorentzian `y/(√π(x²+y²))` (exact Voigt limit once the Gaussian core has
decayed); otherwise the full Faddeeva. Pass `x_far = Inf` to force exact Faddeeva
everywhere (used by the regression gate). See `_X_FAR`.
"""
@inline function _voigt_H(x::Float64, y::Float64, x_far::Float64)::Float64
    return abs(x) > x_far ? y * _INV_SQRT_PI / (x*x + y*y) : faddeeva_voigt(x, y)
end

# ── Unified profile interface ─────────────────────────────────────────────────

"""
    voigt_profile(ν, ν₀, γ_L, γ_D[, method]) -> Float64

Evaluate the normalized Voigt profile at wavenumber `ν` (cm⁻¹),
centered at `ν₀` (cm⁻¹), with Lorentz HWHM `γ_L` and Doppler HWHM `γ_D`.

`method` is a `VoigtMethod` (default `FullFaddeeva`):
- `FullFaddeeva`: machine precision via `SpecialFunctions.erfcx`. Default.
- `Weideman`    : ~1e-4 accuracy, GPU-compatible (Float32 Metal / Float64 CUDA).
- `PseudoVoigt` : ~1% accuracy, closed-form mixing approximation.

Returns profile in units of cm (i.e. ∫V dν = 1 when γ values are in cm⁻¹).
"""
@inline function voigt_profile(ν::Float64, ν₀::Float64,
                                γ_L::Float64, γ_D::Float64,
                                method::VoigtMethod = FullFaddeeva)::Float64
    if method == PseudoVoigt
        return pseudo_voigt_profile(ν, ν₀, γ_L, γ_D)
    end
    f = _SQRT_LN2 / γ_D
    x = (ν - ν₀) * f
    y = γ_L * f
    H = method == Weideman ? weideman_voigt(x, y) : faddeeva_voigt(x, y)
    return f * H * _INV_SQRT_PI
end

# ── KernelAbstractions GPU Kernel (Humlicek only) ────────────────────────────

"""
    voigt_cross_section_kernel!

KernelAbstractions kernel that computes the Voigt cross-section at each
wavenumber grid point by summing contributions from all spectral lines
using the Weideman (1994) approximation.
`σ` (length n_ν) accumulates cross-sections in cm²/molec.

Per-line arrays are precomputed before kernel launch:
  lines_f     = √(ln2) / γ_D          (Doppler scale factor)
  lines_y     = γ_L × lines_f         (dimensionless Lorentz width)
  lines_Snorm = S × lines_f / √π      (combined intensity+normalization)
"""
# Binary search helpers — portable across CPU and GPU backends.
# _lower_bound: first index j where arr[j] >= val  (1-based, returns n+1 if none)
# _upper_bound: last  index j where arr[j] <= val  (1-based, returns 0 if none)
@inline function _lower_bound(arr, val, n)
    lo, hi = 1, n + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        @inbounds arr[mid] < val ? (lo = mid + 1) : (hi = mid)
    end
    return lo
end

@inline function _upper_bound(arr, val, n)
    lo, hi = 0, n
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        @inbounds arr[mid] > val ? (hi = mid - 1) : (lo = mid)
    end
    return lo
end

# Shared vectorized kernel for FullFaddeeva and Weideman. The core H-function
# (`Hcore` = `faddeeva_voigt` or `weideman_voigt`) is passed as an argument so the
# kernel specializes per method at compile time — keeping a single code path while
# letting Metal compile only the GPU-safe Weideman variant. The KernelAbstractions
# CPU backend multithreads this; it is ~50× faster than a scalar threaded loop.
@kernel function voigt_cross_section_kernel!(σ,
                                              ν_grid,
                                              lines_ν0,
                                              lines_f,
                                              lines_y,
                                              lines_Snorm,
                                              lines_H_cutoff,
                                              cutoff,
                                              x_far,
                                              Hcore)
    i = @index(Global, Linear)
    ν       = ν_grid[i]
    n_lines = length(lines_ν0)

    # Lines are sorted by ν0; restrict to the window [ν-cutoff, ν+cutoff]
    j_lo = _lower_bound(lines_ν0, ν - cutoff, n_lines)
    j_hi = _upper_bound(lines_ν0, ν + cutoff, n_lines)

    acc  = zero(ν)   # Float32 on Metal, Float64 on CPU
    rc   = one(ν) / cutoff
    ispi = oftype(ν, _INV_SQRT_PI)
    for j in j_lo:j_hi
        Δν  = ν - lines_ν0[j]
        x   = Δν * lines_f[j]
        y   = lines_y[j]
        zr  = Δν * rc
        # AER parabolic pedestal (2 − (Δν/cutoff)²)·V(cutoff): zeroes both the
        # value AND the slope at the boundary (LBLRTM oprop.f90 CONVF4 fcnt_fn).
        ped = (oftype(ν, 2) - zr * zr) * lines_H_cutoff[j]
        # Far wing (|x| > x_far): Gaussian core dead → pure Lorentzian, skip Hcore.
        H   = (abs(x) > x_far ? y * ispi / (x*x + y*y) : Hcore(x, y)) - ped
        acc += lines_Snorm[j] * H
    end
    σ[i] = max(acc, zero(ν))
end

# Vectorized PseudoVoigt kernel (Thompson η·L + (1−η)·G). Per-line γV, η, S and the
# pedestal baseline (PV value at the cutoff) are precomputed on the host.
@kernel function pseudo_voigt_cross_section_kernel!(σ,
                                                    ν_grid,
                                                    lines_ν0,
                                                    lines_γV,
                                                    lines_η,
                                                    lines_S,
                                                    lines_baseline,
                                                    cutoff)
    i = @index(Global, Linear)
    ν       = ν_grid[i]
    n_lines = length(lines_ν0)

    j_lo = _lower_bound(lines_ν0, ν - cutoff, n_lines)
    j_hi = _upper_bound(lines_ν0, ν + cutoff, n_lines)

    acc  = zero(ν)
    rc   = one(ν) / cutoff
    cG   = oftype(ν, sqrt(log(2.0) / π))
    cln2 = oftype(ν, log(2.0))
    invπ = oftype(ν, inv(π))
    for j in j_lo:j_hi
        γV = lines_γV[j]
        η  = lines_η[j]
        Δν = ν - lines_ν0[j]
        G  = cG / γV * exp(-cln2 * Δν * Δν / (γV * γV))
        L  = γV * invπ / (Δν * Δν + γV * γV)
        zr = Δν * rc
        # AER parabolic pedestal (2 − (Δν/cutoff)²)·PV(cutoff); σ=0 at boundary.
        acc += lines_S[j] * (η * L + (one(ν) - η) * G) -
               (oftype(ν, 2) - zr * zr) * lines_baseline[j]
    end
    σ[i] = max(acc, zero(ν))
end

"""
    _reject_weak_lines(linelist, T, p_atm, coef; vmr_self=0.0, dptmn=0.0) -> HITRANLinelist

Drop lines whose PEAK optical-depth contribution in this layer is below the
absolute floor `dptmn`. The peak is the line-centre cross-section
`Snorm·H(0,y) = S·(√ln2/γ_D)·π^(-1/2)·erfcx(y)` (with `y = γ_L·√ln2/γ_D`), scaled
by the layer column coefficient `coef = vmr·Δp·N_air`, so `τ_peak = peak_σ·coef`.

This is LBLRTM's DPTMIN criterion (`oprop.f90`: `SPEAK > DPTMN`), evaluated PER
LAYER so the rejection auto-adapts to each layer's absorber amount and (T,p) line
shape — a line is dropped only where its strongest possible contribution to *that*
layer's optical depth is negligible.

`dptmn ≤ 0` disables rejection (returns the input unchanged). The floor is
BT-lossless at 1e-6 (≤0.6 mK over the real IASI ILS; ~5× fewer lines per layer —
see scripts/validation/validate_line_rejection_bt.jl). The relative DPTFC term tested there
was found to bias BT for no gain, so it is not carried here.
"""
function _reject_weak_lines(linelist::HITRANLinelist, T::Float64, p_atm::Float64,
                            coef::Float64; vmr_self::Float64 = 0.0,
                            dptmn::Float64 = 0.0)::HITRANLinelist
    dptmn <= 0.0 && return linelist
    lines = linelist.lines
    keep  = falses(length(lines))
    nkeep = 0
    @inbounds for (j, line) in enumerate(lines)
        S      = temperature_scaled_intensity(line, T)
        gl, gd = pressure_broadened_width(line, p_atm, T; vmr_self=vmr_self)
        gd     = max(gd, 1e-10)
        f      = _SQRT_LN2 / gd
        peakσ  = (S * f * _INV_SQRT_PI) * faddeeva_voigt(0.0, gl * f)
        if peakσ * coef > dptmn
            keep[j] = true
            nkeep  += 1
        end
    end
    nkeep == length(lines) && return linelist                               # nothing dropped
    nkeep == 0 && return HITRANLinelist(HITRANLine[], Set{Int}(),           # all dropped (σ≈0)
                                        linelist.ν_min, linelist.ν_max)
    return HITRANLinelist(lines[keep])
end

"""
    compute_voigt_cross_sections(ν_grid, linelist, T, p_atm;
                                  cutoff=25.0, backend=CPU(),
                                  method=FullFaddeeva) -> Vector{Float64}

Compute absorption cross-sections (cm²/molec) on `ν_grid` for all lines
in `linelist` at temperature T (K) and pressure p_atm (atm).

- `cutoff`:  line wing cut-off (cm⁻¹), default 25 cm⁻¹
- `backend`: KernelAbstractions backend, default `CPU()`
- `method`:  `VoigtMethod` enum — `FullFaddeeva` (default, machine precision, CPU-only),
             `Weideman` (~1e-4, GPU-capable), or `PseudoVoigt` (~1%, GPU-capable).
             All three run the vectorized KernelAbstractions kernel.
- `x_far`:   |x| (Doppler-scaled, x=Δν·√ln2/γ_D) beyond which the analytic
             Lorentzian replaces the Faddeeva/Weideman evaluation (default `_X_FAR`
             ≈122, <1e-4-lossless). Pass `Inf` to force exact evaluation everywhere.
             (Unused by `PseudoVoigt`, which has its own closed-form wing.)
"""
function compute_voigt_cross_sections(ν_grid::WavenumberGrid,
                                       linelist::HITRANLinelist,
                                       T::Float64,
                                       p_atm::Float64;
                                       vmr_self::Float64 = 0.0,
                                       cutoff::Float64 = 25.0,
                                       backend = CPU(),
                                       method::VoigtMethod = FullFaddeeva,
                                       x_far::Float64 = _X_FAR)
    # Metal (Apple Silicon) does not support Float64 — use Float32 automatically.
    # `_backend_float_type` defaults to Float64; the Metal package extension
    # (ext/IRSounderLBLMetalExt.jl) overrides it to Float32 for a MetalBackend.
    FT = _backend_float_type(backend)
    (method == FullFaddeeva && FT === Float32) &&
        error("FullFaddeeva uses complex erfcx (Float64/CPU only); use Weideman on a Metal backend.")

    n_ν = ν_grid.n
    n_L = length(linelist.lines)

    # ── Per-line parameters (shared) ─────────────────────────────────────────
    ν0    = Vector{FT}(undef, n_L)
    S_arr = Vector{FT}(undef, n_L)
    γL    = Vector{FT}(undef, n_L)
    γD    = Vector{FT}(undef, n_L)
    f_arr = Vector{FT}(undef, n_L)
    y_arr = Vector{FT}(undef, n_L)
    Snorm = Vector{FT}(undef, n_L)
    for (j, line) in enumerate(linelist.lines)
        ν0[j]  = FT(pressure_shift(line, p_atm; vmr_self=vmr_self))
        S      = temperature_scaled_intensity(line, T)
        gl, gd = pressure_broadened_width(line, p_atm, T; vmr_self=vmr_self)
        gd     = max(gd, 1e-10)
        f      = _SQRT_LN2 / gd
        S_arr[j] = FT(S)
        γL[j]    = FT(gl)
        γD[j]    = FT(gd)
        f_arr[j] = FT(f)
        y_arr[j] = FT(gl * f)
        Snorm[j] = FT(S * f * _INV_SQRT_PI)
    end

    σ   = KernelAbstractions.zeros(backend, FT, n_ν)
    ν_d = KernelAbstractions.allocate(backend, FT, n_ν)
    copyto!(ν_d, FT.(ν_grid.ν))
    todev(v) = (d = KernelAbstractions.allocate(backend, FT, length(v)); copyto!(d, v); d)
    ν0_d = todev(ν0)

    if method == PseudoVoigt
        # Per-line Thompson γV, η and the pedestal baseline (PV at the cutoff).
        γV   = Vector{FT}(undef, n_L)
        η    = Vector{FT}(undef, n_L)
        base = Vector{FT}(undef, n_L)
        cG   = FT(sqrt(log(2.0) / π))
        for j in 1:n_L
            fG = 2 * γD[j]; fL = 2 * γL[j]
            fV = (fG^5 + FT(2.69269)*fG^4*fL + FT(2.42843)*fG^3*fL^2 +
                  FT(4.47163)*fG^2*fL^3 + FT(0.07842)*fG*fL^4 + fL^5)^FT(0.2)
            r     = fL / fV
            γV[j] = FT(0.5) * fV
            η[j]  = FT(1.36603)*r - FT(0.47719)*r^2 + FT(0.11116)*r^3
            G_c   = cG / γV[j] * exp(FT(-log(2.0)) * FT(cutoff)^2 / γV[j]^2)
            L_c   = γV[j] * FT(inv(π)) / (FT(cutoff)^2 + γV[j]^2)
            base[j] = S_arr[j] * (η[j] * L_c + (one(FT) - η[j]) * G_c)
        end
        kernel! = pseudo_voigt_cross_section_kernel!(backend, 256)
        kernel!(σ, ν_d, ν0_d, todev(γV), todev(η), todev(S_arr), todev(base),
                FT(cutoff); ndrange = n_ν)
    else
        # FullFaddeeva or Weideman — shared kernel, core H selected per method.
        Hcore   = method == FullFaddeeva ? faddeeva_voigt : weideman_voigt
        xfar_FT = FT(x_far)
        # Pedestal value at the cutoff; x_cut = cutoff·f ≫ x_far so this is the
        # Lorentzian limit, matching the kernel's far-wing branch (zero at ±cutoff).
        H_cutoff = Vector{FT}(undef, n_L)
        for j in 1:n_L
            x_cut = FT(cutoff) * f_arr[j]
            H_cutoff[j] = abs(x_cut) > xfar_FT ?
                y_arr[j] * FT(_INV_SQRT_PI) / (x_cut*x_cut + y_arr[j]^2) :
                FT(Hcore(x_cut, y_arr[j]))
        end
        kernel! = voigt_cross_section_kernel!(backend, 256)
        kernel!(σ, ν_d, ν0_d, todev(f_arr), todev(y_arr), todev(Snorm),
                todev(H_cutoff), FT(cutoff), xfar_FT, Hcore; ndrange = n_ν)
    end

    KernelAbstractions.synchronize(backend)
    return Float64.(Array(σ))
end
