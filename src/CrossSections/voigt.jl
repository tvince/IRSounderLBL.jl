using KernelAbstractions
using Metal
using SpecialFunctions: erfcx

const _SQRT_LN2    = sqrt(log(2.0))
const _INV_SQRT_PI = inv(sqrt(π))

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

@kernel function voigt_cross_section_kernel!(σ,
                                              ν_grid,
                                              lines_ν0,
                                              lines_f,
                                              lines_y,
                                              lines_Snorm,
                                              lines_H_cutoff,
                                              cutoff)
    i = @index(Global, Linear)
    ν       = ν_grid[i]
    n_lines = length(lines_ν0)

    # Lines are sorted by ν0; restrict to the window [ν-cutoff, ν+cutoff]
    j_lo = _lower_bound(lines_ν0, ν - cutoff, n_lines)
    j_hi = _upper_bound(lines_ν0, ν + cutoff, n_lines)

    acc = zero(ν)   # Float32 on Metal, Float64 on CPU
    for j in j_lo:j_hi
        x   = (ν - lines_ν0[j]) * lines_f[j]
        H   = weideman_voigt(x, lines_y[j]) - lines_H_cutoff[j]
        acc += lines_Snorm[j] * H
    end
    σ[i] = max(acc, zero(ν))
end

# ── CPU threaded path for PseudoVoigt and FullFaddeeva ───────────────────────

function _compute_voigt_cpu(ν_grid::WavenumberGrid,
                             linelist::HITRANLinelist,
                             T::Float64,
                             p_atm::Float64,
                             cutoff::Float64,
                             method::VoigtMethod)
    n_ν = ν_grid.n
    n_L = length(linelist.lines)

    ν0     = Vector{Float64}(undef, n_L)
    S_arr  = Vector{Float64}(undef, n_L)
    γL_arr = Vector{Float64}(undef, n_L)
    γD_arr = Vector{Float64}(undef, n_L)
    f_arr  = Vector{Float64}(undef, n_L)
    y_arr  = Vector{Float64}(undef, n_L)
    Snorm  = Vector{Float64}(undef, n_L)

    for (j, line) in enumerate(linelist.lines)
        ν0[j]     = pressure_shift(line, p_atm)
        S         = temperature_scaled_intensity(line, T)
        gl, gd    = pressure_broadened_width(line, p_atm, T)
        gd        = max(gd, 1e-10)
        f         = _SQRT_LN2 / gd
        S_arr[j]  = S
        γL_arr[j] = gl
        γD_arr[j] = gd
        f_arr[j]  = f
        y_arr[j]  = gl * f
        Snorm[j]  = S * f * _INV_SQRT_PI
    end

    σ = zeros(Float64, n_ν)

    if method == PseudoVoigt
        # Precompute per-line Thompson parameters so the inner loop is cheap.
        γV_arr = Vector{Float64}(undef, n_L)
        η_arr  = Vector{Float64}(undef, n_L)
        for j in 1:n_L
            fG       = 2.0 * γD_arr[j]
            fL       = 2.0 * γL_arr[j]
            fV       = (fG^5 + 2.69269*fG^4*fL + 2.42843*fG^3*fL^2 +
                        4.47163*fG^2*fL^3 + 0.07842*fG*fL^4 + fL^5)^0.2
            r        = fL / fV
            γV_arr[j] = 0.5 * fV
            η_arr[j]  = 1.36603*r - 0.47719*r^2 + 0.11116*r^3
        end
        # Precompute cutoff baseline per line (MT-CKD convention: σ=0 at boundary)
        baseline_arr = Vector{Float64}(undef, n_L)
        for j in 1:n_L
            γV = γV_arr[j]
            η  = η_arr[j]
            G_c = sqrt(log(2.0) / π) / γV * exp(-log(2.0) * cutoff^2 / γV^2)
            L_c = γV / (π * (cutoff^2 + γV^2))
            baseline_arr[j] = S_arr[j] * (η * L_c + (1.0 - η) * G_c)
        end
        Threads.@threads for i in 1:n_ν
            νi   = ν_grid.ν[i]
            j_lo = _lower_bound(ν0, νi - cutoff, n_L)
            j_hi = _upper_bound(ν0, νi + cutoff, n_L)
            acc  = 0.0
            for j in j_lo:j_hi
                γV  = γV_arr[j]
                η   = η_arr[j]
                Δν  = νi - ν0[j]
                G   = sqrt(log(2.0) / π) / γV * exp(-log(2.0) * Δν^2 / γV^2)
                L   = γV / (π * (Δν^2 + γV^2))
                acc += S_arr[j] * (η * L + (1.0 - η) * G) - baseline_arr[j]
            end
            σ[i] = max(acc, 0.0)
        end
    else  # FullFaddeeva
        # Precompute cutoff baseline per line (MT-CKD convention: σ=0 at boundary)
        H_cutoff = Vector{Float64}(undef, n_L)
        for j in 1:n_L
            H_cutoff[j] = faddeeva_voigt(cutoff * f_arr[j], y_arr[j])
        end
        Threads.@threads for i in 1:n_ν
            νi   = ν_grid.ν[i]
            j_lo = _lower_bound(ν0, νi - cutoff, n_L)
            j_hi = _upper_bound(ν0, νi + cutoff, n_L)
            acc  = 0.0
            for j in j_lo:j_hi
                x   = (νi - ν0[j]) * f_arr[j]
                H   = faddeeva_voigt(x, y_arr[j]) - H_cutoff[j]
                acc += Snorm[j] * H
            end
            σ[i] = max(acc, 0.0)
        end
    end

    return σ
end

"""
    compute_voigt_cross_sections(ν_grid, linelist, T, p_atm;
                                  cutoff=25.0, backend=CPU(),
                                  method=FullFaddeeva) -> Vector{Float64}

Compute absorption cross-sections (cm²/molec) on `ν_grid` for all lines
in `linelist` at temperature T (K) and pressure p_atm (atm).

- `cutoff`:  line wing cut-off (cm⁻¹), default 25 cm⁻¹
- `backend`: KernelAbstractions backend, default `CPU()`
- `method`:  `VoigtMethod` enum — `FullFaddeeva` (default, machine precision),
             `PseudoVoigt`, or `FullFaddeeva` (both CPU-only)
"""
function compute_voigt_cross_sections(ν_grid::WavenumberGrid,
                                       linelist::HITRANLinelist,
                                       T::Float64,
                                       p_atm::Float64;
                                       cutoff::Float64 = 25.0,
                                       backend = CPU(),
                                       method::VoigtMethod = FullFaddeeva)
    if method != Weideman
        return _compute_voigt_cpu(ν_grid, linelist, T, p_atm, cutoff, method)
    end

    # Metal (Apple Silicon) does not support Float64 — use Float32 automatically.
    FT = (backend isa Metal.MetalBackend) ? Float32 : Float64

    n_ν = ν_grid.n
    n_L = length(linelist.lines)

    ν0    = Vector{FT}(undef, n_L)
    f_arr = Vector{FT}(undef, n_L)
    y_arr = Vector{FT}(undef, n_L)
    Snorm = Vector{FT}(undef, n_L)

    for (j, line) in enumerate(linelist.lines)
        ν0[j]  = FT(pressure_shift(line, p_atm))
        S      = temperature_scaled_intensity(line, T)
        gl, gd = pressure_broadened_width(line, p_atm, T)
        gd     = max(gd, 1e-10)
        f      = _SQRT_LN2 / gd
        f_arr[j] = FT(f)
        y_arr[j] = FT(gl * f)
        Snorm[j] = FT(S * f * _INV_SQRT_PI)
    end

    σ    = KernelAbstractions.zeros(backend, FT, n_ν)
    ν_d  = KernelAbstractions.allocate(backend, FT, n_ν)
    copyto!(ν_d, FT.(ν_grid.ν))

    H_cutoff = Vector{FT}(undef, n_L)
    for j in 1:n_L
        H_cutoff[j] = FT(weideman_voigt(FT(cutoff) * f_arr[j], y_arr[j]))
    end

    ν0_d       = KernelAbstractions.allocate(backend, FT, n_L)
    f_d        = KernelAbstractions.allocate(backend, FT, n_L)
    y_d        = KernelAbstractions.allocate(backend, FT, n_L)
    Snorm_d    = KernelAbstractions.allocate(backend, FT, n_L)
    H_cutoff_d = KernelAbstractions.allocate(backend, FT, n_L)
    copyto!(ν0_d,       ν0)
    copyto!(f_d,        f_arr)
    copyto!(y_d,        y_arr)
    copyto!(Snorm_d,    Snorm)
    copyto!(H_cutoff_d, H_cutoff)

    kernel! = voigt_cross_section_kernel!(backend, 256)
    kernel!(σ, ν_d, ν0_d, f_d, y_d, Snorm_d, H_cutoff_d, FT(cutoff); ndrange=n_ν)
    KernelAbstractions.synchronize(backend)

    return Float64.(Array(σ))
end
