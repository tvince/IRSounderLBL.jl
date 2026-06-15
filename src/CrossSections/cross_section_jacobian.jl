"""
Temperature derivative of the Voigt cross section (roadmap Phase 3, the hard atom).

`compute_voigt_cross_sections_dT` returns `(σ, ∂σ/∂T)` on the grid, reproducing
`compute_voigt_cross_sections` (FullFaddeeva path) for `σ` exactly — same AER
parabolic pedestal, same far-wing Lorentzian shortcut, same `max(·,0)` clamp — and
differentiating each w.r.t. temperature.

## The chain

Per line the contribution is `Snorm·(H − ped)`, `Snorm = S·f/√π`, `f = √ln2/γ_D`,
`x = Δν·f`, `y = γ_L·f`. Only `S`, `γ_L`, `γ_D` carry T (the pressure-shifted centre
`ν₀` does not), so `dx/dT = Δν·f′`, `dy/dT = γ_L′·f + γ_L·f′`, with
`f′ = −f·(γ_D′/γ_D)`. The Faddeeva gradient uses `w′(z) = −2z·w + 2i/√π` with
`w = K + iL` (one `erfcx`):

    ∂H/∂x = −2xK + 2yL,   ∂H/∂y = 2(xL + yK) − 2/√π,
    ∂H/∂T = ∂H/∂x·dx/dT + ∂H/∂y·dy/dT.

The pedestal `ped = (2 − (Δν/cutoff)²)·H_cutoff` differentiates through `H_cutoff`
(at `x_cut = cutoff·f`, far-wing branch). The clamp gives `∂σ/∂T = 0` wherever the
summed value is clamped to 0 (shared discontinuity, roadmap §6.1).
"""

"""
    _faddeeva_w(x, y) -> (K, L)

Real and imaginary parts of the Faddeeva function `w(x+iy) = erfcx(y − ix)`.
`K = Re w` is the Voigt H-function (`faddeeva_voigt`); `L = Im w` is the dispersion
term needed for the analytic gradient.
"""
@inline function _faddeeva_w(x::Float64, y::Float64)
    w = erfcx(complex(y, -x))
    return real(w), imag(w)
end

"""
    _voigt_H_and_grad(x, y, x_far) -> (H, ∂H/∂x, ∂H/∂y)

Voigt H-function and its `(x, y)` gradient, with the same far-wing Lorentzian
shortcut as the forward kernel (`|x| > x_far` → `H = y/(√π(x²+y²))`).
"""
@inline function _voigt_H_and_grad(x::Float64, y::Float64, x_far::Float64)
    if abs(x) > x_far
        d  = x*x + y*y
        d2 = d * d
        H  = y * _INV_SQRT_PI / d
        Hx = -2.0 * x * y * _INV_SQRT_PI / d2
        Hy = _INV_SQRT_PI * (x*x - y*y) / d2
        return H, Hx, Hy
    end
    K, L = _faddeeva_w(x, y)
    Hx = -2.0 * x * K + 2.0 * y * L
    Hy = 2.0 * (x * L + y * K) - 2.0 * _INV_SQRT_PI
    return K, Hx, Hy
end

"""
    compute_voigt_cross_sections_dT(ν_grid, linelist, T, p_atm;
                                    vmr_self=0.0, cutoff=25.0, x_far=_X_FAR)
        -> (σ, dσdT)

Absorption cross sections (cm²/molec) and their temperature derivative (cm²/molec/K)
on `ν_grid` at temperature `T` (K) and pressure `p_atm` (atm). `σ` matches
`compute_voigt_cross_sections` (FullFaddeeva) to round-off. CPU/Float64; the analytic
companion of the forward kernel for the temperature Jacobian.
"""
function compute_voigt_cross_sections_dT(ν_grid::WavenumberGrid,
                                         linelist::HITRANLinelist,
                                         T::Float64,
                                         p_atm::Float64;
                                         vmr_self::Float64 = 0.0,
                                         cutoff::Float64 = 25.0,
                                         x_far::Float64 = _X_FAR)
    n_ν = ν_grid.n
    n_L = length(linelist.lines)

    ν0     = Vector{Float64}(undef, n_L)
    f_arr  = Vector{Float64}(undef, n_L)
    df_arr = Vector{Float64}(undef, n_L)
    y_arr  = Vector{Float64}(undef, n_L)
    dy_arr = Vector{Float64}(undef, n_L)
    Sn     = Vector{Float64}(undef, n_L)
    dSn    = Vector{Float64}(undef, n_L)
    Hc     = Vector{Float64}(undef, n_L)
    dHc    = Vector{Float64}(undef, n_L)

    @inbounds for (j, line) in enumerate(linelist.lines)
        ν0[j] = pressure_shift(line, p_atm; vmr_self=vmr_self)   # T-independent
        S, dS = temperature_scaled_intensity_deriv(line, T)
        gl, gd, dgl, dgd = pressure_broadened_width_deriv(line, p_atm, T; vmr_self=vmr_self)
        gd  = max(gd, 1e-10)
        f   = _SQRT_LN2 / gd
        df  = -f * (dgd / gd)
        f_arr[j]  = f
        df_arr[j] = df
        y_arr[j]  = gl * f
        dy_arr[j] = dgl * f + gl * df
        Sn[j]     = S * f * _INV_SQRT_PI
        dSn[j]    = (dS * f + S * df) * _INV_SQRT_PI
        # Pedestal anchor at the cutoff (far-wing branch in practice).
        x_cut  = cutoff * f
        dx_cut = cutoff * df
        Hcv, Hcx, Hcy = _voigt_H_and_grad(x_cut, y_arr[j], x_far)
        Hc[j]  = Hcv
        dHc[j] = Hcx * dx_cut + Hcy * dy_arr[j]
    end

    σ    = zeros(Float64, n_ν)
    dσdT = zeros(Float64, n_ν)
    invc = 1.0 / cutoff
    @inbounds for i in 1:n_ν
        ν    = ν_grid.ν[i]
        acc  = 0.0
        dacc = 0.0
        for j in 1:n_L
            Δν = ν - ν0[j]
            abs(Δν) > cutoff && continue
            x  = Δν * f_arr[j]
            dx = Δν * df_arr[j]
            H, Hx, Hy = _voigt_H_and_grad(x, y_arr[j], x_far)
            dH = Hx * dx + Hy * dy_arr[j]
            pedfac = 2.0 - (Δν * invc)^2
            ped    = pedfac * Hc[j]
            dped   = pedfac * dHc[j]
            acc  += Sn[j]  * (H - ped)
            dacc += dSn[j] * (H - ped) + Sn[j] * (dH - dped)
        end
        if acc > 0.0
            σ[i]    = acc
            dσdT[i] = dacc
        end   # clamped region: σ = 0, dσ/dT = 0
    end

    return σ, dσdT
end
