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
    g = compute_voigt_cross_sections_grad(ν_grid, linelist, T, p_atm;
                                          vmr_self=vmr_self, cutoff=cutoff, x_far=x_far)
    return g.σ, g.dT
end

"""
    compute_voigt_cross_sections_grad(ν_grid, linelist, T, p_atm;
                                      vmr_self=0.0, cutoff=25.0, x_far=_X_FAR)
        -> (σ, dT, dp, dself)

Cross section and its derivatives w.r.t. temperature `T` (K), pressure `p_atm` (atm),
and the self-broadening fraction `vmr_self`, in a single Faddeeva pass — the building
block for the temperature Jacobian (`dT`) and the Phase-2b VMR coupling (`dp`, `dself`).
`σ` matches `compute_voigt_cross_sections` to round-off.

Only `S`, `γ_L`, `γ_D`, and the line centre carry these variables:
`T` → `S(T)`, `γ_L∝(T_ref/T)^n`, `γ_D∝√T`; `p` → `γ_L∝p` and the shift `ν₀+δ·p`;
`vmr_self` → the air/self mix of `γ_L` and of `δ`. Each derivative is
`Σ_j [∂Snorm·(H−ped) + Snorm·(∂H−∂ped)]`, where `∂H = H_x·∂x + H_y·∂y` and the pedestal
prefactor `2−(Δν/cutoff)²` itself moves with `ν₀` for the `p`/`vmr_self` derivatives
(but not `T`). The `max(·,0)` clamp zeroes every derivative where `σ` is clamped.
"""
function compute_voigt_cross_sections_grad(ν_grid::WavenumberGrid,
                                           linelist::HITRANLinelist,
                                           T::Float64,
                                           p_atm::Float64;
                                           vmr_self::Float64 = 0.0,
                                           cutoff::Float64 = 25.0,
                                           x_far::Float64 = _X_FAR)
    n_ν = ν_grid.n
    n_L = length(linelist.lines)

    ν0    = Vector{Float64}(undef, n_L)
    f_arr = Vector{Float64}(undef, n_L)
    df_arr= Vector{Float64}(undef, n_L)      # ∂f/∂T
    y_arr = Vector{Float64}(undef, n_L)
    dyT   = Vector{Float64}(undef, n_L)       # ∂y/∂T
    dyp   = Vector{Float64}(undef, n_L)       # ∂y/∂p
    dyv   = Vector{Float64}(undef, n_L)       # ∂y/∂vmr_self
    dν0p  = Vector{Float64}(undef, n_L)       # ∂ν0/∂p   = δ_eff
    dν0v  = Vector{Float64}(undef, n_L)       # ∂ν0/∂vmr_self
    Sn    = Vector{Float64}(undef, n_L)
    dSnT  = Vector{Float64}(undef, n_L)        # ∂Snorm/∂T (∂Snorm/∂p = ∂Snorm/∂vs = 0)
    Hc    = Vector{Float64}(undef, n_L)
    dHcT  = Vector{Float64}(undef, n_L)
    dHcp  = Vector{Float64}(undef, n_L)
    dHcv  = Vector{Float64}(undef, n_L)

    @inbounds for (j, line) in enumerate(linelist.lines)
        mol     = Int(line.mol_id)
        n_air_  = Float64(line.temp_depend)
        n_self_ = get(_N_SELF, mol, n_air_)
        δ_air   = Float64(line.pressure_shift)
        δ_self  = get(_DELTA_SELF, mol, 0.0)
        ν0[j]   = pressure_shift(line, p_atm; vmr_self=vmr_self)
        dν0p[j] = δ_air * (1.0 - vmr_self) + δ_self * vmr_self           # ∂ν0/∂p
        dν0v[j] = (δ_self - δ_air) * p_atm                              # ∂ν0/∂vmr_self

        S, dS = temperature_scaled_intensity_deriv(line, T)
        gl, gd, dglT, dgd = pressure_broadened_width_deriv(line, p_atm, T; vmr_self=vmr_self)
        gd = max(gd, 1e-10)
        f  = _SQRT_LN2 / gd
        df = -f * (dgd / gd)
        # γ_L air/self coefficients (per atm) for the p and vmr_self derivatives.
        t_ratio = T_REF / T
        a_coef  = t_ratio^n_air_  * line.air_broad
        b_coef  = t_ratio^n_self_ * line.self_broad
        dglp = gl / p_atm                                               # ∂γ_L/∂p
        dglv = (b_coef - a_coef) * p_atm                                # ∂γ_L/∂vmr_self

        f_arr[j]  = f
        df_arr[j] = df
        y_arr[j]  = gl * f
        dyT[j]    = dglT * f + gl * df
        dyp[j]    = dglp * f
        dyv[j]    = dglv * f
        Sn[j]     = S * f * _INV_SQRT_PI
        dSnT[j]   = (dS * f + S * df) * _INV_SQRT_PI

        x_cut = cutoff * f
        Hcv, Hcx, Hcy = _voigt_H_and_grad(x_cut, y_arr[j], x_far)
        Hc[j]   = Hcv
        dHcT[j] = Hcx * (cutoff * df) + Hcy * dyT[j]     # x_cut moves with T (via f)
        dHcp[j] = Hcy * dyp[j]                           # x_cut p-independent
        dHcv[j] = Hcy * dyv[j]
    end

    σ  = zeros(Float64, n_ν)
    dT = zeros(Float64, n_ν)
    dp = zeros(Float64, n_ν)
    dv = zeros(Float64, n_ν)
    invc = 1.0 / cutoff
    invc2 = invc * invc
    @inbounds for i in 1:n_ν
        ν = ν_grid.ν[i]
        acc = 0.0; aT = 0.0; ap = 0.0; av = 0.0
        for j in 1:n_L
            Δν = ν - ν0[j]
            abs(Δν) > cutoff && continue
            f = f_arr[j]
            x = Δν * f
            H, Hx, Hy = _voigt_H_and_grad(x, y_arr[j], x_far)
            # ∂x: T via f, p/vs via the centre shift (∂x/∂{p,vs} = −∂ν0·f).
            dHt = Hx * (Δν * df_arr[j]) + Hy * dyT[j]
            dHp = Hx * (-dν0p[j] * f)   + Hy * dyp[j]
            dHv = Hx * (-dν0v[j] * f)   + Hy * dyv[j]
            pedfac = 2.0 - (Δν * invc)^2
            # Pedestal prefactor moves with ν0 for p/vs (Δν depends on the centre).
            dpedfac_p = 2.0 * Δν * dν0p[j] * invc2
            dpedfac_v = 2.0 * Δν * dν0v[j] * invc2
            Hcj = Hc[j]
            dpedT = pedfac * dHcT[j]
            dpedp = dpedfac_p * Hcj + pedfac * dHcp[j]
            dpedv = dpedfac_v * Hcj + pedfac * dHcv[j]
            Hmped = H - pedfac * Hcj
            Snj = Sn[j]
            acc += Snj * Hmped
            aT  += dSnT[j] * Hmped + Snj * (dHt - dpedT)
            ap  += Snj * (dHp - dpedp)
            av  += Snj * (dHv - dpedv)
        end
        if acc > 0.0
            σ[i] = acc; dT[i] = aT; dp[i] = ap; dv[i] = av
        end   # clamped: σ = 0 and all derivatives 0
    end

    return (σ=σ, dT=dT, dp=dp, dself=dv)
end

# ── Cross-section gradient with optional line mixing (roadmap §6.4) ───────────
#
# `_species_cross_section_grad` is the line-mixing-aware sibling of
# `compute_voigt_cross_sections_grad`, mirroring `_species_cross_section`: it
# returns `(σ, dT, dp, dself)` for the species' *implemented* cross section so the
# Jacobian works with LM active. The plain-Voigt baseline keeps its analytic
# derivatives; the additive LM perturbation Δσ (VP_Y dispersive or VP_W
# eigendecomposition — see §6.4: the eigenvector perturbation is materially
# harder, so it is finite-differenced) is central-differenced in `(T, p)`. The
# perturbation is cheap relative to the Voigt baseline (a handful of bands), so a
# localized 2-point FD per derivative — the same technique Phase 2b/2c use for the
# Curtis-Godson and continuum couplings — is robust and matches the forward.

const _LM_GRAD_DT = 0.02      # K, central-FD step for ∂Δσ/∂T
const _LM_GRAD_DP_REL = 1e-4  # relative central-FD step for ∂Δσ/∂p_atm

# nothing / non-LM model → the analytic Voigt grad (exact, one Faddeeva pass).
function _species_cross_section_grad(::Nothing, sp::GasSpecies,
                                     ν_grid::WavenumberGrid, ll::HITRANLinelist,
                                     T::Float64, p_atm::Float64;
                                     vmr_self::Float64 = 0.0, cutoff::Float64 = 25.0,
                                     x_far::Float64 = _X_FAR, backend = nothing)
    compute_voigt_cross_sections_grad(ν_grid, ll, T, p_atm;
                                      vmr_self=vmr_self, cutoff=cutoff, x_far=x_far)
end

function _species_cross_section_grad(lm::AbstractLineMixing, sp::GasSpecies,
                                     ν_grid::WavenumberGrid, ll::HITRANLinelist,
                                     T::Float64, p_atm::Float64;
                                     vmr_self::Float64 = 0.0, cutoff::Float64 = 25.0,
                                     x_far::Float64 = _X_FAR, backend = nothing)
    # Species not handled by this LM model fall through to the analytic Voigt grad.
    sp == lm.data.species || return compute_voigt_cross_sections_grad(ν_grid, ll, T, p_atm;
                                          vmr_self=vmr_self, cutoff=cutoff, x_far=x_far)

    g = compute_voigt_cross_sections_grad(ν_grid, ll, T, p_atm;
                                          vmr_self=vmr_self, cutoff=cutoff, x_far=x_far)

    # LM perturbation at the centre and its central-FD derivatives in (T, p).
    Δσ0 = _lm_perturbation(lm, ν_grid, T, p_atm; cutoff=cutoff)
    hT  = _LM_GRAD_DT
    dΔT = (_lm_perturbation(lm, ν_grid, T + hT, p_atm; cutoff=cutoff) .-
           _lm_perturbation(lm, ν_grid, T - hT, p_atm; cutoff=cutoff)) ./ (2hT)
    hp  = max(p_atm, 1e-6) * _LM_GRAD_DP_REL
    dΔp = (_lm_perturbation(lm, ν_grid, T, p_atm + hp; cutoff=cutoff) .-
           _lm_perturbation(lm, ν_grid, T, p_atm - hp; cutoff=cutoff)) ./ (2hp)

    # Total cross section with the forward's max(σ_voigt + Δσ, 0) clamp; the
    # derivatives are the summed sensitivities, zeroed wherever the total is clamped.
    σ  = g.σ .+ Δσ0
    dT = g.dT .+ dΔT
    dp = g.dp .+ dΔp
    @inbounds for i in eachindex(σ)
        if σ[i] <= 0.0
            σ[i] = 0.0; dT[i] = 0.0; dp[i] = 0.0
        end
    end
    # dself: only H₂O (vmr_self ≠ 0) consumes it, and the LM species is not H₂O,
    # so the analytic Voigt dself passes through unused (and Δσ has no vmr_self
    # dependence). Returned for signature parity with the no-LM grad.
    return (σ=σ, dT=dT, dp=dp, dself=g.dself)
end
