"""
Analytic Jacobian of the Schwarzschild RTE (roadmap Phase 1).

Differentiates `schwarzschild_rte` with respect to its own inputs — the per-layer
optical depths `τ_layers`, the level temperatures `T_levels`, the surface
temperature `T_sfc`, and the emissivity `ε` — holding the cross sections (hence the
mapping VMR/T → τ) fixed. This is the RTE backbone of the full weighting-function
chain; Phases 2–3 compose `∂I/∂τ` with `∂τ/∂VMR` and `∂τ/∂T`.

## The unified emission form

Both source functions write the TOA radiance as

    I = ε·B(T_sfc)·𝒯₁ + Σ_k 𝒯_{k+1}·em_k

where `𝒯_k` is the transmittance from level k to TOA (`𝒯_{k+1}` = top of layer k),
`a_k = τ_k/μ`, and `em_k` is the layer's downward-looking emission:

  - `:cim`  `em_k = (1−e^{−a})·(B_avg + (B_top−B_avg)·f_cim(a))`,
            `B_avg = B(T_ave_k)`, `B_top = B(T_lev_{k+1})`
  - `:toon` `em_k = B_top·(1−e^{−a}) + (B_bot−B_top)·f_lit(a)`,
            `B_bot = B(T_lev_k)`, `B_top = B(T_lev_{k+1})`

## Derivatives

- **ε:**     `∂I/∂ε = B(T_sfc)·𝒯₁`
- **T_sfc:** `∂I/∂T_sfc = ε·B′(T_sfc)·𝒯₁`   (`B′ = dB_dT`)
- **τ_m:**   `∂I/∂τ_m = (1/μ)·(−I_below(m) + 𝒯_{m+1}·∂em_m/∂a_m)`, where
             `I_below(m) = ε·B(T_sfc)·𝒯₁ + Σ_{k<m} 𝒯_{k+1}·em_k` is the TOA
             radiance contributed by the surface and the layers below m. (The
             attenuation of everything below m, plus the change in layer m's own
             emission.) Computed in O(n_layers) per channel with a running
             `I_below`. Verified against the isothermal-Kirchhoff limit (≡ 0).
- **T_lev[j]:** through the layer brightnesses only. For `:cim`, `T_lev[j]` enters
             `B_top_{j−1}` and the mass-weighted `T_ave` of layers j−1 and j;
             since `T_ave_k = T_lev_k·(1−frac_k) + T_lev_{k+1}·frac_k` is linear with
             pressure-only `frac_k` (= `cg_temperature_mass`), the chain rule is exact
             and CIM-consistent (roadmap §6.2). For `:toon`, `T_lev[j]` enters
             `B_bot_j` and `B_top_{j−1}` directly.

`source_function=:cim` requires `p_levels`; `T_ave` is recomputed internally from
`(T_levels, p_levels)` so the source and its derivative cannot drift apart.
"""

# frac weight of cg_temperature_mass (pressure-only); T_ave = T1·(1−frac) + T2·frac.
# Mirrors the branch logic of cg_temperature_mass exactly.
@inline function _cg_mass_frac(p1::Float64, p2::Float64)::Float64
    p1 ≈ p2 && return 0.5
    α = log(p2 / p1)
    abs(α) < 1e-3 && return 0.5 + α / 12.0
    β = p2 / p1
    return ((α - 1.0) * β + 1.0) / (α * (β - 1.0))
end

"""
    schwarzschild_rte_jacobian(ν_grid, τ_layers, T_levels, T_sfc;
                               μ=1.0, ε_sfc=1.0,
                               source_function=:toon, p_levels=nothing)
        -> (I, dI_dτ, dI_dTlev, dI_dTsfc, dI_dε)

Radiance and its analytic Jacobians w.r.t. the RTE inputs. `I` matches
`schwarzschild_rte` to round-off. Shapes: `dI_dτ` `(n_ν × n_layers)`, `dI_dTlev`
`(n_ν × n_levels)`, `dI_dTsfc` and `dI_dε` `(n_ν)`. See the module docstring for the
math. `:cim` requires `p_levels` (the length-`n_levels` level pressures, hPa).
"""
function schwarzschild_rte_jacobian(ν_grid::WavenumberGrid,
                                     τ_layers::AbstractMatrix{<:Real},
                                     T_levels::AbstractVector{<:Real},
                                     T_sfc::Float64;
                                     μ::Float64 = 1.0,
                                     ε_sfc::Float64 = 1.0,
                                     source_function::Symbol = :toon,
                                     p_levels::Union{AbstractVector{<:Real}, Nothing} = nothing)
    n_ν, n_layers = size(τ_layers)
    n_levels = n_layers + 1
    n_ν == ν_grid.n              || error("τ_layers rows $n_ν ≠ ν_grid.n $(ν_grid.n)")
    length(T_levels) == n_levels || error("T_levels must have n_layers+1 = $n_levels entries")
    0.0 < μ <= 1.0               || error("μ must be in (0, 1]")
    source_function in (:toon, :cim) ||
        error("source_function must be :toon or :cim, got :$source_function")

    # CIM: rebuild T_ave and the frac weights from (T_levels, p_levels) so the
    # derivative is consistent with the source (roadmap §6.2).
    is_cim = source_function == :cim
    T_ave  = Vector{Float64}(undef, n_layers)
    fracs  = Vector{Float64}(undef, n_layers)
    if is_cim
        isnothing(p_levels) && error(":cim source_function requires p_levels keyword")
        length(p_levels) == n_levels ||
            error("p_levels must have n_levels = $n_levels entries, got $(length(p_levels))")
        @inbounds for k in 1:n_layers
            p1, p2 = Float64(p_levels[k]), Float64(p_levels[k + 1])
            T1, T2 = Float64(T_levels[k]), Float64(T_levels[k + 1])
            T_ave[k] = cg_temperature_mass(T1, T2, p1, p2)
            fracs[k] = _cg_mass_frac(p1, p2)
        end
    end

    𝒯 = level_transmittances(τ_layers, μ)        # (n_ν × n_levels)
    invμ = 1.0 / μ

    I        = Vector{Float64}(undef, n_ν)
    dI_dτ    = Matrix{Float64}(undef, n_ν, n_layers)
    dI_dTlev = zeros(Float64, n_ν, n_levels)       # accumulated (each level touched twice)
    dI_dTsfc = Vector{Float64}(undef, n_ν)
    dI_dε    = Vector{Float64}(undef, n_ν)

    @inbounds for i in 1:n_ν
        ν    = ν_grid.ν[i]
        Bsfc = planck_radiance(ν, T_sfc)
        𝒯1   = 𝒯[i, 1]

        I_i          = ε_sfc * Bsfc * 𝒯1
        dI_dε[i]     = Bsfc * 𝒯1
        dI_dTsfc[i]  = ε_sfc * dB_dT(ν, T_sfc) * 𝒯1
        I_below      = ε_sfc * Bsfc * 𝒯1            # surface + layers below current m

        for m in 1:n_layers
            a    = τ_layers[i, m] * invμ
            w    = exp(-a)
            u    = 1.0 - w
            𝒯top = 𝒯[i, m + 1]

            if is_cim
                Bavg = planck_radiance(ν, T_ave[m])
                Btop = planck_radiance(ν, Float64(T_levels[m + 1]))
                f    = _cim_correction(a)
                fp   = _cim_correction_deriv(a)
                core   = Bavg + (Btop - Bavg) * f
                em     = u * core
                dem_da = w * core + u * (Btop - Bavg) * fp
                # T_lev derivatives: B_avg via T_ave (split k / k+1 by frac), B_top.
                cavg = 𝒯top * u * (1.0 - f) * dB_dT(ν, T_ave[m])
                fr   = fracs[m]
                dI_dTlev[i, m]     += cavg * (1.0 - fr)
                dI_dTlev[i, m + 1] += cavg * fr + 𝒯top * u * f * dB_dT(ν, Float64(T_levels[m + 1]))
            else
                Bbot = planck_radiance(ν, Float64(T_levels[m]))
                Btop = planck_radiance(ν, Float64(T_levels[m + 1]))
                fl   = _lit_correction(a)
                flp  = _lit_correction_deriv(a)
                em     = Btop * u + (Bbot - Btop) * fl
                dem_da = Btop * w + (Bbot - Btop) * flp
                dI_dTlev[i, m]     += 𝒯top * fl * dB_dT(ν, Float64(T_levels[m]))
                dI_dTlev[i, m + 1] += 𝒯top * (u - fl) * dB_dT(ν, Float64(T_levels[m + 1]))
            end

            # τ_m derivative needs I_below(m) BEFORE this layer is accumulated.
            dI_dτ[i, m] = invμ * (-I_below + 𝒯top * dem_da)

            I_i     += 𝒯top * em
            I_below += 𝒯top * em
        end

        I[i] = I_i
    end

    # Reflected downwelling (RFM Eq. 14 term 3) and its derivatives, mirroring
    # `add_reflected_downwelling!` in schwarzschild.jl. Skipped for ε==1.
    if ε_sfc < 1.0
        add_reflected_downwelling_jacobian!(I, dI_dτ, dI_dTlev, dI_dε,
                                            ν_grid, τ_layers, T_levels, 𝒯, μ,
                                            ε_sfc, is_cim, T_ave, fracs)
    end

    return I, dI_dτ, dI_dTlev, dI_dTsfc, dI_dε
end

"""
    add_reflected_downwelling_jacobian!(I, dI_dτ, dI_dTlev, dI_dε, ν_grid,
                                        τ_layers, T_levels, 𝒯, μ, ε_sfc,
                                        is_cim, T_ave, fracs)

Add the reflected-downwelling surface term `R = (1−ε)·I↓·𝒯₁` and its analytic
derivatives in place. `I↓ = Σ_k g_k·em↓_k` is the atmosphere-emitted downwelling
radiance at the surface, `g_k = Π_{j<k} e^{−a_j}` the level-k→surface
transmittance, `𝒯₁` the total surface→TOA transmittance, and `em↓_k` the layer's
**downward** emission (anchored at the lower boundary; see the forward helper).

Derivatives (the surface term carries no T_sfc dependence, so `dI_dTsfc` is
untouched):

  - `∂R/∂ε       = −I↓·𝒯₁`
  - `∂R/∂T_lev[·] = (1−ε)·𝒯₁·Σ_k g_k·∂em↓_k/∂T_lev[·]`
  - `∂R/∂τ_m     = (1−ε)·(1/μ)·𝒯₁·(−I↓_above_m + g_m·∂em↓_m/∂a_m − I↓)`,
    where `I↓_above_m = Σ_{k>m} g_k·em↓_k` (the part of `I↓` emitted above layer
    m, which both layer m attenuates and 𝒯₁ re-attenuates).
"""
function add_reflected_downwelling_jacobian!(I::Vector{Float64},
                                             dI_dτ::Matrix{Float64},
                                             dI_dTlev::Matrix{Float64},
                                             dI_dε::Vector{Float64},
                                             ν_grid::WavenumberGrid,
                                             τ_layers::AbstractMatrix{<:Real},
                                             T_levels::AbstractVector{<:Real},
                                             𝒯::AbstractMatrix{<:Real},
                                             μ::Float64,
                                             ε_sfc::Float64,
                                             is_cim::Bool,
                                             T_ave::AbstractVector{<:Real},
                                             fracs::AbstractVector{<:Real})
    n_ν, n_layers = size(τ_layers)
    invμ = 1.0 / μ
    refl = 1.0 - ε_sfc
    g    = Vector{Float64}(undef, n_layers)   # transmittance level k → surface
    em   = Vector{Float64}(undef, n_layers)   # downward emission of layer k
    dem  = Vector{Float64}(undef, n_layers)   # ∂em↓_k/∂a_k

    @inbounds for i in 1:n_ν
        ν  = ν_grid.ν[i]
        𝒯1 = 𝒯[i, 1]

        # Forward downward sweep: g_k, em↓_k, dem↓_k, I↓, and T_lev derivatives.
        gk     = 1.0
        I_down = 0.0
        for k in 1:n_layers
            a = τ_layers[i, k] * invμ
            w = exp(-a)
            u = 1.0 - w
            g[k] = gk

            if is_cim
                Bavg = planck_radiance(ν, T_ave[k])
                Bbot = planck_radiance(ν, Float64(T_levels[k]))
                f    = _cim_correction(a)
                fp   = _cim_correction_deriv(a)
                core   = Bavg + (Bbot - Bavg) * f
                em[k]  = u * core
                dem[k] = w * core + u * (Bbot - Bavg) * fp
                # T_lev derivs: B_avg via T_ave (split k/k+1 by frac), B_bot at k.
                cavg = refl * 𝒯1 * gk * u * (1.0 - f) * dB_dT(ν, T_ave[k])
                fr   = fracs[k]
                dI_dTlev[i, k]     += cavg * (1.0 - fr) +
                                      refl * 𝒯1 * gk * u * f * dB_dT(ν, Float64(T_levels[k]))
                dI_dTlev[i, k + 1] += cavg * fr
            else
                Bbot = planck_radiance(ν, Float64(T_levels[k]))
                Btop = planck_radiance(ν, Float64(T_levels[k + 1]))
                fl   = _lit_correction(a)
                flp  = _lit_correction_deriv(a)
                em[k]  = Bbot * u + (Btop - Bbot) * fl
                dem[k] = Bbot * w + (Btop - Bbot) * flp
                # em↓ = B_bot·(u−f) + B_top·f
                dI_dTlev[i, k]     += refl * 𝒯1 * gk * (u - fl) * dB_dT(ν, Float64(T_levels[k]))
                dI_dTlev[i, k + 1] += refl * 𝒯1 * gk * fl * dB_dT(ν, Float64(T_levels[k + 1]))
            end

            I_down += gk * em[k]
            gk     *= w
        end

        # Apply R to I and dI_dε.
        I[i]     += refl * I_down * 𝒯1
        dI_dε[i] += -I_down * 𝒯1

        # Backward sweep for τ derivatives, maintaining I↓_above_m.
        I_down_above = 0.0
        for m in n_layers:-1:1
            a = τ_layers[i, m] * invμ
            dI_dτ[i, m] += refl * invμ * 𝒯1 *
                           (-I_down_above + g[m] * dem[m] - I_down)
            I_down_above += g[m] * em[m]
        end
    end
    return I
end
