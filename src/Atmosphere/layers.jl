"""
    pressure_layers(pressure_levels)

Compute mid-layer pressures and layer pressure thicknesses from
pressure level boundaries (hPa).

Returns `(p_mid, Δp)` where `p_mid[i]` is the average pressure of layer `i`
and `Δp[i]` is the pressure thickness (hPa).
"""
function pressure_layers(pressure_levels::AbstractVector{<:Real})
    n = length(pressure_levels)
    n_layers = n - 1
    p_mid = Vector{Float64}(undef, n_layers)
    Δp    = Vector{Float64}(undef, n_layers)
    for i in 1:n_layers
        p_mid[i] = 0.5 * (pressure_levels[i] + pressure_levels[i+1])
        Δp[i]    = abs(pressure_levels[i] - pressure_levels[i+1])
    end
    return p_mid, Δp
end

"""
    cg_column_vmr(v1, v2, p1, p2) -> Float64

Curtis-Godson effective VMR for the column integral of a layer:

    vmr_eff = ∫ VMR(p) dp / |Δp|

Assumes a power-law profile VMR(p) = v1 × (p/p1)^α between the level
boundaries (p1, v1) and (p2, v2), where p1 > p2 (p1 is the surface side).

This gives the exact column integral for an exponentially varying VMR,
which is the physically motivated model for species like H2O and O3.
Degenerates to the arithmetic mean when v1 ≈ v2 (constant VMR, e.g. CO2).
"""
function cg_column_vmr(v1::Float64, v2::Float64, p1::Float64, p2::Float64)::Float64
    (v1 <= 0.0 || v2 <= 0.0) && return 0.5 * (v1 + v2)
    v1 ≈ v2 && return v1
    α = log(v2 / v1) / log(p2 / p1)
    r = p2 / p1          # < 1 (p2 is upper, lower pressure)
    β = α + 1
    # Limit as β → 0 (VMR ∝ 1/p): log-mean VMR
    abs(β) < 1e-8 && return (v1 - v2) / log(v1 / v2)
    return v1 * p1 * (1.0 - r^β) / (β * abs(p1 - p2))
end

"""
    cg_pressure(v1, v2, p1, p2) -> Float64

Curtis-Godson effective pressure for evaluating pressure-broadened line shapes:

    p̄ = ∫ p × VMR(p) dp / ∫ VMR(p) dp

This is the VMR-weighted mean pressure, which is the correct pressure for
evaluating the Lorentz half-width (γ_L ∝ p) under the CG approximation.

Same power-law profile assumption as `cg_column_vmr`.
Degenerates to the arithmetic mean pressure when v1 ≈ v2.
"""
function cg_pressure(v1::Float64, v2::Float64, p1::Float64, p2::Float64)::Float64
    (v1 <= 0.0 || v2 <= 0.0) && return 0.5 * (p1 + p2)
    v1 ≈ v2 && return 0.5 * (p1 + p2)
    α  = log(v2 / v1) / log(p2 / p1)
    r  = p2 / p1
    β1 = α + 1
    β2 = α + 2
    # Limit as β1 → 0: log-mean pressure
    abs(β1) < 1e-8 && return abs(p1 - p2) / log(p1 / p2)
    # Limit as β2 → 0 (rare): series expansion gives p2·ln(p2/p1)/(p2/p1 - 1)
    abs(β2) < 1e-8 && return p2 * log(p2 / p1) / (p2 / p1 - 1.0)
    return p1 * β1 / β2 * (1.0 - r^β2) / (1.0 - r^β1)
end

"""
    cg_temperature_mass(T1, T2, p1, p2) -> Float64

Mass-weighted layer T = ∫ T(p) dp / |Δp| under the assumption that T varies
linearly with log(p) between (p1, T1) and (p2, T2). Closed form:

    frac = [(α - 1)·β + 1] / [α·(β - 1)],   α = log(p2/p1), β = p2/p1
    T_mass = T1 + (T2 - T1)·frac

This is the convention LBLRTM's LBLATM uses for the per-layer effective T
(printed as TBAR in TAPE6), which differs from Julia's default `cg_pressure +
T(p_cg)` recipe by the curvature of T(log p) within the layer. The two agree
to leading order in α; they diverge for thick layers with steep T gradient,
producing a sign-flipping offset around the stratopause/mesopause turning
points where T(z) has an extremum.

For |α| < 1e-3 (thin layer), uses the Taylor series  frac ≈ 1/2 + α/12 to
avoid catastrophic cancellation in (β − 1).
"""
function cg_temperature_mass(T1::Float64, T2::Float64,
                              p1::Float64, p2::Float64)::Float64
    T1 ≈ T2 && return T1
    p1 ≈ p2 && return 0.5 * (T1 + T2)
    α = log(p2 / p1)
    if abs(α) < 1e-3
        # Series: frac = 1/2 + α/12 + O(α³)
        frac = 0.5 + α / 12.0
    else
        β = p2 / p1
        frac = ((α - 1.0) * β + 1.0) / (α * (β - 1.0))
    end
    return T1 + (T2 - T1) * frac
end

"""
    layer_properties(prof; T_method=:logp_at_pcg)

Compute layer properties from an `AtmosphericProfile` for the Schwarzschild RTE.

# Keyword arguments
- `T_method`: how to compute the per-layer effective temperature for LBL
  cross-section evaluation.
  - `:logp_at_pcg`  (default): T linearly interpolated in log(p) at the
                    Curtis-Godson pressure p_cg. Optimal single-point quadrature
                    for a power-law VMR(p) profile.
  - `:mass_weighted`: mass-weighted layer T = ∫T dp / Δp (LBLATM convention,
                    matches the TBAR printed in LBLRTM TAPE6). Better for
                    layers with strong T gradient relative to the level spacing
                    (e.g., upper-stratosphere/mesosphere with 5 km layers).

Returns a NamedTuple with:
- `p_mid`:   arithmetic mid-layer pressure (hPa) — for continuum, dz, Planck function
- `T_mid`:   temperature at `p_mid` (K) — for Planck source function
- `Δp`:      layer pressure thickness (hPa)
- `vmr_mid`: Dict of linearly-interpolated mid-layer VMRs — for continuum absorption
- `vmr_cg`:  Dict of Curtis-Godson effective VMRs — for LBL column amounts
- `p_cg`:    Dict of CG pressure-weighted pressures (hPa) — for LBL cross-section evaluation
- `T_cg`:    Dict of effective temperatures (K) per the `T_method` choice
"""
function layer_properties(prof::AtmosphericProfile; T_method::Symbol = :logp_at_pcg)
    T_method in (:logp_at_pcg, :mass_weighted) ||
        error("T_method must be :logp_at_pcg or :mass_weighted, got :$T_method")

    p_lev = prof.pressure          # level pressures (n_lev)
    T_lev = prof.temperature
    n_lay = length(p_lev) - 1

    p_mid, Δp = pressure_layers(p_lev)
    T_mid = interp_profile(p_lev, T_lev, p_mid)

    # Linear-interpolated VMR at arithmetic mid-layer (for continuum etc.)
    vmr_mid = Dict{GasSpecies, Vector{Float64}}()
    for (sp, vmr) in prof.vmr
        vmr_mid[sp] = interp_vmr(p_lev, vmr, p_mid)
    end

    # Mass-weighted T per layer is gas-independent — precompute once.
    T_mass = if T_method == :mass_weighted
        [cg_temperature_mass(T_lev[k], T_lev[k+1], p_lev[k], p_lev[k+1])
         for k in 1:n_lay]
    else
        nothing
    end

    # Curtis-Godson effective VMR and pressure per species
    vmr_cg = Dict{GasSpecies, Vector{Float64}}()
    p_cg   = Dict{GasSpecies, Vector{Float64}}()
    T_cg   = Dict{GasSpecies, Vector{Float64}}()
    for (sp, vmr) in prof.vmr
        vc = Vector{Float64}(undef, n_lay)
        pc = Vector{Float64}(undef, n_lay)
        for k in 1:n_lay
            vc[k] = cg_column_vmr(vmr[k], vmr[k+1], p_lev[k], p_lev[k+1])
            pc[k] = cg_pressure(vmr[k], vmr[k+1], p_lev[k], p_lev[k+1])
        end
        vmr_cg[sp] = vc
        p_cg[sp]   = pc
        T_cg[sp]   = T_method == :mass_weighted ? copy(T_mass) :
                     interp_profile(p_lev, T_lev, pc)
    end

    return (p_mid=p_mid, T_mid=T_mid, Δp=Δp,
            vmr_mid=vmr_mid,
            vmr_cg=vmr_cg, p_cg=p_cg, T_cg=T_cg)
end

"""
    column_amount(vmr, Δp_hPa) -> Float64

Convert VMR and layer Δp to molecular column density (molec/cm²).

Uses hydrostatic relation: N = VMR × Δp × Nₐ / (g × Mair)
with Δp in Pa, g=9.80665 m/s², Mair=0.028964 kg/mol.
"""
function column_amount(vmr::Float64, Δp_hPa::Float64)
    const_Nair = 2.1209e26   # molec/m² per hPa (= Nₐ/(g·Mair) × 100)
    return vmr * Δp_hPa * const_Nair * 1e-4  # molec/cm²
end
