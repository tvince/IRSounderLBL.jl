"""
Layer optical depth calculation.

Combines HITRAN line-by-line absorption, MT-CKD continuum, and
hydrostatic column amounts to produce layer optical depths τ(ν).
"""

"""
    layer_optical_depth(ν_grid, linelist, vmr_h2o, vmr_co2,
                        p_hPa, T, Δp_hPa;
                        cutoff=25.0, backend=CPU()) -> Vector{Float64}

Compute the optical depth τ(ν) for a single atmospheric layer.

# Arguments
- `ν_grid`:    `WavenumberGrid` for the calculation
- `linelist`:  `HITRANLinelist` for the species in this layer
- `vmr_h2o`:   H₂O volume mixing ratio (mol/mol)
- `vmr_co2`:   CO₂ volume mixing ratio (mol/mol)
- `p_hPa`:     mid-layer pressure (hPa)
- `T`:         mid-layer temperature (K)
- `Δp_hPa`:    pressure thickness of layer (hPa)
- `cutoff`:    Voigt line wing cutoff (cm⁻¹)
- `backend`:   KernelAbstractions compute backend

# Returns
τ(ν) vector of optical depths (dimensionless), one value per grid point.
"""
function layer_optical_depth(ν_grid::WavenumberGrid,
                              linelist::HITRANLinelist,
                              vmr_h2o::Float64,
                              vmr_co2::Float64,
                              p_hPa::Float64,
                              T::Float64,
                              Δp_hPa::Float64;
                              cutoff::Float64 = 25.0,
                              backend = CPU())
    p_atm = p_hPa / 1013.25

    # Molecular column density (molec/cm²) per unit VMR
    # N = VMR × Δp[Pa] × Nₐ / (g × Mair)
    # = VMR × Δp[hPa] × 100 × 6.02214e23 / (9.80665 × 0.028964)
    # = VMR × Δp[hPa] × 2.1521e25 × 1e-4  [per cm²]
    Nair_per_vmr = 2.1209e22  # molec/cm² per (VMR × hPa)

    # ── Line-by-line cross-sections ──────────────────────────────────────
    σ_lbl = compute_voigt_cross_sections(ν_grid, linelist, T, p_atm;
                                          cutoff=cutoff, backend=backend)

    # Sum τ for each species present in the linelist
    # For now assume the entire linelist is one species with a single VMR
    # (caller is responsible for splitting linelists by species)
    # Determine dominant molecule from linelist
    mol_vmr = _linelist_vmr(linelist, vmr_h2o, vmr_co2)
    N_col   = mol_vmr * Δp_hPa * Nair_per_vmr   # molec/cm²
    τ_lbl   = σ_lbl .* N_col

    # ── H₂O continuum ───────────────────────────────────────────────────
    k_h2o_cont = h2o_continuum(ν_grid, vmr_h2o, p_hPa, T)
    # k_cont [cm⁻¹] × dz [cm]; convert Δp to geometric thickness
    dz_cm = _dp_to_dz_cm(Δp_hPa, p_hPa, T)
    τ_h2o_cont = k_h2o_cont .* dz_cm

    # ── CO₂ continuum ───────────────────────────────────────────────────
    k_co2_cont = co2_continuum(ν_grid, vmr_co2, p_hPa, T)
    τ_co2_cont = k_co2_cont .* dz_cm

    return τ_lbl .+ τ_h2o_cont .+ τ_co2_cont
end

"""
    layer_optical_depth(ν_grid, linelists, layer; kwargs...) -> Vector{Float64}

Multi-species convenience method accepting a Dict of HITRANLinelists
keyed by GasSpecies and a layer NamedTuple from `layer_properties`.
"""
function layer_optical_depth(ν_grid::WavenumberGrid,
                              linelists::Dict{GasSpecies, HITRANLinelist},
                              vmr_mid::Dict{GasSpecies, Float64},
                              p_hPa::Float64,
                              T::Float64,
                              Δp_hPa::Float64;
                              kwargs...)
    τ = zeros(Float64, ν_grid.n)
    vmr_h2o = get(vmr_mid, H2O, 0.0)
    vmr_co2 = get(vmr_mid, CO2, 4.15e-4)

    for (sp, ll) in linelists
        vmr = get(vmr_mid, sp, 0.0)
        vmr == 0.0 && continue

        # Compute line-by-line OD for this species
        vmr_s = (sp == H2O) ? vmr : 0.0
        σ_sp = compute_voigt_cross_sections(ν_grid, ll, T,
                                             p_hPa / 1013.25;
                                             vmr_self=vmr_s, kwargs...)
        Nair_per_vmr = 2.1209e22
        N_col = vmr * Δp_hPa * Nair_per_vmr
        τ .+= σ_sp .* N_col
    end

    # Add continua once (avoid double-counting)
    dz_cm = _dp_to_dz_cm(Δp_hPa, p_hPa, T)
    τ .+= h2o_continuum(ν_grid, vmr_h2o, p_hPa, T) .* dz_cm
    τ .+= co2_continuum(ν_grid, vmr_co2, p_hPa, T) .* dz_cm

    return τ
end

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Convert pressure thickness Δp (hPa) to geometric thickness (cm) via hydrostatics."""
function _dp_to_dz_cm(Δp_hPa::Float64, p_mid_hPa::Float64, T::Float64)
    g    = 9.80665       # m/s²
    Mair = 0.028964      # kg/mol
    R    = 8.314462      # J/(mol·K)
    H    = R * T / (Mair * g)          # scale height (m)
    dz_m = H * Δp_hPa / p_mid_hPa    # hydrostatic approximation (m)
    return dz_m * 100.0                # → cm
end

"""Return a rough VMR for the dominant molecule in a linelist."""
function _linelist_vmr(linelist::HITRANLinelist,
                       vmr_h2o::Float64, vmr_co2::Float64)
    mols = linelist.molecules
    if 1 in mols
        return vmr_h2o
    elseif 2 in mols
        return vmr_co2
    else
        # Default to CO2-level for trace gases; caller should use the
        # multi-species method for proper treatment.
        return 1e-6
    end
end
