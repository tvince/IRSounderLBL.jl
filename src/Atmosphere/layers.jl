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
    layer_properties(prof)

Interpolate an `AtmosphericProfile` onto layer mid-points.

Returns a NamedTuple with:
- `p_mid`:  mid-layer pressure (hPa)
- `T_mid`:  mid-layer temperature (K)
- `Δp`:     layer pressure thickness (hPa)
- `vmr_mid`: Dict{GasSpecies, Vector{Float64}} of mid-layer VMRs
"""
function layer_properties(prof::AtmosphericProfile)
    p_mid, Δp = pressure_layers(prof.pressure)
    n_layers = length(p_mid)

    T_mid = interp_profile(prof.pressure, prof.temperature, p_mid)

    vmr_mid = Dict{GasSpecies, Vector{Float64}}()
    for (sp, vmr) in prof.vmr
        vmr_mid[sp] = interp_vmr(prof.pressure, vmr, p_mid)
    end

    return (p_mid=p_mid, T_mid=T_mid, Δp=Δp, vmr_mid=vmr_mid)
end

"""
    column_amount(vmr_mid, Δp_hPa, dry_air_column)

Convert VMR and layer Δp to molecular column density (molec/cm²).

Uses hydrostatic relation: N = VMR × Δp × Nₐ / (g × Mair)
with Δp in Pa, g=9.80665 m/s², Mair=0.028964 kg/mol.
"""
function column_amount(vmr::Float64, Δp_hPa::Float64)
    const_Nair = 2.1209e26   # molec/m² per hPa (= Nₐ/(g·Mair) × 100)
    return vmr * Δp_hPa * const_Nair * 1e-4  # molec/cm²
end
