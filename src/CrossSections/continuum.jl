"""
MT-CKD (Mlawer–Tobin–Clough–Kneizys–Davies) water vapor and CO2 continuum
absorption.

This module implements simplified analytical fits to the MT-CKD 3.5 continuum,
suitable for research and operational forward modelling.

Reference: Mlawer et al. (2012), Phil. Trans. R. Soc. A 370, 2520–2556.
           Clough et al. (1989), JQSRT 41, 209–217.
"""

# Physical constants
const _KB = 1.380649e-23   # J/K
const _NA = 6.02214076e23  # mol⁻¹
const _P0 = 1013.25        # reference pressure, hPa

"""
    h2o_self_continuum_coeff(ν) -> Float64

Self-continuum absorption coefficient for water vapor at 296 K (cm²/molec/atm).
Polynomial fit to MT-CKD 3.5 values in the 0–3000 cm⁻¹ window.
"""
function h2o_self_continuum_coeff(ν::Float64)::Float64
    # Piecewise representation of the self-continuum
    if ν < 100.0
        return 0.0
    elseif ν <= 600.0
        # Microwave/far-IR window: relatively flat
        return 6.0e-24
    elseif ν <= 1250.0
        # Rotational band vicinity — use exponential decay from band edges
        c = 6.0e-24
        return c * exp(-2.0e-3 * (ν - 600.0))
    elseif ν <= 1700.0
        # 8–12 μm window
        return 2.0e-26
    elseif ν <= 2200.0
        # Between 6 μm and 4.3 μm windows
        return 4.0e-25 * exp(-3.5e-4 * (ν - 1700.0))
    elseif ν <= 2800.0
        # 4 μm window
        return 1.0e-26
    else
        return 1.0e-27
    end
end

"""
    h2o_foreign_continuum_coeff(ν) -> Float64

Foreign (air-broadened) continuum coefficient at 296 K (cm²/molec/atm).
Approximately 5× smaller than self-continuum.
"""
function h2o_foreign_continuum_coeff(ν::Float64)::Float64
    return h2o_self_continuum_coeff(ν) * 0.2
end

"""
    h2o_continuum(ν_grid, vmr_h2o, p_hPa, T) -> Vector{Float64}

Compute MT-CKD water vapour continuum absorption coefficient k_cont
(cm⁻¹) for each wavenumber in `ν_grid`.

Uses the Clough (1992) temperature scaling for the self continuum:
    C_s(T) = C_s(296) × exp[−1800 × (1/T − 1/296)]
"""
function h2o_continuum(ν_grid::WavenumberGrid,
                       vmr_h2o::Float64,
                       p_hPa::Float64,
                       T::Float64)::Vector{Float64}
    p_atm  = p_hPa / 1013.25
    p_h2o  = vmr_h2o * p_atm   # partial pressure of H2O (atm)
    p_dry  = p_atm - p_h2o

    # MT-CKD temperature scaling exponents (approximate)
    T_scale_self    = exp(-1800.0 * (1.0/T - 1.0/296.0))
    T_scale_foreign = exp( -700.0 * (1.0/T - 1.0/296.0))

    # Number density of H2O (molec/cm³)
    n_h2o = vmr_h2o * p_hPa * 100.0 / (_KB * T) * 1e-6  # Pa→cm³

    k = Vector{Float64}(undef, ν_grid.n)
    for (i, ν) in enumerate(ν_grid.ν)
        Cs = h2o_self_continuum_coeff(ν)    * T_scale_self
        Cf = h2o_foreign_continuum_coeff(ν) * T_scale_foreign
        # k [cm⁻¹] = n_h2o × (Cs × p_h2o + Cf × p_dry)
        k[i] = n_h2o * (Cs * p_h2o + Cf * p_dry)
    end
    return k
end

"""
    co2_continuum(ν_grid, vmr_co2, p_hPa, T) -> Vector{Float64}

CO2 collision-induced absorption continuum (cm⁻¹).
Important in the 1200–1500 cm⁻¹ region (near-wing of ν₂ band).

Simplified fit based on Hartmann et al. (2002) broadband co2 CIA.
"""
function co2_continuum(ν_grid::WavenumberGrid,
                       vmr_co2::Float64,
                       p_hPa::Float64,
                       T::Float64)::Vector{Float64}
    p_atm = p_hPa / 1013.25
    n_co2 = vmr_co2 * p_hPa * 100.0 / (_KB * T) * 1e-6  # molec/cm³

    # CO2 CIA coefficient (cm⁵/molec²) — approximate fit
    function co2_cia(ν)
        if 1200.0 <= ν <= 1500.0
            return 1.0e-47 * exp(-((ν - 1350.0)/150.0)^2)
        elseif 2300.0 <= ν <= 2400.0
            return 5.0e-48
        else
            return 0.0
        end
    end

    n_air = p_hPa * 100.0 / (_KB * T) * 1e-6  # total number density

    k = Vector{Float64}(undef, ν_grid.n)
    for (i, ν) in enumerate(ν_grid.ν)
        k[i] = co2_cia(ν) * n_co2 * n_air * (296.0/T)^0.5
    end
    return k
end
