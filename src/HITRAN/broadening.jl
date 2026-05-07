"""
Pressure and temperature broadening of HITRAN spectral lines.

References:
  - HITRAN 2020 database paper (Gordon et al. 2022, JQSRT)
  - Voigt profile width formulae from Schreier (2011)
"""

# C2 = hc/k in cm·K (defined again locally to avoid circular dependency)
const _C2 = 1.4387752  # cm·K

"""
    pressure_broadened_width(line, p, T) -> (γ_L, γ_D)

Compute the Lorentzian (pressure) half-width γ_L and the Doppler half-width
γ_D (both HWHM, in cm⁻¹) for a given `HITRANLine` at pressure p (atm)
and temperature T (K).

Lorentzian half-width (air-broadened, self-broadening ignored here):
    γ_L(T, p) = (T_ref/T)^n × (γ_air × p)

Gaussian (Doppler) half-width:
    γ_D = ν₀/c × sqrt(2 ln2 × k T / M)
        = ν₀ × 3.5812e-7 × sqrt(T / M_amu)

where M_amu is the molecular mass in atomic mass units.
"""
function pressure_broadened_width(line::HITRANLine, p_atm::Float64, T::Float64)
    γ_L = (T_REF / T)^line.temp_depend * line.air_broad * p_atm

    # Molecular mass in amu (approximate, for Doppler width)
    M = _molecular_mass_amu(Int(line.mol_id))
    γ_D = line.wavenumber * 3.58126e-7 * sqrt(T / M)

    return Float64(γ_L), Float64(γ_D)
end

"""
    temperature_scaled_intensity(line, T) -> Float64

Scale the HITRAN 296 K reference intensity to temperature T (K) using the
HITRAN formula:

    S(T) = S_ref × [Q_ref/Q(T)] × exp[−C2 E″ (1/T − 1/T_ref)]
                 × [1 − exp(−C2 ν₀ / T)] / [1 − exp(−C2 ν₀ / T_ref)]
"""
function temperature_scaled_intensity(line::HITRANLine, T::Float64)
    ν₀ = line.wavenumber
    E″ = line.lower_energy
    Qr = Q_ratio(Int(line.mol_id), Int(line.iso_id), T)

    stim_factor = (1.0 - exp(-_C2 * ν₀ / T)) /
                  (1.0 - exp(-_C2 * ν₀ / T_REF))
    boltzmann   = exp(-_C2 * E″ * (1.0/T - 1.0/T_REF))

    return line.intensity * Qr * boltzmann * stim_factor
end

# Approximate molecular masses (amu) for HITRAN molecule IDs 1–11
const _MOL_MASS_AMU = Float64[
    18.010,  # 1  H2O
    44.010,  # 2  CO2
    47.998,  # 3  O3
    44.013,  # 4  N2O
    28.010,  # 5  CO
    16.043,  # 6  CH4
    31.999,  # 7  O2
    34.015,  # 8  HF   (placeholder)
    64.065,  # 9  SO2
    36.461,  # 10 HCl  (placeholder)
    17.031,  # 11 NH3
]

function _molecular_mass_amu(mol_id::Int)
    (mol_id < 1 || mol_id > length(_MOL_MASS_AMU)) && return 30.0
    return _MOL_MASS_AMU[mol_id]
end

"""
    pressure_shift(line, p_atm) -> Float64

Apply HITRAN pressure shift to line center wavenumber:
    ν_shifted = ν₀ + δ × p
"""
pressure_shift(line::HITRANLine, p_atm::Float64) =
    line.wavenumber + Float64(line.pressure_shift) * p_atm
