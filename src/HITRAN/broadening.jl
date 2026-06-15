"""
Pressure and temperature broadening of HITRAN spectral lines.

References:
  - HITRAN 2020 database paper (Gordon et al. 2022, JQSRT)
  - Voigt profile width formulae from Schreier (2011)
"""

# C2 = hc/k in cm·K (defined again locally to avoid circular dependency)
const _C2 = 1.4387752  # cm·K

# Species-average self-broadening temperature exponents (n_self) and self pressure
# shifts (δ_self), used when per-line values are unavailable in HITRAN 2020.
# H2O: Birk et al. (2021), mean over 620–2785 cm⁻¹ transitions.
# CO2: HITRAN supplemental mean; n_self ≈ n_air so correction is minor.
# All other species: VMR too small for self-broadening to matter; fall back to n_air.
const _N_SELF = Dict{Int,Float64}(
    1 => 0.95,   # H2O
    2 => 0.75,   # CO2
)
const _DELTA_SELF = Dict{Int,Float64}(
    1 =>  0.002,  # H2O  (cm⁻¹/atm)
    2 => -0.003,  # CO2  (cm⁻¹/atm)
)

"""
    pressure_broadened_width(line, p, T; vmr_self) -> (γ_L, γ_D)

Compute the Lorentzian (pressure) half-width γ_L and the Doppler half-width
γ_D (both HWHM, in cm⁻¹) for a given `HITRANLine` at pressure p (atm)
and temperature T (K).

Air and self terms are scaled with their own temperature exponents:
    γ_L = (T_ref/T)^n_air  × γ_air  × (1 − vmr_self) × p
        + (T_ref/T)^n_self × γ_self × vmr_self        × p

n_self is taken from `_N_SELF` (species-average literature values) when available,
falling back to n_air for unlisted species.
"""
function pressure_broadened_width(line::HITRANLine, p_atm::Float64, T::Float64;
                                   vmr_self::Float64 = 0.0)
    mol = Int(line.mol_id)
    n_air_  = Float64(line.temp_depend)
    n_self_ = get(_N_SELF, mol, n_air_)
    t_ratio = T_REF / T
    γ_L = t_ratio^n_air_  * line.air_broad  * (1.0 - vmr_self) * p_atm +
          t_ratio^n_self_ * line.self_broad * vmr_self          * p_atm

    M = _molecular_mass_amu(mol)
    γ_D = line.wavenumber * 3.58126e-7 * sqrt(T / M)

    return Float64(γ_L), Float64(γ_D)
end

"""
    pressure_broadened_width_deriv(line, p_atm, T; vmr_self) -> (γ_L, γ_D, dγ_L/dT, dγ_D/dT)

`pressure_broadened_width` plus its temperature derivatives (cm⁻¹/K). The air and
self Lorentz terms scale as `(T_ref/T)^n`, so each contributes `−n·term/T`; the
Doppler width scales as `√T`, so `dγ_D/dT = γ_D/(2T)`.
"""
function pressure_broadened_width_deriv(line::HITRANLine, p_atm::Float64, T::Float64;
                                         vmr_self::Float64 = 0.0)
    mol = Int(line.mol_id)
    n_air_  = Float64(line.temp_depend)
    n_self_ = get(_N_SELF, mol, n_air_)
    t_ratio = T_REF / T
    A = t_ratio^n_air_  * line.air_broad  * (1.0 - vmr_self) * p_atm
    B = t_ratio^n_self_ * line.self_broad * vmr_self          * p_atm
    γ_L  = A + B
    dγ_L = -(n_air_ * A + n_self_ * B) / T

    M    = _molecular_mass_amu(mol)
    γ_D  = line.wavenumber * 3.58126e-7 * sqrt(T / M)
    dγ_D = γ_D / (2.0 * T)

    return Float64(γ_L), Float64(γ_D), Float64(dγ_L), Float64(dγ_D)
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

"""
    temperature_scaled_intensity_deriv(line, T) -> (S, dS/dT)

`temperature_scaled_intensity` plus its temperature derivative (cm⁻¹/(molec·cm⁻²)/K).
Logarithmic derivative of the three HITRAN factors:

    dlnS/dT = −Q′(T)/Q(T)  +  C₂E″/T²  −  (C₂ν₀/T²)/(exp(C₂ν₀/T) − 1)

(`Q′` is the TIPS table slope, `partition_derivative`; the stimulated-emission term is
negative — the `1 − e^{−C₂ν₀/T}` factor *decreases* with T). `dS = S·dlnS/dT`.
"""
function temperature_scaled_intensity_deriv(line::HITRANLine, T::Float64)
    ν₀ = line.wavenumber
    E″ = line.lower_energy
    mol, iso = Int(line.mol_id), Int(line.iso_id)
    Qr  = Q_ratio(mol, iso, T)
    QT  = partition_function(mol, iso, T)
    dQT = partition_derivative(mol, iso, T)

    stim_num    = 1.0 - exp(-_C2 * ν₀ / T)
    stim_factor = stim_num / (1.0 - exp(-_C2 * ν₀ / T_REF))
    boltzmann   = exp(-_C2 * E″ * (1.0/T - 1.0/T_REF))
    S = line.intensity * Qr * boltzmann * stim_factor

    a     = _C2 * ν₀
    dlnS  = -dQT / QT + _C2 * E″ / T^2 - (a / T^2) / (exp(a / T) - 1.0)
    return S, S * dlnS
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
    pressure_shift(line, p_atm; vmr_self) -> Float64

Apply pressure shift to line center wavenumber, mixing air and self contributions:
    ν_shifted = ν₀ + (δ_air × (1 − vmr_self) + δ_self × vmr_self) × p
"""
function pressure_shift(line::HITRANLine, p_atm::Float64; vmr_self::Float64 = 0.0)
    δ_self = get(_DELTA_SELF, Int(line.mol_id), 0.0)
    return line.wavenumber +
           (Float64(line.pressure_shift) * (1.0 - vmr_self) + δ_self * vmr_self) * p_atm
end
