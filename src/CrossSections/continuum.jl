"""
H₂O and CO₂ continuum absorption.

H₂O: MT-CKD 4.3 continuum (Mlawer et al. 2012, doi:10.1098/rsta.2011.0295).
  Tabulated self and foreign continuum reference coefficients at T=296 K, p=1013 mbar
  are read from data/mt_ckd_h2o/mt_ckd43_h2o_coeffs.csv (pre-processed from the
  AER netCDF file absco-ref_wv-mt-ckd.nc, MT_CKD_H2O v4.3).

  Formula (from mt_ckd_h2o_module.f90):
    rho_rat  = (p/p_ref) * (T_ref/T)
    rad(ν,T) = ν × tanh(RADCN2×ν / 2T)          [cm⁻¹]
    k_self   = Cs(ν) × (T_ref/T)^texp × vmr × rho_rat × rad × n_H₂O
    k_for    = Cf(ν)              × (1−vmr) × rho_rat × rad × n_H₂O
    k_cont   = k_self + k_for                      [cm⁻¹]

CO₂: simplified CIA fit (Hartmann et al. 2002).
"""

const _KB      = 1.380649e-23   # J/K
const _RADCN2  = 1.4387769      # cm·K  (hc/k)
const _P_REF   = 1013.0         # reference pressure (mbar = hPa)
const _T_REF   = 296.0          # reference temperature (K)

# ── MT-CKD 4.3 lookup table (loaded once at module init) ─────────────────────

const _CKD_CSV = joinpath(@__DIR__, "..", "..", "data", "mt_ckd_h2o", "mt_ckd43_h2o_coeffs.csv")

function _load_ckd_table()
    nu_v   = Float64[]
    Cs_v   = Float64[]
    Cf_v   = Float64[]
    texp_v = Float64[]
    open(_CKD_CSV) do f
        readline(f)  # skip header
        for line in eachline(f)
            parts = split(line, ',')
            push!(nu_v,   parse(Float64, parts[1]))
            push!(Cs_v,   parse(Float64, parts[2]))
            push!(Cf_v,   parse(Float64, parts[3]))
            push!(texp_v, parse(Float64, parts[4]))
        end
    end
    itp_Cs   = linear_interpolation(nu_v, Cs_v,   extrapolation_bc=Flat())
    itp_Cf   = linear_interpolation(nu_v, Cf_v,   extrapolation_bc=Flat())
    itp_texp = linear_interpolation(nu_v, texp_v, extrapolation_bc=Flat())
    return (itp_Cs, itp_Cf, itp_texp)
end

const _CKD = _load_ckd_table()

# ── Radiation field term ──────────────────────────────────────────────────────

@inline function _radfn(ν::Float64, T::Float64)::Float64
    x = _RADCN2 * ν / T
    x ≤ 0.01 && return 0.5 * x * ν
    x ≤ 10.0 && return ν * (1.0 - exp(-x)) / (1.0 + exp(-x))
    return ν
end

# ── Public API ────────────────────────────────────────────────────────────────

"""
    h2o_continuum(ν_grid, vmr_h2o, p_hPa, T) -> Vector{Float64}

MT-CKD 4.3 H₂O self + foreign continuum absorption coefficient [cm⁻¹].
"""
function h2o_continuum(ν_grid::WavenumberGrid,
                       vmr_h2o::Float64,
                       p_hPa::Float64,
                       T::Float64)::Vector{Float64}
    rho_rat = (p_hPa / _P_REF) * (_T_REF / T)
    n_h2o   = vmr_h2o * p_hPa * 100.0 / (_KB * T) * 1e-6  # molec/cm³

    itp_Cs, itp_Cf, itp_texp = _CKD

    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        Cs   = itp_Cs(ν)
        Cf   = itp_Cf(ν)
        texp = itp_texp(ν)
        rad  = _radfn(ν, T)

        k_self    = Cs * (_T_REF / T)^texp * vmr_h2o       * rho_rat * rad * n_h2o
        k_foreign = Cf                     * (1.0 - vmr_h2o) * rho_rat * rad * n_h2o
        k[i] = k_self + k_foreign
    end
    return k
end

"""
    co2_continuum(ν_grid, vmr_co2, p_hPa, T) -> Vector{Float64}

CO₂ collision-induced absorption continuum [cm⁻¹]. Simplified CIA fit.
"""
function co2_continuum(ν_grid::WavenumberGrid,
                       vmr_co2::Float64,
                       p_hPa::Float64,
                       T::Float64)::Vector{Float64}
    n_co2 = vmr_co2 * p_hPa * 100.0 / (_KB * T) * 1e-6  # molec/cm³
    n_air = p_hPa * 100.0 / (_KB * T) * 1e-6

    function co2_cia(ν)
        if 1200.0 ≤ ν ≤ 1500.0
            return 1.0e-47 * exp(-((ν - 1350.0) / 150.0)^2)
        elseif 2300.0 ≤ ν ≤ 2400.0
            return 5.0e-48
        else
            return 0.0
        end
    end

    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        k[i] = co2_cia(ν) * n_co2 * n_air * (296.0 / T)^0.5
    end
    return k
end
