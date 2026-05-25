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

CO₂ continuum (`co2_continuum`): MT-CKD line-coupling continuum (Hartmann; from
  LBLRTM contnm.f90 BLOCK DATA BFCO2). Tabulated coefficient S(ν) read from
  data/mt_ckd_co2/mt_ckd_co2_coeffs.csv. This is the sub-Lorentzian far-wing /
  line-coupling residual the truncated-Voigt LBL omits — physically distinct from
  the CIA below, and the term that dominates the CO₂ ν₂ wings near 15 µm.
  Formula (contnm.f90 FRNCO2; RADFN and RHOAVE conventions match the H₂O path):
    rho_rat = (p/p_ref) × (T_ref/T)             [RHOAVE; p_ref=1013, T_ref=296]
    k_CO₂   = S(ν) × xfac(ν) × (T/T_eff)^tdep(ν) × 1e-20 × n_CO₂ × rho_rat × rad(ν,T)
  where xfac (XFACCO2, 2000–2998 cm⁻¹) and the bandhead temperature exponent
  tdep (2386–2434 cm⁻¹, T_eff=246 K) are identity outside their ranges, so this
  is valid across the full IASI range. All three tables come from contnm.f90.

CO₂/N₂/O₂ CIA (`co2_cia`, `n2_cia`, `o2_cia`): HITRAN CIA tables
  (Karman et al. 2019, doi:10.1016/j.icarus.2019.02.034). Tabulated
  collision-induced cross sections σ(ν, T) [cm⁵/molec²] read from
  data/cia/{CO2-CO2_2024,N2-N2_2021,O2-O2_2024}.cia.
  Scaling:
    k_CO₂ = σ_CO₂-CO₂(ν,T) × n_CO₂ × n_air     (co2_cia; scaled-to-air approx)
    k_N₂  = σ_N₂-N₂(ν,T)   × n_N₂²              (n2_cia; homogeneous)
    k_O₂  = σ_O₂-O₂(ν,T)   × n_O₂²              (o2_cia; homogeneous)
"""

const _KB      = 1.380649e-23   # J/K
const _RADCN2  = 1.4387769      # cm·K  (hc/k)
const _P_REF   = 1013.0         # reference pressure (mbar = hPa)
const _T_REF   = 296.0          # reference temperature (K)
const _N2_DRY_VMR = 0.78084     # N₂ mole fraction of dry air
const _O2_DRY_VMR = 0.20946     # O₂ mole fraction of dry air

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

# ── MT-CKD CO₂ continuum table (Hartmann line-coupling; from contnm.f90) ─────
# Single coefficient S(ν) on a 2 cm⁻¹ grid (−4 … 10000 cm⁻¹). Extracted from
# LBLRTM's BLOCK DATA BFCO2 by scripts/extract_ckd_co2.py.

const _CKD_CO2_CSV = joinpath(@__DIR__, "..", "..", "data", "mt_ckd_co2", "mt_ckd_co2_coeffs.csv")

function _load_ckd_co2_table()
    isfile(_CKD_CO2_CSV) || return nothing
    nu_v   = Float64[]
    Seff_v = Float64[]   # S × xfac: XFACCO2 (2000–2998) folded in on the coarse grid
    tdep_v = Float64[]   # bandhead exponent (nonzero only on 2386–2434)
    open(_CKD_CO2_CSV) do f
        header = readline(f)
        ncol = length(split(header, ','))
        ncol == 4 || error("$_CKD_CO2_CSV has $ncol columns; expected 4 " *
                           "(nu_cm1,S,xfac,tdep_exp). Regenerate with scripts/extract_ckd_co2.py.")
        for line in eachline(f)
            parts = split(line, ',')
            push!(nu_v,   parse(Float64, parts[1]))
            push!(Seff_v, parse(Float64, parts[2]) * parse(Float64, parts[3]))
            push!(tdep_v, parse(Float64, parts[4]))
        end
    end
    itp_S    = linear_interpolation(nu_v, Seff_v, extrapolation_bc=Flat())
    itp_tdep = linear_interpolation(nu_v, tdep_v, extrapolation_bc=Flat())
    return (itp_S, itp_tdep)
end

const _CKD_CO2 = _load_ckd_co2_table()
const _CKD_CO2_WARNED = Ref(false)
const _T_EFF = 246.0   # bandhead temperature-correction reference (contnm.f90 FRNCO2)

# ── HITRAN CIA lookup tables ─────────────────────────────────────────────────

# A HITRAN .cia file is a list of (T, ν-range, σ(ν)) blocks. Different blocks
# can cover different (and disjoint) spectral regions, so we keep them as a
# vector and do per-query lookup: for a given ν, filter to blocks containing
# ν, then linearly interpolate σ in T between the two bracketing applicable
# blocks. `_load_cia_table` returns `nothing` if the data file is missing.
struct _CIABlock
    T::Float64
    νmin::Float64
    νmax::Float64
    itp::Any  # ν → σ linear interpolator (extrapolates to 0 outside ν range)
end

function _load_cia_table(path::AbstractString)
    isfile(path) || return nothing

    blocks = _CIABlock[]
    cur_T  = NaN
    cur_ν  = Float64[]
    cur_σ  = Float64[]

    function flush!()
        isempty(cur_ν) && return
        # HITRAN sometimes stores ν descending; sort ascending for interp.
        if !issorted(cur_ν)
            perm = sortperm(cur_ν)
            cur_ν = cur_ν[perm]
            cur_σ = cur_σ[perm]
        end
        itp = linear_interpolation(cur_ν, cur_σ, extrapolation_bc=0.0)
        push!(blocks, _CIABlock(cur_T, first(cur_ν), last(cur_ν), itp))
        cur_ν = Float64[]; cur_σ = Float64[]
    end

    open(path) do f
        for line in eachline(f)
            stripped = strip(line)
            isempty(stripped) && continue
            # Header lines start with the species name (alphabetic first
            # non-whitespace char); data lines start with a numeric wavenumber.
            if isletter(stripped[1])
                flush!()
                parts = split(stripped)
                # Format: pair νmin νmax N T σmax ...
                cur_T = parse(Float64, parts[5])
            else
                parts = split(stripped)
                push!(cur_ν, parse(Float64, parts[1]))
                # HITRAN CIA tables contain occasional small negative σ values
                # from baseline-subtraction noise; clamp to 0 (nonphysical for
                # absorption).
                push!(cur_σ, max(0.0, parse(Float64, parts[2])))
            end
        end
        flush!()
    end

    return isempty(blocks) ? nothing : blocks
end

const _CIA_DIR     = joinpath(@__DIR__, "..", "..", "data", "cia")
const _CIA_CO2_FILE = joinpath(_CIA_DIR, "CO2-CO2_2024.cia")
const _CIA_N2_FILE  = joinpath(_CIA_DIR, "N2-N2_2021.cia")
const _CIA_O2_FILE  = joinpath(_CIA_DIR, "O2-O2_2024.cia")

const _CIA_CO2 = _load_cia_table(_CIA_CO2_FILE)
const _CIA_N2  = _load_cia_table(_CIA_N2_FILE)
const _CIA_O2  = _load_cia_table(_CIA_O2_FILE)

const _CIA_CO2_WARNED = Ref(false)
const _CIA_N2_WARNED  = Ref(false)
const _CIA_O2_WARNED  = Ref(false)

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

MT-CKD CO₂ line-coupling continuum absorption coefficient [cm⁻¹] (Hartmann
parameters, isotopes 1+2), from LBLRTM's tabulated S(ν):

    k(ν) = S(ν) × xfac(ν) × (T/T_eff)^tdep(ν) × 1e-20 × n_CO₂ × (p/p_ref)(T_ref/T) × rad(ν,T)

where xfac and tdep are the XFACCO2 (2000–2998 cm⁻¹) and bandhead-temperature
(2386–2434 cm⁻¹, T_eff=246 K) corrections, identity elsewhere. This is the
sub-Lorentzian far-wing residual the truncated-Voigt LBL omits; it dominates the
CO₂ ν₂ band wings (≈690–750 cm⁻¹).

Returns zeros (with a one-time warning) if the coefficient table is absent;
run `scripts/extract_ckd_co2.py` to regenerate it.
"""
function co2_continuum(ν_grid::WavenumberGrid,
                       vmr_co2::Float64,
                       p_hPa::Float64,
                       T::Float64)::Vector{Float64}
    if _CKD_CO2 === nothing
        if !_CKD_CO2_WARNED[]
            @warn "MT-CKD CO₂ coefficient table not found at $_CKD_CO2_CSV; co2_continuum returns zeros. " *
                  "Run scripts/extract_ckd_co2.py to generate it."
            _CKD_CO2_WARNED[] = true
        end
        return zeros(Float64, ν_grid.n)
    end

    itp_S, itp_tdep = _CKD_CO2
    rho_rat = (p_hPa / _P_REF) * (_T_REF / T)
    n_co2   = vmr_co2 * p_hPa * 100.0 / (_KB * T) * 1e-6  # molec/cm³
    pref    = 1e-20 * n_co2 * rho_rat
    t_rat   = T / _T_EFF

    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        td   = itp_tdep(ν)                       # 0 except on the 2386–2434 bandhead
        tcor = td == 0.0 ? 1.0 : t_rat^td
        k[i] = itp_S(ν) * tcor * pref * _radfn(ν, T)
    end
    return k
end

"""
    co2_cia(ν_grid, vmr_co2, p_hPa, T) -> Vector{Float64}

CO₂ collision-induced absorption [cm⁻¹] from HITRAN CIA (CO₂–CO₂,
Karman et al. 2019), scaled by `n_CO₂ × n_air` so CO₂–N₂/O₂ collisions are
treated as equivalent to CO₂–CO₂. Distinct from `co2_continuum` (MT-CKD
line-coupling continuum); near-zero at 15 µm.

Returns zeros (with a one-time warning) if `data/cia/CO2-CO2_2024.cia` is
absent. T outside the table's range is clamped; ν outside the table's range
contributes zero.
"""
function co2_cia(ν_grid::WavenumberGrid,
                 vmr_co2::Float64,
                 p_hPa::Float64,
                 T::Float64)::Vector{Float64}
    if _CIA_CO2 === nothing
        if !_CIA_CO2_WARNED[]
            @warn "HITRAN CO₂–CO₂ CIA table not found at $_CIA_CO2_FILE; co2_cia returns zeros. " *
                  "Download CO2-CO2_2024.cia from hitran.org/cia/ to enable."
            _CIA_CO2_WARNED[] = true
        end
        return zeros(Float64, ν_grid.n)
    end

    n_co2 = vmr_co2 * p_hPa * 100.0 / (_KB * T) * 1e-6  # molec/cm³
    n_air = p_hPa            * 100.0 / (_KB * T) * 1e-6
    nn    = n_co2 * n_air

    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        k[i] = _cia_σ(_CIA_CO2, ν, T) * nn
    end
    return k
end

"""
    n2_cia(ν_grid, vmr_n2, p_hPa, T) -> Vector{Float64}

N₂–N₂ collision-induced absorption [cm⁻¹] from HITRAN CIA (homogeneous pair,
scaled by `n_N₂²`). Dominates the 4 µm region around 2400 cm⁻¹ for
atmospheric conditions.
"""
function n2_cia(ν_grid::WavenumberGrid,
                vmr_n2::Float64,
                p_hPa::Float64,
                T::Float64)::Vector{Float64}
    if _CIA_N2 === nothing
        if !_CIA_N2_WARNED[]
            @warn "HITRAN N₂–N₂ CIA table not found at $_CIA_N2_FILE; n2_cia returns zeros. " *
                  "Download N2-N2_2021.cia from hitran.org/cia/ to enable."
            _CIA_N2_WARNED[] = true
        end
        return zeros(Float64, ν_grid.n)
    end
    n_n2 = vmr_n2 * p_hPa * 100.0 / (_KB * T) * 1e-6
    nn   = n_n2 * n_n2
    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        k[i] = _cia_σ(_CIA_N2, ν, T) * nn
    end
    return k
end

"""
    o2_cia(ν_grid, vmr_o2, p_hPa, T) -> Vector{Float64}

O₂–O₂ collision-induced absorption [cm⁻¹] from HITRAN CIA (homogeneous pair,
scaled by `n_O₂²`). Dominates around 1556 cm⁻¹ (O₂ fundamental) — overlaps
the H₂O ν₂ band.
"""
function o2_cia(ν_grid::WavenumberGrid,
                vmr_o2::Float64,
                p_hPa::Float64,
                T::Float64)::Vector{Float64}
    if _CIA_O2 === nothing
        if !_CIA_O2_WARNED[]
            @warn "HITRAN O₂–O₂ CIA table not found at $_CIA_O2_FILE; o2_cia returns zeros. " *
                  "Download O2-O2_2024.cia from hitran.org/cia/ to enable."
            _CIA_O2_WARNED[] = true
        end
        return zeros(Float64, ν_grid.n)
    end
    n_o2 = vmr_o2 * p_hPa * 100.0 / (_KB * T) * 1e-6
    nn   = n_o2 * n_o2
    k = Vector{Float64}(undef, ν_grid.n)
    @inbounds for (i, ν) in enumerate(ν_grid.ν)
        k[i] = _cia_σ(_CIA_O2, ν, T) * nn
    end
    return k
end

# Find blocks containing ν, then linearly interp σ in T between the bracketing
# applicable blocks. Clamps to nearest block T outside the available range.
function _cia_σ(blocks::Vector{_CIABlock}, ν::Float64, T::Float64)::Float64
    T_lo, σ_lo = -Inf, 0.0
    T_hi, σ_hi =  Inf, 0.0
    any_match  = false
    @inbounds for b in blocks
        (ν < b.νmin || ν > b.νmax) && continue
        any_match = true
        σ = b.itp(ν)
        if b.T ≤ T && b.T > T_lo
            T_lo, σ_lo = b.T, σ
        end
        if b.T ≥ T && b.T < T_hi
            T_hi, σ_hi = b.T, σ
        end
    end
    any_match || return 0.0
    if !isfinite(T_lo)
        return σ_hi
    elseif !isfinite(T_hi)
        return σ_lo
    elseif T_lo == T_hi
        return σ_lo
    else
        w = (T - T_lo) / (T_hi - T_lo)
        return (1.0 - w) * σ_lo + w * σ_hi
    end
end
