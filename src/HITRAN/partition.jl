# Partition functions from TIPS-2024 v1.2 (Gamache et al. 2025, JQSRT 345, 109568).
# Q(T) tables are embedded in tips2024_data.jl; linear interpolation at 1 K resolution.

include("tips2024_data.jl")

const T_REF = 296.0

# Linear interpolation into a 1K-step TIPS table.
# The table covers TIPS_T_MIN:TIPS_T_MAX K; clamps silently at boundaries.
function _tips_lookup(vec::Vector{Float64}, T::Float64)
    i_lo = floor(Int, T) - TIPS_T_MIN + 1
    i_hi = i_lo + 1
    n    = length(vec)
    i_lo = clamp(i_lo, 1, n)
    i_hi = clamp(i_hi, 1, n)
    frac = T - floor(T)
    return vec[i_lo] * (1.0 - frac) + vec[i_hi] * frac
end

"""
    partition_function(mol_id, iso_id, T) -> Float64

Total internal partition sum Q(T) from TIPS-2024 tables.
Falls back to power-law Q ∝ T for unknown isotopologues.
"""
function partition_function(mol_id::Int, iso_id::Int, T::Float64)
    vec = get(TIPS2024_DATA, (mol_id, iso_id), nothing)
    isnothing(vec) && return (T / T_REF)
    return _tips_lookup(vec, T)
end

"""
    Q_ratio(mol_id, iso_id, T) -> Float64

Return Q(T_ref) / Q(T), the partition function ratio used in the HITRAN
temperature-scaled line intensity:

    S(T) = S(T_ref) × [Q(T_ref)/Q(T)] × exp[-C2·E″·(1/T − 1/T_ref)]
                     × [1 − exp(-C2·ν₀/T)] / [1 − exp(-C2·ν₀/T_ref)]
"""
function Q_ratio(mol_id::Int, iso_id::Int, T::Float64)
    vec = get(TIPS2024_DATA, (mol_id, iso_id), nothing)
    isnothing(vec) && return (T_REF / T)
    Q_T   = _tips_lookup(vec, T)
    Q_ref = _tips_lookup(vec, T_REF)
    return Q_ref / Q_T
end
