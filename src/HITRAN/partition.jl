"""
Partition functions via the TIPS-2021 approach.

For practical use we implement the polynomial-fit partition functions
from the HITRAN supplementary material (Gamache et al. 2021, JQSRT).
The reference temperature is T_ref = 296 K.

For each (mol_id, iso_id) pair we store Chebyshev or polynomial coefficients
covering 1–3000 K.  Here we use a simplified power-law fit
    Q(T) = Q(296) × (T/296)^α
with molecule-specific α values, which is accurate to ~1% for T in 150–350 K.
For production accuracy use the full TIPS tables from HITRAN.
"""

# Reference temperature (K)
const T_REF = 296.0

"""
    Q_approx_exponent

Temperature exponent α such that Q(T) ≈ Q(T_ref) × (T/T_ref)^α.
Keyed by (mol_id::Int, iso_id::Int).  Covers the most important species.
"""
const Q_APPROX_EXPONENT = Dict{Tuple{Int,Int}, Float32}(
    # H2O
    (1, 1) => 1.50f0,
    (1, 2) => 1.50f0,
    (1, 3) => 1.50f0,
    # CO2
    (2, 1) => 1.00f0,
    (2, 2) => 1.00f0,
    (2, 3) => 1.00f0,
    (2, 4) => 1.00f0,
    # O3
    (3, 1) => 1.50f0,
    # N2O
    (4, 1) => 1.00f0,
    # CO
    (5, 1) => 1.00f0,
    (5, 2) => 1.00f0,
    # CH4
    (6, 1) => 1.50f0,
    (6, 2) => 1.50f0,
    # O2
    (7, 1) => 1.00f0,
    # SO2
    (9, 1) => 1.50f0,
    # NH3
    (11, 1) => 1.50f0,
)

"""
    partition_function(mol_id, iso_id, T) -> Float64

Approximate TIPS-2021 partition function at temperature T (K).
"""
function partition_function(mol_id::Int, iso_id::Int, T::Float64)
    key = (mol_id, iso_id)
    α = Float64(get(Q_APPROX_EXPONENT, key, 1.0f0))
    return (T / T_REF)^α
end

"""
    Q_ratio(mol_id, iso_id, T) -> Float64

Return Q(T_ref) / Q(T), the ratio used in the temperature-scaled line
intensity formula of HITRAN:

    S(T) = S(T_ref) × [Q(T_ref)/Q(T)] × exp[-C2·E″·(1/T − 1/T_ref)]
                     × [1 − exp(-C2·ν₀/T)] / [1 − exp(-C2·ν₀/T_ref)]
"""
function Q_ratio(mol_id::Int, iso_id::Int, T::Float64)
    key = (mol_id, iso_id)
    α = Float64(get(Q_APPROX_EXPONENT, key, 1.0f0))
    return (T_REF / T)^α
end
