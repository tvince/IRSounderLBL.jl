"""Dump TIPS-2024 Q(T) for H2O and CO2 isotopologues to data/tips2024_qt.csv."""

using RadiativeTransfer

OUT = joinpath(@__DIR__, "..", "data", "tips2024_qt.csv")

# (mol_id, iso_id) pairs to export
SPECIES = [
    (1, 1),  # H2O iso1
    (1, 2),  # H2O iso2
    (1, 3),  # H2O iso3
    (2, 1),  # CO2 iso1
    (2, 2),  # CO2 iso2
    (2, 3),  # CO2 iso3
]

T_range = 150.0:1.0:400.0

open(OUT, "w") do io
    println(io, "T,mol_id,iso_id,Q")
    for (mol, iso) in SPECIES
        for T in T_range
            Q = RadiativeTransfer.partition_function(mol, iso, T)
            println(io, "$T,$mol,$iso,$Q")
        end
    end
end

println("Wrote $OUT")
