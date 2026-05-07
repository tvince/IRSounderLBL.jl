"""
    GasSpecies

Enumeration of supported atmospheric trace gases, with HITRAN 2020 molecule IDs.
"""
@enum GasSpecies begin
    H2O  = 1
    CO2  = 2
    O3   = 3
    N2O  = 4
    CO   = 5
    CH4  = 6
    O2   = 7
    SO2  = 9
    NH3  = 11
end

"""
    HITRAN_MOLECULE_ID

Mapping from `GasSpecies` to HITRAN integer molecule ID.
"""
const HITRAN_MOLECULE_ID = Dict{GasSpecies, Int}(
    H2O  => 1,
    CO2  => 2,
    O3   => 3,
    N2O  => 4,
    CO   => 5,
    CH4  => 6,
    O2   => 7,
    SO2  => 9,
    NH3  => 11,
)

"""
    SPECIES_NAME

Human-readable names for each gas species.
"""
const SPECIES_NAME = Dict{GasSpecies, String}(
    H2O  => "H₂O",
    CO2  => "CO₂",
    O3   => "O₃",
    N2O  => "N₂O",
    CO   => "CO",
    CH4  => "CH₄",
    O2   => "O₂",
    SO2  => "SO₂",
    NH3  => "NH₃",
)

Base.show(io::IO, s::GasSpecies) = print(io, SPECIES_NAME[s])
