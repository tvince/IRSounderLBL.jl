"""
    AtmosphericProfile

Atmospheric state defined on a set of pressure levels.

# Fields
- `pressure`:    pressure levels (hPa), length N
- `temperature`: temperature at each level (K), length N
- `altitude`:    altitude at each level (km), length N
- `vmr`:         volume mixing ratios; Dict{GasSpecies, Vector{Float64}}, each length N
"""
struct AtmosphericProfile
    pressure::Vector{Float64}            # hPa
    temperature::Vector{Float64}         # K
    altitude::Vector{Float64}            # km
    vmr::Dict{GasSpecies, Vector{Float64}}
end

"""
    n_levels(prof)

Return the number of pressure levels in the profile.
"""
n_levels(prof::AtmosphericProfile) = length(prof.pressure)

"""
    species(prof)

Return the gas species present in the profile.
"""
species(prof::AtmosphericProfile) = keys(prof.vmr)

function Base.show(io::IO, prof::AtmosphericProfile)
    nl = n_levels(prof)
    sp = join([SPECIES_NAME[s] for s in keys(prof.vmr)], ", ")
    print(io, "AtmosphericProfile($nl levels, $(prof.pressure[end])–$(prof.pressure[1]) hPa, species: $sp)")
end
