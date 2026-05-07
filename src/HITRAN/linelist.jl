"""
    HITRANLine

Single spectral line from the HITRAN 2020 database.

Field names follow the HITRAN .par fixed-width format (160-character records).

# Fields
- `mol_id`:        HITRAN molecule number (1–49)
- `iso_id`:        isotopologue number (1 = most abundant)
- `wavenumber`:    vacuum wavenumber of line center ν₀ (cm⁻¹)
- `intensity`:     spectral line intensity S at 296 K (cm⁻¹/(molec·cm⁻²))
- `a_coeff`:       Einstein A coefficient (s⁻¹)
- `air_broad`:     air-broadened half-width at 296 K (cm⁻¹/atm)
- `self_broad`:    self-broadened half-width at 296 K (cm⁻¹/atm)
- `lower_energy`:  lower-state energy E″ (cm⁻¹)
- `temp_depend`:   temperature dependence exponent n for air broadening
- `pressure_shift`: pressure shift of line center δ at 296 K (cm⁻¹/atm)
"""
struct HITRANLine
    mol_id::Int8
    iso_id::Int8
    wavenumber::Float64       # cm⁻¹
    intensity::Float64        # cm⁻¹/(molec·cm⁻²) at 296 K
    a_coeff::Float64          # s⁻¹
    air_broad::Float32        # cm⁻¹/atm
    self_broad::Float32       # cm⁻¹/atm
    lower_energy::Float64     # cm⁻¹
    temp_depend::Float32      # dimensionless
    pressure_shift::Float32   # cm⁻¹/atm
end

"""
    HITRANLinelist

Collection of HITRAN lines for one or more molecules over a spectral range.

# Fields
- `lines`:     vector of `HITRANLine`
- `molecules`: set of molecule IDs present
- `ν_min`:     minimum wavenumber (cm⁻¹)
- `ν_max`:     maximum wavenumber (cm⁻¹)
"""
struct HITRANLinelist
    lines::Vector{HITRANLine}
    molecules::Set{Int}
    ν_min::Float64
    ν_max::Float64
end

function HITRANLinelist(lines::Vector{HITRANLine})
    mols = Set(Int(l.mol_id) for l in lines)
    νs   = [l.wavenumber for l in lines]
    return HITRANLinelist(lines, mols, minimum(νs), maximum(νs))
end

Base.length(ll::HITRANLinelist) = length(ll.lines)

"""
    filter_linelist(ll, ν_min, ν_max) -> HITRANLinelist

Return a new `HITRANLinelist` containing only lines with wavenumber in [ν_min, ν_max].
"""
function filter_linelist(ll::HITRANLinelist, ν_min::Float64, ν_max::Float64)
    lines = filter(l -> ν_min <= l.wavenumber <= ν_max, ll.lines)
    isempty(lines) && return HITRANLinelist(lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(lines)
end

"""
    filter_linelist(ll, S_min) -> HITRANLinelist

Return a new `HITRANLinelist` keeping only lines with intensity ≥ `S_min`
(cm⁻¹/(molec·cm⁻²) at 296 K).
"""
function filter_linelist(ll::HITRANLinelist, S_min::Float64)
    lines = filter(l -> l.intensity >= S_min, ll.lines)
    isempty(lines) && return HITRANLinelist(lines, Set{Int}(), ll.ν_min, ll.ν_max)
    return HITRANLinelist(lines)
end
Base.show(io::IO, ll::HITRANLinelist) =
    print(io, "HITRANLinelist($(length(ll)) lines, mol_ids=$(sort(collect(ll.molecules))), $(ll.ν_min)–$(ll.ν_max) cm⁻¹)")

"""
    parse_hitran_par(line::AbstractString) -> HITRANLine

Parse a single 160-character HITRAN .par format record.
"""
function parse_hitran_par(record::AbstractString)
    # HITRAN .par fixed-width columns (1-indexed, Julia slicing)
    mol_id        = parse(Int8,    record[1:2])
    iso_id        = parse(Int8,    record[3:3])
    wavenumber    = parse(Float64, record[4:15])
    intensity     = parse(Float64, record[16:25])
    a_coeff       = parse(Float64, record[26:35])
    air_broad     = parse(Float32, record[36:40])
    self_broad    = parse(Float32, record[41:45])
    lower_energy  = parse(Float64, record[46:55])
    temp_depend   = parse(Float32, record[56:59])
    pressure_shift= parse(Float32, record[60:67])
    return HITRANLine(mol_id, iso_id, wavenumber, intensity, a_coeff,
                      air_broad, self_broad, lower_energy, temp_depend,
                      pressure_shift)
end
