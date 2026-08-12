using Downloads
using JSON3
using ProgressMeter

"""
    load_hitran_par(filepath; ν_min=nothing, ν_max=nothing) -> HITRANLinelist

Load spectral lines from a locally stored HITRAN .par file.

Optionally filter to the wavenumber range [ν_min, ν_max] (cm⁻¹).
"""
function load_hitran_par(filepath::AbstractString;
                         ν_min::Union{Float64, Nothing} = nothing,
                         ν_max::Union{Float64, Nothing} = nothing)
    isfile(filepath) || error("File not found: $filepath")
    lines = HITRANLine[]
    open(filepath, "r") do fh
        for record in eachline(fh)
            length(record) < 67 && continue   # skip short/blank records
            try
                line = parse_hitran_par(record)
                if (isnothing(ν_min) || line.wavenumber >= ν_min) &&
                   (isnothing(ν_max) || line.wavenumber <= ν_max)
                    push!(lines, line)
                end
            catch
                # skip malformed records silently
            end
        end
    end
    isempty(lines) && @warn "No lines loaded from $filepath"
    return HITRANLinelist(lines)
end

"""
Global isotopologue IDs for HITRAN API (hitran.org/docs/iso-meta/).
The API requires a single global integer ID, not the (mol_id, iso_id) pair.
"""
const HITRAN_GLOBAL_ISO_ID = Dict{Tuple{Int,Int}, Int}(
    # H2O
    (1,1)=>1,  (1,2)=>2,  (1,3)=>3,  (1,4)=>4,  (1,5)=>5,  (1,6)=>6,
    # CO2
    (2,1)=>7,  (2,2)=>8,  (2,3)=>9,  (2,4)=>10, (2,5)=>11, (2,6)=>12,
    (2,7)=>13, (2,8)=>14, (2,9)=>121,(2,10)=>15,
    # O3
    (3,1)=>16, (3,2)=>17, (3,3)=>18, (3,4)=>19, (3,5)=>20,
    # N2O
    (4,1)=>21, (4,2)=>22, (4,3)=>23, (4,4)=>24, (4,5)=>25,
    # CO
    (5,1)=>26, (5,2)=>27, (5,3)=>28, (5,4)=>29, (5,5)=>30, (5,6)=>31,
    # CH4
    (6,1)=>32, (6,2)=>33, (6,3)=>34, (6,4)=>35,
    # O2
    (7,1)=>36, (7,2)=>37, (7,3)=>38,
    # SO2
    (9,1)=>46, (9,2)=>47,
    # NH3
    (11,1)=>52,(11,2)=>53,
)

"""
    fetch_hitran_api(mol_id, iso_id, ν_min, ν_max;
                     outfile=nothing, api_key=nothing) -> HITRANLinelist

Download line data from the HITRAN API (hitran.org) and return a
`HITRANLinelist`.  Requires an API key from hitran.org/profile.

The response is the HITRAN .par ASCII format, written to `outfile` if provided.

# Arguments
- `mol_id`:  HITRAN molecule number
- `iso_id`:  isotopologue number (1 = most abundant)
- `ν_min`:   minimum wavenumber (cm⁻¹)
- `ν_max`:   maximum wavenumber (cm⁻¹)
- `outfile`: optional path to cache the downloaded .par file
- `api_key`: HITRAN API key (also read from ENV["HITRAN_API_KEY"])
"""
function fetch_hitran_api(mol_id::Int, iso_id::Int,
                          ν_min::Float64, ν_max::Float64;
                          outfile::Union{String, Nothing} = nothing,
                          api_key::Union{String, Nothing} = nothing)
    key = something(api_key, get(ENV, "HITRAN_API_KEY", nothing))
    isnothing(key) && error("HITRAN API key required. Set ENV[\"HITRAN_API_KEY\"] or pass api_key=.")

    global_id = get(HITRAN_GLOBAL_ISO_ID, (mol_id, iso_id), nothing)
    isnothing(global_id) && error("Unknown (mol_id=$mol_id, iso_id=$iso_id). Check HITRAN_GLOBAL_ISO_ID table.")

    base_url = "https://hitran.org/lbl/api"
    query    = "?iso_ids_list=$global_id&numin=$ν_min&numax=$ν_max"
    url      = base_url * query * "&api_key=$key"
    # Same URL with the key elided, for anything that gets logged or thrown.
    safe_url = base_url * query * "&api_key=<redacted>"

    mktempdir() do tmpdir
        tmpfile = joinpath(tmpdir, "hitran.par")
        @info "Downloading HITRAN data for mol_id=$mol_id, iso_id=$iso_id (global_id=$global_id), $(ν_min)–$(ν_max) cm⁻¹"
        resp = open(tmpfile, "w") do io
            Downloads.request(url;
                output  = io,
                headers = ["Accept" => "text/plain"],
                throw   = false)
        end
        # resp is either a Response or a RequestError (on truncation)
        http_status = resp isa Downloads.Response ? resp.status : resp.response.status
        if http_status ∉ (200, 206)
            error("HITRAN API returned HTTP $http_status\nURL: $safe_url")
        end
        # Truncated transfers (e.g. missing last few bytes) are acceptable
        if resp isa Downloads.RequestError
            @warn "Download truncated ($(resp.message)); partial data will be used"
        end

        if !isnothing(outfile)
            mkpath(dirname(outfile))
            cp(tmpfile, outfile; force=true)
            @info "Saved to $outfile"
        end

        return load_hitran_par(tmpfile; ν_min=ν_min, ν_max=ν_max)
    end
end

"""
    fetch_hitran_api(species::GasSpecies, ν_min, ν_max; kwargs...) -> HITRANLinelist

Convenience wrapper accepting a `GasSpecies` instead of a numeric mol_id.
Downloads the primary isotopologue (iso_id=1) unless overridden via `iso_id` kwarg.
"""
function fetch_hitran_api(species::GasSpecies, ν_min::Float64, ν_max::Float64;
                          iso_id::Int = 1, kwargs...)
    mol_id = HITRAN_MOLECULE_ID[species]
    return fetch_hitran_api(mol_id, iso_id, ν_min, ν_max; kwargs...)
end
