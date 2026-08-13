"""
First-run data bootstrap for HITRAN line lists.

`download_data(:cia)` (see `Utils/datasets.jl`) fetches fixed files with pinned
checksums. Line lists are different in three ways that shape the API here:

1. they need a **free HITRAN API key** (`ENV["HITRAN_API_KEY"]`);
2. there is **no checksum to pin** — the LBL API serves whatever the current
   HITRAN release holds for a `(species, isotopologue, ν-range)` query, and
   HITRAN revises records. Since a retrieval's results depend on the line list
   version, each download writes a provenance sidecar recording *what was pulled
   and when* instead of pretending to verify a hash;
3. they are a **query**, not a file, so the range and species are parameters.

The default is the 15 µm working set — CO₂ isotopologues 1–4 and H₂O 1–3 over
620–825 cm⁻¹ — which covers the ν₂ band the package is validated against, with the
±25 cm⁻¹ line-wing margin the forward model needs.
"""

using Dates

# Default query: the validated 15 µm CO₂ sounding window (645–800) plus the
# ±25 cm⁻¹ cutoff margin the LBL sum needs at the edges.
const LINELIST_DEFAULT_ν = (620.0, 825.0)

"""
Species and isotopologues fetched by `download_data(:linelists)` by default.
CO₂ iso 4 (627) matters more than its ~0.04% abundance suggests: omitting it
leaves a spurious −8 K line-mixing artefact near 665 cm⁻¹.
"""
const LINELIST_DEFAULT_SPECIES = Pair{GasSpecies,UnitRange{Int}}[
    CO2 => 1:4,
    H2O => 1:3,
]

# ASCII species tag for filenames ("CO₂" -> "co2"); SPECIES_NAME is Unicode.
const _SPECIES_TAG = Dict{GasSpecies,String}(
    H2O => "h2o", CO2 => "co2", O3 => "o3", N2O => "n2o", CO => "co",
    CH4 => "ch4", O2 => "o2", SO2 => "so2", NH3 => "nh3",
)

"""
    linelist_base(species; ν_min, ν_max) -> String

Path (relative to a data root) that [`download_data`](@ref)`(:linelists)` writes
for `species`, without the `.par` extension — the `base` argument
[`load_linelist`](@ref) expects. Isotopologue 1 is `"\$(base).par"`, the rest
`"\$(base)_iso\$(n).par"`.
"""
function linelist_base(species::GasSpecies;
                       ν_min::Float64 = LINELIST_DEFAULT_ν[1],
                       ν_max::Float64 = LINELIST_DEFAULT_ν[2])
    tag = get(_SPECIES_TAG, species, lowercase(string(species)))
    return joinpath("linelists", "$(tag)_$(round(Int, ν_min))_$(round(Int, ν_max))")
end

_linelist_file(base::AbstractString, iso::Int) =
    iso == 1 ? "$(base).par" : "$(base)_iso$(iso).par"

"""
    hitran_api_key_available() -> Bool

Whether `ENV["HITRAN_API_KEY"]` is set and non-empty. The key itself is never
logged or returned.
"""
hitran_api_key_available() = !isempty(get(ENV, "HITRAN_API_KEY", ""))

const _NO_KEY_MESSAGE = """
HITRAN line lists need a free API key, which is not set.

  1. register at https://hitran.org/register/ (free)
  2. copy your key from https://hitran.org/profile/
  3. export HITRAN_API_KEY=<your key>     # in your shell, not in the repo

Then re-run `download_data(:linelists)`. Everything else — the CIA tables,
continua and reference atmospheres — needs no key."""

# Fetch one species' isotopologues. Returns a vector of (relpath, nlines).
function _download_linelist_species(dir::AbstractString, sp::GasSpecies, isos,
                                    ν_min::Float64, ν_max::Float64,
                                    force::Bool, verbose::Bool)
    base    = linelist_base(sp; ν_min, ν_max)
    fetched = Tuple{String,Int}[]
    for iso in isos
        rel = _linelist_file(base, iso)
        if !force
            found = find_data_file(rel)
            if !isnothing(found)
                verbose && @info "already present, skipping" file = rel path = found
                continue
            end
        end
        # Checked here, not up front: a user handed a populated data directory
        # has nothing to fetch and should not need a key at all.
        hitran_api_key_available() || error(_NO_KEY_MESSAGE)
        dest = joinpath(dir, rel)
        mkpath(dirname(dest))
        verbose && @info "fetching line list" species = SPECIES_NAME[sp] isotopologue = iso ν = (ν_min, ν_max)
        # fetch_hitran_api writes `outfile` only after a successful transfer.
        ll = fetch_hitran_api(sp, ν_min, ν_max; iso_id = iso, outfile = dest)
        push!(fetched, (rel, length(ll.lines)))
    end
    return fetched
end

# Line lists carry no pinnable checksum (see module docstring), so record what
# the query was and when it ran. This is the reproducibility trail for a
# retrieval: a result is only as pinned as the line list behind it.
function _write_linelist_provenance(dir::AbstractString, entries, ν_min, ν_max)
    isempty(entries) && return nothing
    path = joinpath(dir, "linelists", "PROVENANCE.md")
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, "## Fetched ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)
        println(io, "Source: HITRAN LBL API (https://hitran.org/lbl/), range ",
                ν_min, "–", ν_max, " cm⁻¹.")
        println(io, "HITRAN serves the current release, which is revised over time, so")
        println(io, "these counts identify the snapshot a retrieval was run against.")
        println(io)
        println(io, "| file | lines |")
        println(io, "|---|---|")
        for (rel, n) in entries
            println(io, "| `", rel, "` | ", n, " |")
        end
        println(io)
        println(io, "Cite HITRAN2020 (Gordon et al., JQSRT 277, 107949, 2022). ",
                "HITRAN data is not redistributable — see NOTICE.")
        println(io)
    end
    return path
end

"""
    download_linelists(; dir = data_download_dir(), species = LINELIST_DEFAULT_SPECIES,
                       ν_min, ν_max, force = false, verbose = true) -> String

Fetch HITRAN line lists via the LBL API into `dir`, following the on-disk naming
[`load_linelist`](@ref) expects. Requires `ENV["HITRAN_API_KEY"]` — but only if
something actually needs fetching, so a fully populated data directory works
with no key.

Usually reached through [`download_data`](@ref)`(:linelists)`. Defaults to the
15 µm working set (see module docstring); pass `ν_min`/`ν_max`/`species` to widen.
Files already on [`data_search_path`](@ref) are skipped unless `force=true`.
Each run appends to `linelists/PROVENANCE.md` recording the query and line counts.

# Example
```julia
# full IASI range, adding ozone
download_linelists(; ν_min = 620.0, ν_max = 2785.0,
                     species = [CO2 => 1:4, H2O => 1:3, O3 => 1:4])
```
"""
function download_linelists(; dir::AbstractString = data_download_dir(),
                            species = LINELIST_DEFAULT_SPECIES,
                            ν_min::Float64 = LINELIST_DEFAULT_ν[1],
                            ν_max::Float64 = LINELIST_DEFAULT_ν[2],
                            force::Bool = false,
                            verbose::Bool = true)
    mkpath(dir)
    entries = Tuple{String,Int}[]
    for (sp, isos) in species
        append!(entries, _download_linelist_species(dir, sp, isos, ν_min, ν_max, force, verbose))
    end
    prov = _write_linelist_provenance(dir, entries, ν_min, ν_max)
    if verbose
        @info "line lists complete" fetched = length(entries) dir = dir provenance = prov
    end
    return dir
end

"""
    default_linelists(; ν_min, ν_max, species = LINELIST_DEFAULT_SPECIES)
        -> Dict{GasSpecies, HITRANLinelist}

Load the line lists [`download_data`](@ref)`(:linelists)` installed, ready to hand
straight to [`forward_model`](@ref). Errors naming the missing files (and how to
get them) rather than silently returning a partial set — a quietly absent
isotopologue is a physics error, not a convenience.

# Example
```julia
using IRSounderLBL
download_data(:linelists)                     # once
prof = afgl_us_standard_50lev()
ν, R, BT = forward_model(prof, default_linelists())
```
"""
function default_linelists(; ν_min::Float64 = LINELIST_DEFAULT_ν[1],
                           ν_max::Float64 = LINELIST_DEFAULT_ν[2],
                           species = LINELIST_DEFAULT_SPECIES)
    out     = Dict{GasSpecies,HITRANLinelist}()
    missing = String[]
    for (sp, isos) in species
        base  = linelist_base(sp; ν_min, ν_max)
        paths = String[]
        for iso in isos
            p = find_data_file(_linelist_file(base, iso))
            isnothing(p) ? push!(missing, _linelist_file(base, iso)) : push!(paths, p)
        end
        isempty(paths) && continue
        lines = HITRANLine[]
        for p in paths
            append!(lines, load_linelist(p).lines)
        end
        out[sp] = HITRANLinelist(lines)
    end
    isempty(missing) || error("""
        Missing line list file(s): $(join(missing, ", "))
        Run `download_data(:linelists)` to fetch them (needs HITRAN_API_KEY),
        or `data_status()` to see what is present.""")
    return out
end
