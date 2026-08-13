using JLD2
using Scratch
using Preferences

"""
Bump whenever the on-disk cache layout or the `HITRANLine` struct changes, so old
caches auto-invalidate instead of deserializing into the wrong layout.
"""
const LINELIST_CACHE_VERSION = 1

"""
    linelist_cache_dir() -> String

Directory where parsed-linelist `.jld2` caches are stored. Resolution order:

1. `ENV["IRSOUNDER_LINELIST_CACHE"]` (e.g. HPC fast scratch), if set & non-empty;
2. the `linelist_cache_dir` Preference (see [`set_linelist_cache_dir!`](@ref));
3. a package-owned `Scratch.jl` space (default) — user-writable, survives package
   upgrades, and is garbage-collected when the package is removed.

The directory is created if it does not exist.
"""
function linelist_cache_dir()
    env = get(ENV, "IRSOUNDER_LINELIST_CACHE", "")
    if !isempty(env)
        mkpath(env)
        return env
    end
    pref = @load_preference("linelist_cache_dir", nothing)
    if !isnothing(pref)
        mkpath(pref)
        return pref
    end
    return @get_scratch!("linelists")   # created+managed by Scratch.jl
end

"""
    set_linelist_cache_dir!(path)
    set_linelist_cache_dir!(nothing)

Persist a custom linelist cache directory via `Preferences.jl` (written to
`LocalPreferences.toml`). Pass `nothing` to clear the override and fall back to the
Scratch default. The `IRSOUNDER_LINELIST_CACHE` env var, if set, still wins over
this. Triggers recompilation on next load, as Preferences changes do.
"""
function set_linelist_cache_dir!(path::Union{AbstractString,Nothing})
    if isnothing(path)
        @delete_preferences!("linelist_cache_dir")
    else
        @set_preferences!("linelist_cache_dir" => String(path))
    end
    return nothing
end

"""
    clear_linelist_cache()

Delete all `.jld2` linelist caches in the current [`linelist_cache_dir`](@ref).
Returns the number of files removed. The source `.par` files are untouched.
"""
function clear_linelist_cache()
    dir = linelist_cache_dir()
    isdir(dir) || return 0
    n = 0
    for f in readdir(dir; join = true)
        if endswith(f, ".jld2")
            rm(f; force = true)
            n += 1
        end
    end
    return n
end

# Cache file path for a given .par, keyed by abspath + size + mtime + version so a
# re-downloaded or edited .par (or a version bump) produces a different key and the
# stale cache is simply ignored.
function _cache_path(parpath::AbstractString)
    st  = stat(parpath)
    key = string(hash((abspath(parpath), st.size, st.mtime, LINELIST_CACHE_VERSION)); base = 16)
    return joinpath(linelist_cache_dir(), "linelist_$(key).jld2")
end

function _save_cache(cpath::AbstractString, lines::Vector{HITRANLine})
    mkpath(dirname(cpath))
    # write to a temp file then rename, so a crash mid-write can't leave a
    # half-written cache that later reads as valid.
    tmp = "$(cpath).tmp.$(getpid())"
    jldopen(tmp, "w") do f
        f["version"] = LINELIST_CACHE_VERSION
        f["lines"]   = lines        # Vector{HITRANLine} is isbits -> one dense blob
    end
    mv(tmp, cpath; force = true)
    return cpath
end

# Returns the cached Vector{HITRANLine}, or `nothing` if the file is missing,
# unreadable, or version-mismatched (caller rebuilds in that case).
function _load_cache(cpath::AbstractString)
    isfile(cpath) || return nothing
    try
        return jldopen(cpath, "r") do f
            (haskey(f, "version") && f["version"] == LINELIST_CACHE_VERSION) ?
                f["lines"]::Vector{HITRANLine} : nothing
        end
    catch err
        @warn "Ignoring unreadable linelist cache; will rebuild" cpath err
        return nothing
    end
end

"""
    load_linelist(parpath; ν_min=nothing, ν_max=nothing, cache=true, rebuild=false)
        -> HITRANLinelist

Load a HITRAN `.par` linelist, transparently caching the parsed result as a
Julia-native `.jld2` blob so subsequent loads skip the slow ASCII parse.

The **whole** file is parsed and cached once; the optional `[ν_min, ν_max]`
(cm⁻¹) window is then applied in memory via [`filter_linelist`](@ref), so one
cache serves every wavenumber window.

- `cache=false` bypasses the cache entirely (plain parse, nothing written).
- `rebuild=true` forces a re-parse and overwrites the cache.

The cache lives in [`linelist_cache_dir`](@ref) and self-invalidates when the
`.par` changes (size/mtime) or `LINELIST_CACHE_VERSION` is bumped. `.par` is the
source of truth; the `.jld2` is a derived, disposable cache.
"""
function load_linelist(parpath::AbstractString;
                       ν_min::Union{Float64,Nothing} = nothing,
                       ν_max::Union{Float64,Nothing} = nothing,
                       cache::Bool = true,
                       rebuild::Bool = false)
    isfile(parpath) || error("File not found: $parpath")

    local ll::HITRANLinelist
    if cache
        cpath  = _cache_path(parpath)
        cached = rebuild ? nothing : _load_cache(cpath)
        if isnothing(cached)
            ll = load_hitran_par(parpath)        # full file, no ν filter
            _save_cache(cpath, ll.lines)
        else
            ll = HITRANLinelist(cached)
        end
    else
        ll = load_hitran_par(parpath)
    end

    if isnothing(ν_min) && isnothing(ν_max)
        return ll
    end
    return filter_linelist(ll, something(ν_min, -Inf), something(ν_max, Inf))
end

"""
    load_linelist(base, isotopologues; ν_min=nothing, ν_max=nothing,
                  cache=true, rebuild=false) -> HITRANLinelist

Multi-isotopologue convenience: load and merge several `.par` files that share a
`base` path, following the on-disk naming convention `"\$(base).par"` for iso 1 and
`"\$(base)_iso\$(iso).par"` for the rest (the layout the validation scripts use).
Each `.par` is cached independently; the merged result is windowed in memory.

If a requested isotopologue's `.par` file is missing an error is raised naming the
absent file(s), so a silently-dropped isotopologue can never masquerade as a
successful load.

# Example
```julia
ll = load_linelist("data/co2_645_2760", 1:3; ν_min=645.0, ν_max=800.0)
```
"""
function load_linelist(base::AbstractString, isotopologues;
                       ν_min::Union{Float64,Nothing} = nothing,
                       ν_max::Union{Float64,Nothing} = nothing,
                       cache::Bool = true,
                       rebuild::Bool = false)
    all_lines = HITRANLine[]
    missing_files = String[]
    for iso in isotopologues
        fpath = iso == 1 ? "$(base).par" : "$(base)_iso$(iso).par"
        if !isfile(fpath)
            push!(missing_files, fpath)
            continue
        end
        ll = load_linelist(fpath; ν_min, ν_max, cache, rebuild)
        append!(all_lines, ll.lines)
    end
    isempty(missing_files) ||
        error("load_linelist: requested isotopologue(s) with no .par file on disk: " *
              join(missing_files, ", ") *
              " (base=$(base), isotopologues=$(isotopologues))")
    isempty(all_lines) &&
        error("No lines found for base=$(base), isotopologues=$(isotopologues) " *
              "(files present but empty in the requested window)")
    return HITRANLinelist(all_lines)
end
