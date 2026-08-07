"""
Managed data files — where the package looks for its tabulated inputs, and how to
fetch the ones that cannot be redistributed with the source.

Most tables the package needs are small, static, and ship in `<pkg>/data`. The
HITRAN collision-induced-absorption (CIA) tables are the exception: HITRAN's terms
forbid redistribution, so they are **not** in the repository and must be fetched
once with [`download_data`](@ref).

## Where files are looked up

[`find_data_file`](@ref) walks [`data_search_path`](@ref) in order and returns the
first hit:

1. `ENV["IRSOUNDER_DATA_DIR"]`, or the `data_dir` Preference (see
   [`set_data_dir!`](@ref)) — an explicit override, if either is set;
2. `<pkg>/data` — the in-repo tables (and a developer's own untracked data);
3. a package-owned `Scratch.jl` space — where [`download_data`](@ref) writes.

Lookup happens **lazily, on first use**, not at precompile time, so data fetched
after the package is loaded is picked up without a restart or a recompile.
"""

using Downloads
using Preferences
using SHA: sha256
using Scratch

# ── Registry of downloadable files ────────────────────────────────────────────

"""
    DataFile

One fetchable data file: its path relative to a data root, where to get it, and
the SHA-256 it must hash to. `bytes` is the expected size (for progress display
only); `provider`/`note` carry the licensing provenance surfaced by
[`data_status`](@ref).
"""
struct DataFile
    relpath::String
    url::String
    sha256::String
    bytes::Int
    provider::String
    note::String
end

const _HITRAN_CIA_NOTE =
    "HITRAN terms permit use with citation but not redistribution, so this file " *
    "is fetched rather than bundled. Cite Karman et al. (2019), " *
    "doi:10.1016/j.icarus.2019.02.034, and the current HITRAN edition."

"""
Downloadable datasets, keyed by group. Pass a key to [`download_data`](@ref).
"""
const DATASETS = Dict{Symbol,Vector{DataFile}}(
    :cia => [
        DataFile("cia/CO2-CO2_2024.cia",
                 "https://hitran.org/data/CIA/main/CO2-CO2_2024.cia",
                 "9db0856cb008cb147728e3a945f818c60a02370442d492c38d7f8884c9cfadca",
                 2_115_361, "HITRAN", _HITRAN_CIA_NOTE),
        DataFile("cia/N2-N2_2021.cia",
                 "https://hitran.org/data/CIA/main/N2-N2_2021.cia",
                 "f0525f75c667802e8dbbfd7054a0a7e07863d8a4ffdb449fe80666bc658c849d",
                 5_692_968, "HITRAN", _HITRAN_CIA_NOTE),
        DataFile("cia/O2-O2_2024.cia",
                 "https://hitran.org/data/CIA/main/O2-O2_2024.cia",
                 "0146acfe5c1414b8deaa56be9a63159154479d5804d1615855888a2e1efd4961",
                 4_203_198, "HITRAN", _HITRAN_CIA_NOTE),
    ],
)

# ── Data roots ────────────────────────────────────────────────────────────────

const _PKG_DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "data"))

"""
    data_dir_override() -> Union{String,Nothing}

The explicit data directory, from `ENV["IRSOUNDER_DATA_DIR"]` (wins) or the
`data_dir` Preference, or `nothing` if neither is set.
"""
function data_dir_override()
    env = get(ENV, "IRSOUNDER_DATA_DIR", "")
    isempty(env) || return env
    return @load_preference("data_dir", nothing)
end

"""
    set_data_dir!(path)
    set_data_dir!(nothing)

Persist a data directory via `Preferences.jl` (written to `LocalPreferences.toml`).
It is searched first and is where [`download_data`](@ref) writes. Pass `nothing`
to clear the override and fall back to the `Scratch.jl` default. The
`IRSOUNDER_DATA_DIR` env var, if set, still wins over this.
"""
function set_data_dir!(path::Union{AbstractString,Nothing})
    if isnothing(path)
        @delete_preferences!("data_dir")
    else
        @set_preferences!("data_dir" => String(path))
    end
    return nothing
end

"""
    data_download_dir() -> String

Directory [`download_data`](@ref) writes into: the [`data_dir_override`](@ref) if
set, otherwise a package-owned `Scratch.jl` space (user-writable, survives package
upgrades, garbage-collected when the package is removed). Created if absent.
"""
function data_download_dir()
    ovr = data_dir_override()
    if !isnothing(ovr)
        mkpath(ovr)
        return String(ovr)
    end
    return @get_scratch!("data")
end

"""
    data_search_path() -> Vector{String}

Data roots in lookup order: override (if set), the in-repo `<pkg>/data`, then the
downloaded-data scratch space. See the module docstring.
"""
function data_search_path()
    roots = String[]
    ovr = data_dir_override()
    isnothing(ovr) || push!(roots, String(ovr))
    push!(roots, _PKG_DATA_DIR)
    scratch = @get_scratch!("data")
    scratch in roots || push!(roots, scratch)
    return roots
end

"""
    find_data_file(relpath) -> Union{String,Nothing}

Absolute path of `relpath` (e.g. `"cia/N2-N2_2021.cia"`) in the first
[`data_search_path`](@ref) root that has it, or `nothing` if no root does.
"""
function find_data_file(relpath::AbstractString)
    for root in data_search_path()
        p = joinpath(root, relpath)
        isfile(p) && return p
    end
    return nothing
end

# ── Lazily-loaded tables ──────────────────────────────────────────────────────

const _DATA_LOCK = ReentrantLock()

"""
    DataTable(relpath, loader)

A once-only, thread-safe memo for a table read from a data file that may be
absent. On first [`get_table`](@ref) the file is resolved through
[`find_data_file`](@ref) and passed to `loader`; if no root has it the memo holds
`nothing` and callers degrade (typically to zeros with a one-time warning).

Resolution is deferred to first use rather than done at precompile time, so a file
fetched by [`download_data`](@ref) after load is picked up once
[`reset_data_tables!`](@ref) clears the memos.
"""
mutable struct DataTable
    const relpath::String
    const loader::Function
    tried::Bool
    value::Any
    warned::Bool
end

const _DATA_TABLES = DataTable[]

function DataTable(relpath::AbstractString, loader::Function)
    t = DataTable(String(relpath), loader, false, nothing, false)
    push!(_DATA_TABLES, t)
    return t
end

"""
    get_table(t::DataTable) -> Any

The memoized table, or `nothing` if its file is not on the search path.
"""
function get_table(t::DataTable)
    t.tried && return t.value
    lock(_DATA_LOCK) do
        if !t.tried
            path = find_data_file(t.relpath)
            t.value = isnothing(path) ? nothing : t.loader(path)
            t.tried = true
        end
    end
    return t.value
end

"""
    reset_data_tables!()

Drop every memoized [`DataTable`](@ref) so the next use re-resolves its file.
Called automatically by [`download_data`](@ref); call it by hand after moving data
around or changing [`set_data_dir!`](@ref) within a session.
"""
function reset_data_tables!()
    lock(_DATA_LOCK) do
        for t in _DATA_TABLES
            t.tried = false
            t.value = nothing
            t.warned = false
        end
    end
    return nothing
end

# One-time warning for a table whose file is missing. Returns nothing.
function warn_missing_data(t::DataTable, what::AbstractString)
    t.warned && return nothing
    lock(_DATA_LOCK) do
        t.warned && return
        t.warned = true
        @warn """
        $what data file not found: $(t.relpath) — returning zeros.
        Fetch it with `IRSounderLBL.download_data()`.
        Searched: $(join(data_search_path(), ", "))"""
    end
    return nothing
end

# ── Download ──────────────────────────────────────────────────────────────────

function _sha256_file(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

"""
    download_data(datasets = keys(DATASETS); dir = data_download_dir(),
                  force = false, verbose = true) -> String

Fetch the data files the package cannot redistribute, into `dir` (see
[`data_download_dir`](@ref)). Returns `dir`.

Each file is downloaded to a temporary name, verified against its pinned SHA-256,
and only then moved into place, so an interrupted or corrupted download can never
masquerade as a good file. Files already present anywhere on
[`data_search_path`](@ref) are skipped unless `force=true`.

`datasets` selects groups from [`DATASETS`](@ref) (currently `:cia`, the HITRAN
collision-induced-absorption tables).

!!! note "Third-party terms"
    Downloaded data carries its provider's license, not this package's. The
    HITRAN CIA tables may be used with citation but not redistributed — see
    [`data_status`](@ref) and the repository `NOTICE`.

# Example
```julia
using IRSounderLBL
download_data()          # fetch everything missing
data_status()            # show what is present and where
```
"""
function download_data(datasets = keys(DATASETS);
                       dir::AbstractString = data_download_dir(),
                       force::Bool = false,
                       verbose::Bool = true)
    keys_ = datasets isa Symbol ? [datasets] : collect(datasets)
    for k in keys_
        haskey(DATASETS, k) ||
            error("Unknown dataset :$k. Available: " *
                  join(sort(string.(collect(keys(DATASETS)))), ", "))
    end

    mkpath(dir)
    nfetched = 0
    for k in keys_, f in DATASETS[k]
        dest = joinpath(dir, f.relpath)
        if !force
            found = find_data_file(f.relpath)
            if !isnothing(found)
                verbose && @info "already present, skipping" file = f.relpath path = found
                continue
            end
        end
        mkpath(dirname(dest))
        verbose && @info "downloading" file = f.relpath MB = round(f.bytes / 1e6; digits = 1) url = f.url
        tmp = "$(dest).tmp.$(getpid())"
        try
            Downloads.download(f.url, tmp)
            got = _sha256_file(tmp)
            got == f.sha256 || error("""
                Checksum mismatch for $(f.relpath) from $(f.url)
                  expected sha256 $(f.sha256)
                  got      sha256 $got
                The remote file changed or the download was corrupted; not installing it.""")
            mv(tmp, dest; force = true)
            nfetched += 1
        finally
            isfile(tmp) && rm(tmp; force = true)
        end
    end

    # A custom `dir` off the search path would download files the loaders can
    # never find — say so rather than let it look like a successful install.
    if !(abspath(dir) in abspath.(data_search_path()))
        @warn """
        Downloaded to a directory that is not on the data search path, so these files
        will not be picked up. Set it with `set_data_dir!("$dir")` (or the
        IRSOUNDER_DATA_DIR env var) and re-run.""" search_path = data_search_path()
    end

    # Newly-downloaded files must be visible to already-loaded memos.
    reset_data_tables!()
    verbose && @info "download_data complete" fetched = nfetched dir = dir
    return dir
end

"""
    data_available(dataset::Symbol) -> Bool

Whether every file in `dataset` is present on [`data_search_path`](@ref).
"""
function data_available(dataset::Symbol)
    haskey(DATASETS, dataset) || return false
    return all(!isnothing(find_data_file(f.relpath)) for f in DATASETS[dataset])
end

"""
    data_status(io = stdout)

Print where each managed data file was found (or that it is missing, with the
provider's terms), plus the active search path.
"""
function data_status(io::IO = stdout)
    println(io, "Data search path (first match wins):")
    for (i, root) in enumerate(data_search_path())
        println(io, "  $i. $root", isdir(root) ? "" : "   (does not exist)")
    end
    println(io, "\nDownload directory: ", data_download_dir())
    for k in sort(collect(keys(DATASETS)))
        println(io, "\n[$k]")
        for f in DATASETS[k]
            path = find_data_file(f.relpath)
            if isnothing(path)
                println(io, "  ✗ $(f.relpath)  — MISSING (run download_data())")
            else
                println(io, "  ✓ $(f.relpath)  → $path")
            end
        end
        for (provider, note) in unique((f.provider, f.note) for f in DATASETS[k])
            println(io, "  provider: $provider — $note")
        end
    end
    return nothing
end
