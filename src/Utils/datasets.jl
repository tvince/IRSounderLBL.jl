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
    ManualDataset

Data this package cannot fetch for you — the source is behind an account, or the
selection is scene-specific. [`data_status`](@ref) reports these so a missing
prerequisite shows up as an instruction rather than as a confusing failure later.
`probe` is a path (file or directory) whose presence means "you have it".
"""
struct ManualDataset
    name::String
    probe::String
    url::String
    needs_account::Bool
    used_for::String
end

"""
Account-gated or scene-specific data, reported by [`data_status`](@ref) but never
downloaded automatically.
"""
const MANUAL_DATASETS = ManualDataset[
    ManualDataset("CO₂ line-mixing (HITRAN2020)",
                  joinpath("Line-mixing_HITRAN2020", "data_new"),
                  "https://hitran.org/supplementary/ → Line-Mixing → https://hitran.org/files/LM/",
                  true,
                  "VPYLineMixing / VPWLineMixing. Take the HITRAN2020 package, not 2016. " *
                  "~430 MB. See data/Line-mixing_HITRAN2020/SOURCE.md"),
    ManualDataset("IASI L1C granules",
                  "iasi_l1c",
                  "https://www.class.noaa.gov/",
                  true,
                  "read_iasi_l1c — real-radiance retrievals. Scene-specific: pick your own granules."),
    ManualDataset("EUMETSAT IASI noise covariance (NCM)",
                  "ncm",
                  "https://data.eumetsat.int/",
                  true,
                  "read_iasi_ncm — host instrument noise covariance as retrieval Se."),
]

"""
Downloadable datasets, keyed by group. Pass a key to [`download_data`](@ref).

`:cia` is fixed files with pinned checksums. `:linelists` is a HITRAN API query —
it needs `HITRAN_API_KEY` and is handled by [`download_linelists`](@ref), so it has
no entry here.
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

"""
    find_data_path(relpath) -> Union{String,Nothing}

Like [`find_data_file`](@ref) but matches a file *or* a directory — used for the
bulk data drops in [`MANUAL_DATASETS`](@ref), which are directories.
"""
function find_data_path(relpath::AbstractString)
    for root in data_search_path()
        p = joinpath(root, relpath)
        ispath(p) && return p
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

`datasets` selects what to fetch:

- `:cia` — HITRAN collision-induced-absorption tables. No key needed. **Default.**
- `:linelists` — HITRAN line lists via the LBL API. Needs `ENV["HITRAN_API_KEY"]`;
  extra keyword arguments (`ν_min`, `ν_max`, `species`) are forwarded to
  [`download_linelists`](@ref). Not included by default, since it needs that key.

Run [`data_status`](@ref) for a checklist of what is present, what is missing, and
what has to be downloaded by hand.

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
                       verbose::Bool = true,
                       linelist_kwargs...)
    keys_ = datasets isa Symbol ? [datasets] : collect(datasets)
    for k in keys_
        (haskey(DATASETS, k) || k === :linelists) ||
            error("Unknown dataset :$k. Available: " *
                  join(sort(vcat(string.(collect(keys(DATASETS))), "linelists")), ", "))
    end

    mkpath(dir)
    nfetched = 0

    # Line lists are an API query, not fixed files — separate path, needs a key.
    if :linelists in keys_
        download_linelists(; dir, force, verbose, linelist_kwargs...)
        keys_ = filter(!=(:linelists), keys_)
    end

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

Print a setup checklist: what data is present, what is missing, and the exact next
step for each gap. Covers the bundled tables, the automatic downloads, and the
account-gated sources this package cannot fetch for you ([`MANUAL_DATASETS`](@ref)).

This is the intended starting point on a fresh install — `using IRSounderLBL;
data_status()` should answer "what do I still need?" without reading any docs.
"""
function data_status(io::IO = stdout)
    mark(ok) = ok ? "  ✓ " : "  ✗ "
    todo = String[]

    println(io, "IRSounderLBL data status")
    println(io, "="^62)

    # ── Tier 0: bundled ──────────────────────────────────────────────────────
    println(io, "\nBUNDLED (ships with the package, no setup)")
    for (label, rel) in ("AFGL reference atmospheres" => "afgl_us_standard_50lev.csv",
                         "MT-CKD H₂O continuum"       => joinpath("mt_ckd_h2o", "mt_ckd43_h2o_coeffs.csv"),
                         "MT-CKD CO₂ continuum"       => joinpath("mt_ckd_co2", "mt_ckd_co2_coeffs.csv"))
        found = find_data_file(rel)
        println(io, mark(!isnothing(found)), label)
        isnothing(found) &&
            push!(todo, "(bundled file missing: $rel — reinstall the package)")
    end

    # ── Tier 1: automatic downloads ──────────────────────────────────────────
    println(io, "\nAUTOMATIC DOWNLOAD  (download_data())")
    for k in sort(collect(keys(DATASETS)))
        for f in DATASETS[k]
            path = find_data_file(f.relpath)
            println(io, mark(!isnothing(path)), f.relpath,
                    isnothing(path) ? "" : "  → $path")
        end
        if !all(!isnothing(find_data_file(f.relpath)) for f in DATASETS[k])
            push!(todo, "julia> download_data(:$k)")
        end
        for (provider, note) in unique((f.provider, f.note) for f in DATASETS[k])
            println(io, "      $provider — $note")
        end
    end

    # ── Tier 2: line lists (API key) ─────────────────────────────────────────
    println(io, "\nLINE LISTS  (download_data(:linelists) — needs a free HITRAN API key)")
    have_key = hitran_api_key_available()
    println(io, mark(have_key), "HITRAN_API_KEY ",
            have_key ? "is set" : "NOT set — register free at https://hitran.org/register/")
    nfound = 0
    ntotal = 0
    for (sp, isos) in LINELIST_DEFAULT_SPECIES, iso in isos
        ntotal += 1
        rel = _linelist_file(linelist_base(sp), iso)
        isnothing(find_data_file(rel)) || (nfound += 1)
    end
    println(io, mark(nfound == ntotal),
            "default 15 µm set (CO₂ iso 1–4, H₂O iso 1–3, ",
            Int(LINELIST_DEFAULT_ν[1]), "–", Int(LINELIST_DEFAULT_ν[2]), " cm⁻¹): ",
            nfound, "/", ntotal, " files")
    if nfound < ntotal
        have_key || push!(todo, "shell> export HITRAN_API_KEY=<your key>   " *
                              "# free: https://hitran.org/register/")
        push!(todo, "julia> download_data(:linelists)")
    end
    println(io, "      Without a line list the forward model cannot run.")

    # ── Tier 3: manual ───────────────────────────────────────────────────────
    println(io, "\nMANUAL  (account-gated or scene-specific — cannot be fetched for you)")
    for m in MANUAL_DATASETS
        path = find_data_path(m.probe)
        println(io, mark(!isnothing(path)), m.name, isnothing(path) ? "" : "  → $path")
        if isnothing(path)
            println(io, "      get it: ", m.url, m.needs_account ? "   (free account required)" : "")
        end
        println(io, "      ", m.used_for)
    end

    # ── What to do next ──────────────────────────────────────────────────────
    println(io, "\n", "="^62)
    if isempty(todo)
        println(io, "All automatic data present. Optional manual sets are listed above.")
    else
        println(io, "NEXT STEPS:")
        for t in unique(todo)
            println(io, "  ", t)
        end
    end
    println(io, "\nSearch path (first match wins):")
    for (i, root) in enumerate(data_search_path())
        println(io, "  $i. $root", isdir(root) ? "" : "   (does not exist)")
    end
    println(io, "Downloads go to: ", data_download_dir())
    return nothing
end
