"""
State-vector abstraction for Jacobian / retrieval work (roadmap Phase 0).

The retrieval state vector `x` is a flat `Vector{Float64}` assembled from a fixed,
ordered set of blocks:

    [ temperature levels (N) ] [ VMR block per species (N each) ] [ T_sfc ] [ ε ]

Only the blocks selected in the `StateVectorSpec` are present; absent blocks
occupy no indices. Temperature and VMR live on the profile's **pressure levels**
(length `N = n_levels`), matching `AtmosphericProfile`. VMR is stored as `log(VMR)`
by default (`log_vmr=true`) — the standard choice for positive-definite trace gases,
which also linearises the multiplicative response (roadmap §6.5).

`T_sfc` and `ε` are treated as **independent** state elements: when present they are
always passed explicitly to the forward model, so perturbing the surface does not
move `T_levels[1]` and vice versa. When `T_sfc` is *not* in the state, it follows the
forward-model default (tied to the — possibly perturbed — `T_levels[1]`).

Key operations:
- `pack_state`   : `AtmosphericProfile` (+ T_sfc, ε) → `x`
- `unpack_state` : `x` (+ a base profile for the fixed fields) → `(profile, T_sfc, ε)`
- `state_labels` : human-readable name per column of `x` / `K`
"""

const _VMR_FLOOR = 1e-30   # guard for log of a zero/absent VMR when packing

"""
    VMRParameterization

How a retrieved gas maps onto state elements. A gas can be carried at full vertical
resolution (`FullProfile`, one log-VMR per level — the default) or through a reduced
linear basis `B` (`n_levels × M`) that ties levels together:

    log VMR(level j) = log VMR_ref(j) + Σ_m B[j,m]·θ_m      (θ = the M state params)

The reference `VMR_ref` is the shape of the profile passed to `pack_state` /
`unpack_state`, so `θ = 0` reproduces it (`xa` is all-zeros for a reduced block).
Reduced params are **always multiplicative-log** (a scale factor), independent of
`spec.log_vmr`. `ColumnScale` is the `M=1` case: `B = ones(n_levels,1)`, a single
column-amount scale — one Jacobian column (one ILS convolution) instead of `n_levels`.
`PartialColumns` is the general `M`-block case ("bulk layers"), whose blocks can be
placed to match a pilot averaging kernel (see [`dfs_partition`](@ref)).
"""
abstract type VMRParameterization end

"Retrieve `species` at full vertical resolution (one log-VMR per level)."
struct FullProfile <: VMRParameterization
    species::GasSpecies
end

"""
    ColumnScale(species)

Retrieve a single multiplicative column scale for `species`: `VMR = VMR_ref·exp(θ)`
with `θ` the one state element. Basis `B = ones(n_levels,1)`. Its analytic Jacobian
is the profile-weighted sum of the per-level sensitivities, assembled in the hi-res
domain and convolved **once** — not `n_levels` per-level columns.
"""
struct ColumnScale <: VMRParameterization
    species::GasSpecies
end

"""
    PartialColumns(species, basis)
    PartialColumns(species, pressure; n_blocks=, edges_hPa=, taper=:boxcar)

Retrieve `species` as **N partial-column ("bulk layer") scales** — one
multiplicative amount per contiguous vertical block, tying the levels in each block
together. This is the intermediate between a full profile (`N = n_levels`) and a
single `ColumnScale` (`N = 1`); pick `N` to match the profile's real information
content (`≈ tr(A)` for the species).

`basis` is `n_levels × N` and should partition unity (each level's weights sum to 1),
so setting all params equal reproduces a uniform column scale. Build it with
[`partial_column_basis`](@ref) from either a block count, explicit `edges_hPa`, or
`dfs_partition` blocks (edges placed where a pilot averaging kernel accumulates ~1
DOF — the information-matched grid):

    dfs   = diag(pilot.A)[vmr_range(pilot_spec, O3)]   # per-level DOF from a pilot
    B     = partial_column_basis(n_levels, dfs_partition(dfs; target_dfs=1.0))
    spec  = StateVectorSpec(nlev, [T, PartialColumns(O3, B)])

`taper=:boxcar` (default) gives hard slabs (each block column = the sum of its
levels' sensitivities); `:tent` gives C⁰-continuous overlapping weights.
"""
struct PartialColumns <: VMRParameterization
    species::GasSpecies
    basis::Matrix{Float64}
    function PartialColumns(species::GasSpecies, basis::AbstractMatrix)
        B = Matrix{Float64}(basis)
        size(B, 2) >= 1 || error("PartialColumns needs ≥ 1 block")
        new(species, B)
    end
end

PartialColumns(species::GasSpecies, pressure::AbstractVector; taper::Symbol=:boxcar,
               n_blocks::Union{Int,Nothing}=nothing,
               edges_hPa::Union{AbstractVector,Nothing}=nothing) =
    PartialColumns(species,
                   partial_column_basis(pressure; n_blocks=n_blocks,
                                        edges_hPa=edges_hPa, taper=taper))

param_species(p::VMRParameterization) = p.species
_normalize_param(s::GasSpecies)              = FullProfile(s)
_normalize_param(p::VMRParameterization)     = p

# Basis (nothing = identity ⇒ full-profile fast path) and param count per gas.
_param_basis(::FullProfile, n::Int) = nothing
_param_basis(::ColumnScale, n::Int) = ones(Float64, n, 1)
_param_m(::FullProfile, n::Int)     = n
_param_m(::ColumnScale, n::Int)     = 1
function _param_basis(p::PartialColumns, n::Int)
    size(p.basis, 1) == n ||
        error("PartialColumns basis has $(size(p.basis,1)) levels, spec expects $n")
    p.basis
end
_param_m(p::PartialColumns, n::Int) = size(p.basis, 2)

# ── Partial-column basis construction ─────────────────────────────────────────
# Split 1:n into `nb` contiguous, near-equal-count blocks.
function _blocks_equal_count(n::Int, nb::Int)
    (1 <= nb <= n) || error("n_blocks must be in 1:$n; got $nb")
    e = round.(Int, range(0, n; length=nb+1))
    return [(e[m]+1):e[m+1] for m in 1:nb]
end

# Group levels into contiguous blocks by which interval of `edges_hPa` their
# pressure falls in (pressure assumed monotonic in level).
function _blocks_from_edges(pressure::AbstractVector, edges_hPa::AbstractVector)
    n = length(pressure)
    ed = sort(Float64.(collect(edges_hPa)); rev=true)   # descending pressure
    label(pj) = count(>(pj), ed) + 1
    labels = [label(Float64(pressure[j])) for j in 1:n]
    blocks = UnitRange{Int}[]; start = 1
    for j in 2:n
        labels[j] == labels[j-1] || (push!(blocks, start:(j-1)); start = j)
    end
    push!(blocks, start:n)
    length(blocks) == length(unique(labels)) ||
        error("edges_hPa produced non-contiguous blocks — is `pressure` monotonic?")
    return blocks
end

_blocks_to_basis(n::Int, blocks::Vector{UnitRange{Int}}, ::Val{:boxcar}) = begin
    B = zeros(Float64, n, length(blocks))
    for (m, blk) in enumerate(blocks), j in blk
        B[j, m] = 1.0
    end
    B
end

function _blocks_to_basis(n::Int, blocks::Vector{UnitRange{Int}}, ::Val{:tent})
    N = length(blocks)
    B = zeros(Float64, n, N)
    c = [(first(b) + last(b)) / 2 for b in blocks]     # node = block centre (index space)
    for j in 1:n
        if j <= c[1]
            B[j, 1] = 1.0
        elseif j >= c[end]
            B[j, N] = 1.0
        else
            m = findlast(ci -> ci <= j, c)              # left node
            wR = (j - c[m]) / (c[m+1] - c[m])
            B[j, m]   = 1.0 - wR
            B[j, m+1] = wR
        end
    end
    B
end

"""
    partial_column_basis(n_levels, blocks; taper=:boxcar) -> Matrix
    partial_column_basis(pressure; n_blocks=, edges_hPa=, taper=:boxcar) -> Matrix

Build a partition-of-unity `n_levels × N` basis for [`PartialColumns`](@ref). Give
either explicit `blocks` (contiguous `UnitRange`s covering `1:n_levels`, e.g. from
[`dfs_partition`](@ref)), a block count `n_blocks` (near-equal level counts), or
pressure `edges_hPa` (interior boundaries). `taper` is `:boxcar` (hard slabs) or
`:tent` (C⁰ overlap).
"""
function partial_column_basis(n_levels::Int, blocks::Vector{UnitRange{Int}};
                              taper::Symbol=:boxcar)
    taper in (:boxcar, :tent) || error("taper must be :boxcar or :tent; got :$taper")
    vcat((collect(b) for b in blocks)...) == collect(1:n_levels) ||
        error("blocks must be contiguous and cover 1:$n_levels exactly")
    return _blocks_to_basis(n_levels, blocks, Val(taper))
end

function partial_column_basis(pressure::AbstractVector; taper::Symbol=:boxcar,
                              n_blocks::Union{Int,Nothing}=nothing,
                              edges_hPa::Union{AbstractVector,Nothing}=nothing)
    n = length(pressure)
    blocks = if edges_hPa !== nothing
        n_blocks === nothing || error("give exactly one of n_blocks / edges_hPa")
        _blocks_from_edges(pressure, edges_hPa)
    elseif n_blocks !== nothing
        _blocks_equal_count(n, n_blocks)
    else
        error("give exactly one of n_blocks / edges_hPa")
    end
    return partial_column_basis(n, blocks; taper=taper)
end

"""
    dfs_partition(dfs; target_dfs=1.0) -> Vector{UnitRange{Int}}

Partition levels into contiguous blocks each carrying ≈ `target_dfs` degrees of
freedom, walking the per-level DOF vector `dfs` (the averaging-kernel diagonal
`diag(A)` restricted to a species' block — see [`vmr_range`](@ref)). Places a cut
whenever the accumulated DOF reaches `target_dfs`; the remainder forms the last
block. Feed the result to [`partial_column_basis`](@ref) to retrieve the species at
its information-matched vertical resolution.
"""
function dfs_partition(dfs::AbstractVector; target_dfs::Real=1.0)
    target_dfs > 0 || error("target_dfs must be positive")
    n = length(dfs)
    blocks = UnitRange{Int}[]; start = 1; acc = 0.0
    for j in 1:n
        acc += max(0.0, Float64(dfs[j]))
        if acc >= target_dfs && j < n
            push!(blocks, start:j); start = j + 1; acc = 0.0
        end
    end
    push!(blocks, start:n)
    return blocks
end

"""
    VMRBlock(species, basis, m, range)

Layout record for one retrieved gas: its state-index `range` (length `m`) and the
`basis` (`n_levels × m`, or `nothing` for the identity/full-profile fast path).
"""
struct VMRBlock
    species::GasSpecies
    basis::Union{Nothing, Matrix{Float64}}   # nothing ⇒ identity (full profile)
    m::Int                                    # number of state params (n_levels or reduced)
    range::UnitRange{Int}
end

"""
    StateVectorSpec(n_levels, species;
                    include_temperature=true, include_tsfc=true,
                    include_emissivity=true, log_vmr=true)

Describe the layout of a retrieval state vector. `species` is the ordered list of
retrieved gases; each entry is either a bare `GasSpecies` (⇒ `FullProfile`, an
`n_levels`-long log-VMR block) or a `VMRParameterization` such as `ColumnScale`,
which claims fewer elements. The constructor precomputes the index range of every
block; `spec.n` is the total length. See module docstring for block ordering.
"""
struct StateVectorSpec
    n_levels::Int
    species::Vector{GasSpecies}
    include_temperature::Bool
    include_tsfc::Bool
    include_emissivity::Bool
    log_vmr::Bool
    temp_range::UnitRange{Int}                       # 1:0 (empty) if absent
    vmr_ranges::Vector{Pair{GasSpecies, UnitRange{Int}}}   # (species => range), ordered
    vmr_blocks::Vector{VMRBlock}                     # parallel to vmr_ranges, carries basis
    tsfc_index::Int                                  # 0 if absent
    emis_index::Int                                  # 0 if absent
    n::Int                                           # total state length
end

function StateVectorSpec(n_levels::Int, species::AbstractVector;
                         include_temperature::Bool = true,
                         include_tsfc::Bool        = true,
                         include_emissivity::Bool  = true,
                         log_vmr::Bool             = true)
    n_levels > 0 || error("n_levels must be positive")
    idx = 0
    temp_range = 1:0
    if include_temperature
        temp_range = (idx + 1):(idx + n_levels); idx += n_levels
    end
    vmr_ranges = Pair{GasSpecies, UnitRange{Int}}[]
    vmr_blocks = VMRBlock[]
    for entry in species
        p = _normalize_param(entry)
        s = param_species(p)
        m = _param_m(p, n_levels)
        B = _param_basis(p, n_levels)
        r = (idx + 1):(idx + m); idx += m
        push!(vmr_ranges, s => r)
        push!(vmr_blocks, VMRBlock(s, B, m, r))
    end
    tsfc_index = 0
    if include_tsfc
        idx += 1; tsfc_index = idx
    end
    emis_index = 0
    if include_emissivity
        idx += 1; emis_index = idx
    end
    return StateVectorSpec(n_levels, GasSpecies[b.species for b in vmr_blocks],
                           include_temperature, include_tsfc, include_emissivity, log_vmr,
                           temp_range, vmr_ranges, vmr_blocks, tsfc_index, emis_index, idx)
end

Base.length(spec::StateVectorSpec) = spec.n

"""
    vmr_range(spec, species) -> UnitRange{Int}

State-index range of a retrieved gas's block (for slicing `x`, `K`, or `diag(A)` —
e.g. to pull a species' per-level DOF for [`dfs_partition`](@ref)).
"""
function vmr_range(spec::StateVectorSpec, s::GasSpecies)
    for (sp, r) in spec.vmr_ranges
        sp === s && return r
    end
    error("species $(SPECIES_NAME[s]) is not in the state vector")
end

"""
    state_labels(spec) -> Vector{String}

One human-readable label per column of the state vector / Jacobian, in `x` order:
`T[i]`, `logVMR_<sp>[i]` (or `VMR_<sp>[i]`), `T_sfc`, `emissivity`.
"""
function state_labels(spec::StateVectorSpec)::Vector{String}
    labels = String[]
    spec.include_temperature && append!(labels, ["T[$i]" for i in 1:spec.n_levels])
    pre = spec.log_vmr ? "logVMR" : "VMR"
    for b in spec.vmr_blocks
        nm = SPECIES_NAME[b.species]
        if b.basis === nothing
            append!(labels, ["$(pre)_$nm[$i]" for i in 1:spec.n_levels])
        elseif b.m == 1
            push!(labels, "logVMRscale_$nm")
        else
            append!(labels, ["logVMRscale_$nm[$k]" for k in 1:b.m])
        end
    end
    spec.include_tsfc       && push!(labels, "T_sfc")
    spec.include_emissivity && push!(labels, "emissivity")
    return labels
end

"""
    pack_state(spec, prof; T_sfc=nothing, ε_sfc=1.0) -> Vector{Float64}

Assemble the state vector `x` from a profile. `T_sfc` defaults to
`prof.temperature[1]` (the forward-model convention). VMR blocks are stored as
`log(VMR)` when `spec.log_vmr` (clamped at `_VMR_FLOOR` to keep the log finite).
"""
function pack_state(spec::StateVectorSpec, prof::AtmosphericProfile;
                    T_sfc::Union{Float64, Nothing} = nothing,
                    ε_sfc::Float64 = 1.0)::Vector{Float64}
    length(prof.temperature) == spec.n_levels ||
        error("profile has $(length(prof.temperature)) levels, spec expects $(spec.n_levels)")
    x = zeros(Float64, spec.n)
    spec.include_temperature && (x[spec.temp_range] .= prof.temperature)
    for b in spec.vmr_blocks
        haskey(prof.vmr, b.species) || error("profile is missing retrieved species $(b.species)")
        if b.basis === nothing
            v = prof.vmr[b.species]
            x[b.range] .= spec.log_vmr ? log.(max.(v, _VMR_FLOOR)) : v
        else
            x[b.range] .= 0.0    # reduced block: θ=0 ⇒ the packed profile is the reference
        end
    end
    spec.include_tsfc &&
        (x[spec.tsfc_index] = isnothing(T_sfc) ? prof.temperature[1] : T_sfc)
    spec.include_emissivity && (x[spec.emis_index] = ε_sfc)
    return x
end

"""
    unpack_state(spec, x, base_prof) -> (profile, T_sfc, ε_sfc)

Rebuild a forward-model input set from a state vector. Fields that are not part of
the state (pressure, altitude, non-retrieved species, and — depending on `spec` —
temperature) are copied from `base_prof`. When `T_sfc`/`ε` are not in the state they
fall back to the forward-model defaults (`T_sfc → temperature[1]`, `ε → 1.0`).
"""
function unpack_state(spec::StateVectorSpec, x::AbstractVector{<:Real},
                      base_prof::AtmosphericProfile)
    length(x) == spec.n || error("state vector length $(length(x)) ≠ spec.n $(spec.n)")
    T = spec.include_temperature ? collect(Float64, x[spec.temp_range]) :
                                   copy(base_prof.temperature)
    vmr = Dict{GasSpecies, Vector{Float64}}()
    for (s, v) in base_prof.vmr
        vmr[s] = copy(v)
    end
    for b in spec.vmr_blocks
        xv = collect(Float64, x[b.range])
        if b.basis === nothing
            vmr[b.species] = spec.log_vmr ? exp.(xv) : xv
        else
            # Reduced block: multiplicative-log about the reference shape in base_prof.
            ref = base_prof.vmr[b.species]
            vmr[b.species] = ref .* exp.(b.basis * xv)
        end
    end
    prof = AtmosphericProfile(copy(base_prof.pressure), T, copy(base_prof.altitude), vmr)
    T_sfc = spec.include_tsfc ? Float64(x[spec.tsfc_index]) : T[1]
    ε_sfc = spec.include_emissivity ? Float64(x[spec.emis_index]) : 1.0
    return prof, T_sfc, ε_sfc
end

function Base.show(io::IO, spec::StateVectorSpec)
    blocks = String[]
    spec.include_temperature && push!(blocks, "T($(spec.n_levels))")
    for b in spec.vmr_blocks
        nm = SPECIES_NAME[b.species]
        if b.basis === nothing
            push!(blocks, "$(spec.log_vmr ? "logVMR" : "VMR")_$nm($(spec.n_levels))")
        else
            push!(blocks, "col_$nm($(b.m))")
        end
    end
    spec.include_tsfc       && push!(blocks, "T_sfc")
    spec.include_emissivity && push!(blocks, "ε")
    print(io, "StateVectorSpec(n=$(spec.n): ", join(blocks, " + "), ")")
end
