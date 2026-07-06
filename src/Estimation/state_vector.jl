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

param_species(p::VMRParameterization) = p.species
_normalize_param(s::GasSpecies)              = FullProfile(s)
_normalize_param(p::VMRParameterization)     = p

# Basis (nothing = identity ⇒ full-profile fast path) and param count per gas.
_param_basis(::FullProfile, n::Int) = nothing
_param_basis(::ColumnScale, n::Int) = ones(Float64, n, 1)
_param_m(::FullProfile, n::Int)     = n
_param_m(::ColumnScale, n::Int)     = 1

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
