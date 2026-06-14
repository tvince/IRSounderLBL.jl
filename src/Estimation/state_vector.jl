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
    StateVectorSpec(n_levels, species;
                    include_temperature=true, include_tsfc=true,
                    include_emissivity=true, log_vmr=true)

Describe the layout of a retrieval state vector. `species` is the ordered list of
retrieved gases (each contributes an `n_levels`-long VMR block, in this order).
The constructor precomputes the index range of every block; `spec.n` is the total
length. See module docstring for the block ordering and conventions.
"""
struct StateVectorSpec
    n_levels::Int
    species::Vector{GasSpecies}
    include_temperature::Bool
    include_tsfc::Bool
    include_emissivity::Bool
    log_vmr::Bool
    temp_range::UnitRange{Int}                       # 1:0 (empty) if absent
    vmr_ranges::Vector{Pair{GasSpecies, UnitRange{Int}}}   # ordered as `species`
    tsfc_index::Int                                  # 0 if absent
    emis_index::Int                                  # 0 if absent
    n::Int                                           # total state length
end

function StateVectorSpec(n_levels::Int, species::AbstractVector{GasSpecies};
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
    for s in species
        r = (idx + 1):(idx + n_levels); idx += n_levels
        push!(vmr_ranges, s => r)
    end
    tsfc_index = 0
    if include_tsfc
        idx += 1; tsfc_index = idx
    end
    emis_index = 0
    if include_emissivity
        idx += 1; emis_index = idx
    end
    return StateVectorSpec(n_levels, collect(species), include_temperature,
                           include_tsfc, include_emissivity, log_vmr,
                           temp_range, vmr_ranges, tsfc_index, emis_index, idx)
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
    for (s, _) in spec.vmr_ranges
        append!(labels, ["$(pre)_$(SPECIES_NAME[s])[$i]" for i in 1:spec.n_levels])
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
    for (s, r) in spec.vmr_ranges
        haskey(prof.vmr, s) || error("profile is missing retrieved species $s")
        v = prof.vmr[s]
        x[r] .= spec.log_vmr ? log.(max.(v, _VMR_FLOOR)) : v
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
    for (s, r) in spec.vmr_ranges
        xv = x[r]
        vmr[s] = spec.log_vmr ? exp.(collect(Float64, xv)) : collect(Float64, xv)
    end
    prof = AtmosphericProfile(copy(base_prof.pressure), T, copy(base_prof.altitude), vmr)
    T_sfc = spec.include_tsfc ? Float64(x[spec.tsfc_index]) : T[1]
    ε_sfc = spec.include_emissivity ? Float64(x[spec.emis_index]) : 1.0
    return prof, T_sfc, ε_sfc
end

function Base.show(io::IO, spec::StateVectorSpec)
    blocks = String[]
    spec.include_temperature && push!(blocks, "T($(spec.n_levels))")
    for (s, _) in spec.vmr_ranges
        push!(blocks, "$(spec.log_vmr ? "logVMR" : "VMR")_$(SPECIES_NAME[s])($(spec.n_levels))")
    end
    spec.include_tsfc       && push!(blocks, "T_sfc")
    spec.include_emissivity && push!(blocks, "ε")
    print(io, "StateVectorSpec(n=$(spec.n): ", join(blocks, " + "), ")")
end
