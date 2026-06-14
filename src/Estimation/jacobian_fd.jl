"""
Finite-difference Jacobian reference harness (roadmap Phase 0).

This is the **ground truth** against which every later analytic / AD Jacobian
(Phases 1–3) is validated — the same discipline used for the LBLRTM/ARTS
forward-model validation. It perturbs each state element and re-runs the full
forward model, so it is slow (`2·n_state` forward evaluations for central
differences) but makes no approximation beyond the finite step itself.

## Cutoff-freezing policy (roadmap §6.1)

The forward model carries state-dependent active sets that introduce derivative
**discontinuities**: the `dptmn` per-layer weak-line rejection, the `min_band_strength`
line-mixing band cutoff, and the `max(σ,0)` clamp in the Voigt kernel. Under a
finite perturbation a line/band can cross such a threshold, contaminating the
difference quotient with a jump.

Policy adopted here, and to be matched by the analytic Jacobians: **disable the
lossy line rejection** by forcing `dptmn=0.0` for every forward evaluation (the
degenerate "freeze the active set with everything on"). `min_band_strength` is left
at its exact default of `0.0`. The caller may override `dptmn` via `fm_kwargs`, but
the default makes the reference clean. (The `max(σ,0)` clamp lives inside the kernel
and is shared identically by all evaluations, so it does not bias a difference
quotient unless a channel sits exactly on the kink — rare, and visible as FD noise.)

## Step sizes

`default_fd_steps` returns a per-element step aligned to `x`. Note the partition
function is tabulated at **1 K** resolution (roadmap §3), so very small temperature
steps probe the within-cell linear interpolation; the default 0.5 K is a compromise
between truncation error and that quantization. Tune via `steps` and check
convergence by step-halving if a column looks noisy.
"""

"""
    Jacobian

Result of a Jacobian evaluation. `K` is `n_y × n_x` = `∂y/∂x` at the linearization
point `x0`, with `y0 = forward(x0)`. Rows correspond to `ν` (channel wavenumbers);
columns correspond to `state_labels(spec)`. `observable` is `:bt` (brightness
temperature, K) or `:radiance` (mW/m²/sr/cm⁻¹).
"""
struct Jacobian
    K::Matrix{Float64}          # n_y × n_x  (∂y/∂x)
    spec::StateVectorSpec
    ν::Vector{Float64}          # channel wavenumbers (rows of K)
    x0::Vector{Float64}         # linearization point
    y0::Vector{Float64}         # forward observable at x0
    observable::Symbol          # :bt | :radiance
end

"""
    column(jac, label) -> Vector{Float64}

Return the Jacobian column for the state element named `label` (see `state_labels`).
"""
function column(jac::Jacobian, label::AbstractString)::Vector{Float64}
    labels = state_labels(jac.spec)
    i = findfirst(==(label), labels)
    isnothing(i) && error("no state element labelled \"$label\"")
    return jac.K[:, i]
end

function Base.show(io::IO, jac::Jacobian)
    print(io, "Jacobian($(size(jac.K, 1))×$(size(jac.K, 2)) ∂$(jac.observable)/∂x, ",
          jac.spec, ")")
end

"""
    default_fd_steps(spec; δT=0.5, δlogvmr=1e-3, δvmr=1e-9, δtsfc=0.5, δε=1e-3)

Per-element finite-difference step vector aligned to the state layout. The VMR step
is `δlogvmr` in log-space (`spec.log_vmr=true`, recommended) or the absolute `δvmr`
otherwise. See the module docstring on the 1 K partition-table quantization.
"""
function default_fd_steps(spec::StateVectorSpec;
                          δT::Float64       = 0.5,
                          δlogvmr::Float64  = 1e-3,
                          δvmr::Float64     = 1e-9,
                          δtsfc::Float64    = 0.5,
                          δε::Float64       = 1e-3)::Vector{Float64}
    s = zeros(Float64, spec.n)
    spec.include_temperature && (s[spec.temp_range] .= δT)
    for (_, r) in spec.vmr_ranges
        s[r] .= spec.log_vmr ? δlogvmr : δvmr
    end
    spec.include_tsfc       && (s[spec.tsfc_index] = δtsfc)
    spec.include_emissivity && (s[spec.emis_index] = δε)
    return s
end

"""
    finite_difference_jacobian(prof, linelists, spec;
                               T_sfc=nothing, ε_sfc=1.0,
                               observable=:bt,
                               steps=default_fd_steps(spec),
                               method=:central,
                               fm_kwargs=(;),
                               verbose=false) -> Jacobian

Build the finite-difference reference Jacobian of the IASI forward model about the
state packed from `prof` (with surface `T_sfc`, `ε_sfc`). `method` is `:central`
(default, 2nd-order, `2·n_state` evaluations) or `:forward` (1st-order, `n_state+1`).

`fm_kwargs` is forwarded to `iasi_forward_model` (e.g. `iasi`, `apply_continuum`,
`with_ils`, `line_mixing`, `cutoff`); it is merged **over** the frozen-cutoff default
`dptmn=0.0` (see the §6.1 policy in the module docstring), so passing `dptmn` here
re-enables rejection if you really want it.
"""
function finite_difference_jacobian(prof::AtmosphericProfile,
                                     linelists::Dict{GasSpecies, HITRANLinelist},
                                     spec::StateVectorSpec;
                                     T_sfc::Union{Float64, Nothing} = nothing,
                                     ε_sfc::Float64   = 1.0,
                                     observable::Symbol = :bt,
                                     steps::Vector{Float64} = default_fd_steps(spec),
                                     method::Symbol   = :central,
                                     fm_kwargs        = (;),
                                     verbose::Bool    = false)::Jacobian
    observable in (:bt, :radiance) ||
        error("observable must be :bt or :radiance, got :$observable")
    method in (:central, :forward) ||
        error("method must be :central or :forward, got :$method")
    length(steps) == spec.n ||
        error("steps length $(length(steps)) ≠ spec.n $(spec.n)")

    # §6.1 cutoff-freezing: disable lossy line rejection unless the caller overrides.
    kw = merge((dptmn = 0.0,), fm_kwargs)

    # Forward evaluation: state vector → observable spectrum on the channel grid.
    # NB: the inner grid is `νg`, not `ν` — sharing the name with the outer `ν`
    # would make it a captured binding the loop clobbers (Julia closure scoping).
    function run(x)
        p, Ts, εs = unpack_state(spec, x, prof)
        νg, R, BT = iasi_forward_model(p, linelists; T_sfc = Ts, ε_sfc = εs, kw...)
        y = observable === :bt ? BT : R
        return collect(Float64, νg.ν), y
    end

    x0 = pack_state(spec, prof; T_sfc = T_sfc, ε_sfc = ε_sfc)
    ν, y0 = run(x0)
    n_y, n_x = length(y0), spec.n
    K = Matrix{Float64}(undef, n_y, n_x)

    for i in 1:n_x
        δ = steps[i]
        δ == 0.0 && error("zero FD step for state element $i ($(state_labels(spec)[i]))")
        if method === :central
            xp = copy(x0); xp[i] += δ
            xm = copy(x0); xm[i] -= δ
            _, yp = run(xp)
            _, ym = run(xm)
            @views K[:, i] .= (yp .- ym) ./ (2δ)
        else
            xp = copy(x0); xp[i] += δ
            _, yp = run(xp)
            @views K[:, i] .= (yp .- y0) ./ δ
        end
        verbose && (i % 10 == 0 || i == n_x) &&
            println("  FD Jacobian: column $i / $n_x")
    end

    return Jacobian(K, spec, ν, x0, y0, observable)
end
