"""
A-priori covariance (`Sₐ`) construction for optimal estimation.

`Sₐ` regularises the retrieval by tying neighbouring profile levels together. The
correlation is imposed in **log-pressure**, the natural vertical coordinate for
sounding (one unit of `|ln(pᵢ/pⱼ)|` ≈ one scale height), following the thesis form
`Sₐ[i,j] = σᵢ σⱼ · c(|ln(pᵢ/pⱼ)| / L)` (§3.3).

The choice of correlation function `c` sets the **smoothness of the a-priori
profiles** (the sample paths of the implied Gaussian prior), which is what keeps the
retrieval from developing unphysical kinks:

| `kernel`        | `c(r)`                                  | profile smoothness  |
|-----------------|-----------------------------------------|---------------------|
| `:exponential`  | `exp(−r)`                               | C⁰ — cusp/kink at every level (Markov; Rodgers default) |
| `:matern32`     | `(1+√3 r)·exp(−√3 r)`                    | C¹ — once differentiable |
| `:matern52`     | `(1+√5 r + 5r²/3)·exp(−√5 r)`           | **C² — twice differentiable (default)** |
| `:sqexp`        | `exp(−r²/2)`                            | C^∞ — analytic, but `Sₐ` is near-singular on dense grids |

`r = |ln(pᵢ/pⱼ)| / L`. **Matérn-5/2 is the default**: its sample paths are
continuous to second order (no kinks) while `Sₐ` stays well-conditioned, unlike the
squared-exponential. Use `:exponential` to reproduce a classic Rodgers prior.
"""

using LinearAlgebra: Symmetric, isposdef

# Correlation kernels c(r), r = lag/L ≥ 0, normalised so c(0) = 1. The √(2ν) inside
# the Matérn forms makes L the common range scale across kernels.
@inline _corr(::Val{:exponential}, r::Float64) = exp(-r)
@inline _corr(::Val{:matern32}, r::Float64)    = (1.0 + sqrt(3.0)*r) * exp(-sqrt(3.0)*r)
@inline _corr(::Val{:matern52}, r::Float64)    =
    (1.0 + sqrt(5.0)*r + 5.0*r^2/3.0) * exp(-sqrt(5.0)*r)
@inline _corr(::Val{:sqexp}, r::Float64)       = exp(-0.5*r^2)

# Resolve a σ argument (scalar → filled, vector → checked) to a length-n vector.
function _resolve_σ(σ, n::Int, name::String)::Vector{Float64}
    if σ isa Real
        σ > 0 || error("$name must be positive")
        return fill(Float64(σ), n)
    end
    v = collect(Float64, σ)
    length(v) == n || error("$name must be a scalar or length n_levels=$n; got $(length(v))")
    all(>(0.0), v) || error("$name entries must all be positive")
    return v
end

# Per-species σ / L: accept a scalar/vector (applied to every species) or a Dict.
_species_σ(σ_vmr, s::GasSpecies, n::Int)::Vector{Float64} =
    σ_vmr isa AbstractDict ?
        _resolve_σ(get(() -> error("σ_vmr has no entry for $(SPECIES_NAME[s])"), σ_vmr, s),
                   n, "σ_vmr[$(SPECIES_NAME[s])]") :
        _resolve_σ(σ_vmr, n, "σ_vmr")

function _species_L(L_vmr, s::GasSpecies)::Float64
    L = L_vmr isa AbstractDict ?
        Float64(get(() -> error("L_vmr has no entry for $(SPECIES_NAME[s])"), L_vmr, s)) :
        Float64(L_vmr)
    L > 0 || error("L_vmr must be positive")
    return L
end

# Correlated N×N block: Bᵢⱼ = σᵢ σⱼ · c(|Δlogp|/L).
function _corr_block(logp::Vector{Float64}, σ::Vector{Float64}, L::Float64,
                     kern::Val)::Matrix{Float64}
    n = length(logp)
    B = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        r = abs(logp[i] - logp[j]) / L
        B[i, j] = σ[i] * σ[j] * _corr(kern, r)
    end
    return B
end

"""
    build_sa(spec, base_prof;
             σ_T=2.0, L_T=1.0, σ_vmr=0.5, L_vmr=1.0,
             σ_tsfc=2.0, σ_emis=0.02,
             kernel=:matern52, jitter=0.0) -> Symmetric{Float64}

Build the a-priori covariance `Sₐ` for the state described by `spec`, with each
profile field (temperature, every retrieved VMR species) carried as a vertically
correlated block and `T_sfc`/`ε` as independent scalars. The correlation is in
log-pressure (from `base_prof.pressure`) using the `kernel` smoothness above;
cross-field correlations are zero.

# Variances / correlation lengths
- `σ_T`    : temperature std (K) — scalar or per-level vector (length `n_levels`).
- `σ_vmr`  : **log-VMR** std (since the state is `log(VMR)`; `σ=0.5` ≈ a factor-of-1.65
             1-σ spread). Scalar/vector (all species) **or** a `Dict{GasSpecies}` of
             scalars/vectors for per-species control.
- `L_T`, `L_vmr` : correlation length in log-pressure (≈ scale heights; `L=1` ties
             levels ~one scale height apart). `L_vmr` may be a `Dict` per species.
- `σ_tsfc` : surface-temperature std (K); used iff `spec.include_tsfc`.
- `σ_emis` : emissivity std; used iff `spec.include_emissivity`.
- `jitter` : relative diagonal inflation `Sₐ[i,i] *= (1+jitter)`; a small value
             (e.g. `1e-6`) rescues the near-singular `:sqexp` kernel. Default `0`.

The result is positive-definite for the default kernels with `σ>0`; a warning is
issued otherwise. Pass it straight to `optimal_estimation` as `Sa`.
"""
function build_sa(spec::StateVectorSpec, base_prof::AtmosphericProfile;
                  σ_T = 2.0, L_T::Real = 1.0,
                  σ_vmr = 0.5, L_vmr = 1.0,
                  σ_tsfc::Real = 2.0, σ_emis::Real = 0.02,
                  kernel::Symbol = :matern52,
                  jitter::Real = 0.0)::Symmetric{Float64, Matrix{Float64}}
    kernel in (:matern52, :matern32, :exponential, :sqexp) ||
        error("kernel must be :matern52, :matern32, :exponential, or :sqexp; got :$kernel")
    length(base_prof.pressure) == spec.n_levels ||
        error("base_prof has $(length(base_prof.pressure)) levels, spec expects $(spec.n_levels)")
    all(>(0.0), base_prof.pressure) || error("pressures must be positive for log-pressure correlation")
    jitter >= 0 || error("jitter must be ≥ 0")

    logp = log.(collect(Float64, base_prof.pressure))
    kern = Val(kernel)
    Sa   = zeros(Float64, spec.n, spec.n)

    if spec.include_temperature
        σ = _resolve_σ(σ_T, spec.n_levels, "σ_T")
        Sa[spec.temp_range, spec.temp_range] .= _corr_block(logp, σ, Float64(L_T), kern)
    end
    for (s, r) in spec.vmr_ranges
        σ = _species_σ(σ_vmr, s, spec.n_levels)
        L = _species_L(L_vmr, s)
        Sa[r, r] .= _corr_block(logp, σ, L, kern)
    end
    spec.include_tsfc       && (Sa[spec.tsfc_index, spec.tsfc_index] = Float64(σ_tsfc)^2)
    spec.include_emissivity && (Sa[spec.emis_index, spec.emis_index] = Float64(σ_emis)^2)

    if jitter > 0
        @inbounds for i in 1:spec.n
            Sa[i, i] *= (1.0 + jitter)
        end
    end

    S = Symmetric(Sa)
    isposdef(S) || @warn("build_sa: Sₐ is not positive-definite — try a larger `jitter`, " *
                         "a shorter correlation length, or kernel=:exponential/:matern32.")
    return S
end
