"""
Robust / adaptive-Sₑ retrieval driver — an outer IRLS (iteratively-reweighted
least-squares) loop around [`optimal_estimation`](@ref).

Motivation. The measurement covariance we can *measure* — the EUMETSAT IASI L1C NCM,
or an analytic scene NEΔT — is **instrument noise only**. The total observation error
that optimal estimation actually needs is

    Sₑ = Sₑ_instrument + Sₑ_forward-model + Sₑ_representation ,

and in the CO₂ ν₂ band the forward-model term (line-mixing, far-wing/χ-factor, O₃)
dominates: a plain retrieval leaves residuals whose scatter is √ŝ larger than the
instrument covariance claims (ŝ = χ²/(m−tr A); we measure ŝ ≈ 8 under the *loose*
scene Sₑ, ≈ 30 under the tighter NCM), localized to the ν₂ Q/R-branch spikes.

Rather than hand-mask those channels (throwing away real information and biasing the
fit), this routine keeps the measured covariance as a **hard floor** and lets the
retrieval *learn* how much extra error each channel demands — a per-channel Huber
down-weighting (Andersson & Järvinen 1999 VarQC; Tavolato & Isaksen 2015 Huber-norm),
optionally with a global χ² scale (Dee 1995; Desroziers et al. 2005). The learned
inflation localizes forward-model / spectroscopy error; it is not just a robustness trick.

## Method (one outer iteration)
1. Solve the MAP problem with the current *effective* covariance
   `Sₑ_eff = s · W^{-½} Sₑ_floor W^{-½}`  (`W = Diag(w)`, `s` a scalar; iter 1: `w≡1, s=1`
   ⇒ identical to a plain `optimal_estimation`), warm-started from the previous solution.
2. Form the residual `r = y − F(x̂)` and the normalized departure `zᵢ = rᵢ / (√s · σ_floor,ᵢ)`
   against the **floor** σ (marginal, per-channel — the standard robust-QC convention).
3. Update the Huber weights
       `wᵢ = 1`            if `|zᵢ| ≤ c`   (inlier — trust the floor)
       `wᵢ = c / |zᵢ|`     if `|zᵢ| > c`   (outlier — inflate its variance)
   clamped at `w_min = 1/max_inflation²` so `σ_eff/σ_floor ≤ max_inflation`. Because
   `w ≤ 1`, `Sₑ_eff ≥ Sₑ_floor` **always** — the effective error can only rise above the
   instrument floor, never below it. The effective std dev of an outlier is
   `σᵢ,eff = σ_floor,ᵢ/√wᵢ = σ_floor,ᵢ·√(|zᵢ|/c) = √(σ_floor,ᵢ·|rᵢ|/(√s·c))` — the
   *geometric mean* of the floor and the residual (NOT `|rᵢ|/c`): the standard Huber
   weight `w = c/|z|` makes the outlier's loss grow linearly, so its standardized
   residual under `Sₑ_eff` is `z_eff = √(c·|zᵢ|)`, still rising as `√|z|` rather than
   being clipped to `c`. That sub-quadratic (not zero) residual contribution is why the
   effective χ²/m settles near `~c²` instead of at 1 (see `global_scale` for driving it to 1).
4. (`global_scale=true` only) update the aggregate scale `s ← ŝ = χ²/(m−tr A)`, clamped
   to `≥ 1`, so the floor is never scaled below instrument noise.

The weights are **frozen inside** the inner Levenberg–Marquardt line search — they change
only between outer iterations — so the residual cannot chase itself within a Gauss–Newton
step. The outer loop stops when the weights stabilize (`max|Δw| < weight_tol`) or after
`max_outer` iterations.

## Guards (why this doesn't collapse to σ = |r|)
Setting every channel's σ to its own residual forces χ² = m by construction — consistent
but vacuous, and it discards the floor. The four guardrails above (threshold, `c`-scaling,
NCM floor, lagged reweighting) make this a bounded estimator instead of a tautology.
Setting `max_outer=1` recovers a plain `optimal_estimation`.
"""

using LinearAlgebra: Diagonal, Symmetric, cholesky, diag

"""
    RobustRetrievalResult

Outcome of [`robust_estimation`](@ref).

- `result`      — the final inner [`RetrievalResult`](@ref); all Rodgers diagnostics
  (`A`, `G`, `S_hat`, `dof`, `H`, `chi2`, …) are computed with the *effective* Sₑ.
- `weights`     — final per-channel Huber weights `w` (length `n_y`; `1` on inliers /
  blacklisted channels, `<1` on inflated outliers).
- `σ_floor`     — `sqrt.(diag(Sₑ_floor))`, the hard instrument floor (K, BT space).
- `σ_eff`       — `sqrt.(diag(Sₑ_eff))`, the observation error actually used.
- `z`           — final normalized residuals `rᵢ / (√s·σ_floor,ᵢ)` (length `n_y`).
- `n_outer`     — number of outer IRLS iterations run.
- `outer_converged` — did the weights stabilize before `max_outer`?
- `shat_history`    — ŝ against the **floor** `χ²_floor/(m−tr A)` after each outer solve;
  the excess-over-instrument the retrieval demanded (stays high — that's the diagnostic).
- `s_global`    — final aggregate scale `s` (`1.0` unless `global_scale=true`).
"""
struct RobustRetrievalResult
    result::RetrievalResult
    weights::Vector{Float64}
    σ_floor::Vector{Float64}
    σ_eff::Vector{Float64}
    z::Vector{Float64}
    n_outer::Int
    outer_converged::Bool
    shat_history::Vector{Float64}
    s_global::Float64
end

function Base.show(io::IO, r::RobustRetrievalResult)
    ninfl = count(<(1.0 - 1e-9), r.weights)
    print(io, "RobustRetrievalResult(", r.n_outer, " outer iter",
          r.outer_converged ? "" : " [weights NOT stable]", ", ",
          ninfl, "/", length(r.weights), " channels inflated, ",
          "ŝ_floor=", isempty(r.shat_history) ? "?" : round(r.shat_history[end], sigdigits=3),
          ", ", r.result, ")")
end

# Build the effective covariance Sₑ_eff = s · W^{-½} Sₑ_floor W^{-½} from the floor and
# the current weights. Diagonal in → Diagonal out (keeps the cheap O(n) solve);
# a dense (apodized/scene/NCM) floor has its variances inflated while the correlation
# coefficients are preserved (row/column scaling).
function _apply_weights(Se_floor::AbstractMatrix, w::AbstractVector, s::Real)
    inv_sqrt_w = 1.0 ./ sqrt.(w)
    if Se_floor isa Diagonal
        return Diagonal(s .* collect(diag(Se_floor)) .* inv_sqrt_w .^ 2)
    end
    D = Diagonal(inv_sqrt_w)
    return Symmetric(s .* (D * (Matrix(Se_floor) * D)))
end

"""
    robust_estimation(y, spec, base_prof, linelists;
                      xa, Sa, Se, channel_mask=nothing,
                      huber_c=1.5, max_inflation=10.0,
                      max_outer=5, weight_tol=0.05,
                      global_scale=false, verbose=false,
                      kwargs...) -> RobustRetrievalResult

Robust optimal estimation: an outer Huber-IRLS loop around [`optimal_estimation`](@ref)
that treats `Se` as a **hard floor** (the measured instrument covariance, e.g. the IASI
NCM or a scene NEΔT) and inflates the per-channel observation error where the retrieval
cannot fit the data — localizing forward-model / spectroscopy error instead of masking it.

- `y`, `spec`, `base_prof`, `linelists`, `xa`, `Sa`, `channel_mask` — as in
  [`optimal_estimation`](@ref). `channel_mask` still hard-drops channels; robust
  reweighting is a *soft* alternative applied to the kept ones.
- `Se` — the **floor** covariance. The effective covariance only ever rises above it.
- `huber_c` — Huber threshold `c` on the normalized residual (default `1.5`; larger ⇒
  fewer channels reweighted).
- `max_inflation` — cap on `σ_eff/σ_floor` (default `10`), i.e. `w ≥ 1/max_inflation²`.
- `max_outer` — outer IRLS iterations (default `5`; `1` reproduces `optimal_estimation`).
- `weight_tol` — outer convergence when `max|Δw| < weight_tol` (default `0.1`). The
  aggregate diagnostics (ŝ_floor, χ²/m) typically settle by the 2nd outer iteration; this
  tol stops once the per-channel weights are pinned to ~10%. Tighten (e.g. `0.05`) only if
  you need fully-settled per-channel σ_eff, at ~2× the outer iterations.
- `global_scale` — also apply the aggregate Dee/Desroziers scale `s ← χ²/(m−tr A)`
  (clamped `≥1`) each outer step, on top of the per-channel weights (default `false`).
- `verbose` — stream per-outer-iteration ŝ / inflated-channel counts.
- `kwargs...` — everything else (`ε_fixed`, `fm_kwargs`, `method`, `max_iter`,
  `conv_factor`, `observable`, …) is forwarded to each inner `optimal_estimation` call.
  Inner `verbose` is **off** (the outer loop owns the log); pass no inner `verbose`.

Returns a [`RobustRetrievalResult`](@ref) wrapping the final inner solve plus the learned
weights, effective σ, and ŝ history.
"""
function robust_estimation(y::AbstractVector{<:Real},
                           spec::StateVectorSpec,
                           base_prof::AtmosphericProfile,
                           linelists::Dict{GasSpecies, HITRANLinelist};
                           xa::AbstractVector{<:Real},
                           Sa::AbstractMatrix{<:Real},
                           Se::AbstractMatrix{<:Real},
                           channel_mask::Union{Nothing,AbstractVector{Bool}} = nothing,
                           huber_c::Float64 = 1.5,
                           max_inflation::Float64 = 10.0,
                           max_outer::Int = 5,
                           weight_tol::Float64 = 0.1,
                           global_scale::Bool = false,
                           x0::AbstractVector{<:Real} = copy(xa),
                           verbose::Bool = false,
                           kwargs...)::RobustRetrievalResult
    huber_c > 0 || error("huber_c must be > 0, got $huber_c")
    max_inflation >= 1 || error("max_inflation must be ≥ 1, got $max_inflation")
    max_outer >= 1 || error("max_outer must be ≥ 1, got $max_outer")
    n_y = length(y)
    size(Se) == (n_y, n_y) || error("Se must be n_y×n_y with n_y = length(y) = $n_y")

    keep = channel_mask === nothing ? trues(n_y) : convert(Vector{Bool}, channel_mask)
    kidx = findall(keep)
    w_min = 1.0 / max_inflation^2

    # Floor σ (marginal) and its kept sub-block factorization, for the ŝ_floor diagnostic
    # χ²_floor = r_kᵀ Sₑ_floor[keep,keep]⁻¹ r_k (the excess over instrument noise).
    σ_floor = sqrt.(collect(diag(Se)))
    Se_floor_k = Se isa Diagonal ? Diagonal(collect(diag(Se))[kidx]) : Matrix(Se)[kidx, kidx]
    floor_fac  = Se_floor_k isa Diagonal ? Se_floor_k : cholesky(Symmetric(Se_floor_k))

    w = ones(Float64, n_y)
    s = 1.0
    shat_history = Float64[]
    local result::RetrievalResult
    x_warm = collect(Float64, x0)
    outer_converged = false
    n_outer = 0

    for outer in 1:max_outer
        n_outer = outer
        Se_eff = _apply_weights(Se, w, s)
        result = optimal_estimation(y, spec, base_prof, linelists;
                                    xa=xa, Sa=Sa, Se=Se_eff, channel_mask=channel_mask,
                                    x0=x_warm, verbose=false, kwargs...)
        x_warm = result.x

        # Residual at the solution (full grid), and ŝ against the FLOOR (kept channels).
        r  = result.y_fit .- collect(Float64, y)
        rk = r[kidx]
        χ2_floor = rk' * (floor_fac \ rk)
        m  = length(kidx)
        shat_floor = χ2_floor / (m - result.dof)
        push!(shat_history, shat_floor)

        # Normalized departures against the (globally-scaled) floor, then Huber weights.
        # Only the kept channels are reweighted; blacklisted ones keep w = 1 (unused).
        z = r ./ (sqrt(s) .* σ_floor)
        w_new = copy(w)
        for i in kidx
            az = abs(z[i])
            w_new[i] = az <= huber_c ? 1.0 : max(huber_c / az, w_min)
        end
        Δw = maximum(abs.(w_new .- w))

        if verbose
            ninfl = count(<(1.0 - 1e-9), @view w_new[kidx])
            maxσr = maximum(1.0 ./ sqrt.(@view w_new[kidx]))
            println("  outer $outer: ŝ_floor=$(round(shat_floor, sigdigits=3)) ",
                    "χ²/m=$(round(result.chi2 / m, sigdigits=3)) DOF=$(round(result.dof, digits=2)) ",
                    "inflated=$ninfl/$m maxσ_eff/σ=$(round(maxσr, digits=2)) ",
                    "s=$(round(s, sigdigits=3)) Δw=$(round(Δw, sigdigits=3))")
            flush(stdout)
        end

        w = w_new
        global_scale && (s = max(shat_floor, 1.0))

        if Δw < weight_tol
            outer_converged = true
            # One more solve so `result`/diagnostics reflect the final weights.
            Se_eff = _apply_weights(Se, w, s)
            result = optimal_estimation(y, spec, base_prof, linelists;
                                        xa=xa, Sa=Sa, Se=Se_eff, channel_mask=channel_mask,
                                        x0=x_warm, verbose=false, kwargs...)
            break
        end
    end

    Se_eff_final = _apply_weights(Se, w, s)
    σ_eff = sqrt.(collect(diag(Se_eff_final isa Diagonal ? Se_eff_final : Matrix(Se_eff_final))))
    z = (result.y_fit .- collect(Float64, y)) ./ (sqrt(s) .* σ_floor)

    return RobustRetrievalResult(result, w, σ_floor, σ_eff, z, n_outer,
                                 outer_converged, shat_history, s)
end
