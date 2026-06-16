"""
Optimal-estimation retrieval driver (roadmap Phase 5).

Turns the analytic Jacobian into a maximum-a-posteriori retrieval, following Rodgers
(2000), *Inverse Methods for Atmospheric Sounding*. Minimises the regularised cost

    J(x) = (y − F(x))ᵀ Sₑ⁻¹ (y − F(x)) + (x − xₐ)ᵀ Sₐ⁻¹ (x − xₐ)

with a Levenberg–Marquardt iteration (Rodgers Eq. 5.36; `γ → 0` recovers Gauss–Newton):

    Δx   = [(1+γ)·Sₐ⁻¹ + KᵀSₑ⁻¹K]⁻¹ · [KᵀSₑ⁻¹(y − F(xᵢ)) − Sₐ⁻¹(xᵢ − xₐ)]
    xᵢ₊₁ = xᵢ + Δx

`γ` is decreased on an accepted (cost-reducing) step and increased on a rejected one.
The forward model `F` is the **exact** full forward model (`iasi_forward_model`,
`dptmn=0`), so the retrieval converges to the true MAP solution even though the VMR
columns of `K` carry the deferred Phase-2b coupling — an approximate `K` changes the
step, not the fixed point where `∇J = 0`.

State packing, units (log-VMR), and the fixed-field handling come from
`StateVectorSpec` / `unpack_state`; `xₐ`, `Sₐ`, `Sₑ` all live in that state space.

Diagnostics at the solution (Rodgers Ch. 2–3): posterior covariance
`Ŝ = (Sₐ⁻¹ + KᵀSₑ⁻¹K)⁻¹`, averaging kernel `A = Ŝ·KᵀSₑ⁻¹K = ∂x̂/∂x`, and the degrees
of freedom for signal `DOF = tr(A)`.
"""

using LinearAlgebra: Symmetric, inv, tr, I, cholesky, issuccess

"""
    RetrievalResult

Outcome of `optimal_estimation`. Fields: `x` (retrieved state), `spec`, `converged`,
`n_iter`, `cost` (history, length `n_iter+1`), `S_hat` (posterior covariance), `A`
(averaging kernel), `dof` (`tr A`), `chi2` (`(y−F)ᵀSₑ⁻¹(y−F)` at the solution),
`y_fit` (`F(x̂)`), `ν` (channel grid).
"""
struct RetrievalResult
    x::Vector{Float64}
    spec::StateVectorSpec
    converged::Bool
    n_iter::Int
    cost::Vector{Float64}
    S_hat::Matrix{Float64}
    A::Matrix{Float64}
    dof::Float64
    chi2::Float64
    y_fit::Vector{Float64}
    ν::Vector{Float64}
end

function Base.show(io::IO, r::RetrievalResult)
    print(io, "RetrievalResult($(r.converged ? "converged" : "NOT converged") in ",
          "$(r.n_iter) iter, DOF=$(round(r.dof, digits=2)), ",
          "χ²=$(round(r.chi2, digits=3)), ", r.spec, ")")
end

# Regularised cost J(x) given a residual (y − F) and state departure (x − xa).
@inline function _oe_cost(resid::AbstractVector, Se_inv_resid::AbstractVector,
                          dx::AbstractVector, Sa_inv_dx::AbstractVector)
    return resid' * Se_inv_resid + dx' * Sa_inv_dx
end

"""
    optimal_estimation(y, spec, base_prof, linelists;
                       xa, Sa, Se, x0=copy(xa),
                       method=:levenberg_marquardt,
                       max_iter=15, max_linesearch=8,
                       γ0=1.0, γ_factor=3.0, γ_min=1e-4,
                       conv_factor=1e-2, observable=:bt,
                       fm_kwargs=(;), verbose=false) -> RetrievalResult

Retrieve the state `x` from the observed spectrum `y` by optimal estimation.

- `spec`, `base_prof`, `linelists` — define the state layout and the fixed forward
  context (`base_prof` supplies pressure/altitude/non-retrieved species via
  `unpack_state`).
- `xa`, `Sa` — a-priori state and its covariance (state space; VMR in log).
- `Se` — measurement-error covariance (channel space); a `Diagonal` is efficient.
- `x0` — first guess (default `xa`).
- `method` — `:levenberg_marquardt` (default) or `:gauss_newton` (`γ ≡ 0`).
- convergence: stop when the Rodgers step size `dᵢ² = Δxᵀ Ŝ⁻¹ Δx < conv_factor·n`.

`fm_kwargs` is forwarded to both `analytic_jacobian` (for `K`) and `iasi_forward_model`
(for the line-search `F`, with `dptmn=0`); pass forward-model options (`iasi`,
`apply_continuum`, `with_ils`, `source_function`, …) here — **not** `T_sfc`/`ε_sfc`
(those live in the state) or `dptmn`.
"""
function optimal_estimation(y::AbstractVector{<:Real},
                            spec::StateVectorSpec,
                            base_prof::AtmosphericProfile,
                            linelists::Dict{GasSpecies, HITRANLinelist};
                            xa::AbstractVector{<:Real},
                            Sa::AbstractMatrix{<:Real},
                            Se::AbstractMatrix{<:Real},
                            x0::AbstractVector{<:Real} = copy(xa),
                            method::Symbol = :levenberg_marquardt,
                            max_iter::Int = 15,
                            max_linesearch::Int = 8,
                            γ0::Float64 = 1.0,
                            γ_factor::Float64 = 3.0,
                            γ_min::Float64 = 1e-4,
                            conv_factor::Float64 = 1e-2,
                            observable::Symbol = :bt,
                            fm_kwargs = (;),
                            verbose::Bool = false)::RetrievalResult
    method in (:levenberg_marquardt, :gauss_newton) ||
        error("method must be :levenberg_marquardt or :gauss_newton, got :$method")
    n = spec.n
    length(xa) == n || error("xa length $(length(xa)) ≠ spec.n $n")
    size(Sa) == (n, n) || error("Sa must be $n×$n")
    length(y) == size(Se, 1) == size(Se, 2) ||
        error("Se must be n_y×n_y with n_y = length(y) = $(length(y))")

    Sa_inv = Matrix(inv(Sa))
    xa_v   = collect(Float64, xa)
    yv     = collect(Float64, y)

    # Forward-only F(x) (exact full model, frozen cutoff) for the line search.
    forward(x) = let (p, Ts, εs) = unpack_state(spec, x, base_prof)
        νg, R, BT = iasi_forward_model(p, linelists; T_sfc=Ts, ε_sfc=εs,
                                       dptmn=0.0, fm_kwargs...)
        observable === :bt ? BT : R
    end
    # F(x) and K(x) together (analytic Jacobian's y0 matches `forward` to round-off).
    forward_jac(x) = let (p, Ts, εs) = unpack_state(spec, x, base_prof)
        jac = analytic_jacobian(p, linelists, spec; T_sfc=Ts, ε_sfc=εs,
                                observable=observable, fm_kwargs...)
        jac.y0, jac.K, collect(Float64, jac.ν)
    end

    x = collect(Float64, x0)
    F, K, νout = forward_jac(x)
    resid = yv .- F
    Sei_r = Se \ resid
    dx_a  = x .- xa_v
    J = _oe_cost(resid, Sei_r, dx_a, Sa_inv * dx_a)
    cost_hist = Float64[J]

    γ = method === :gauss_newton ? 0.0 : γ0
    converged = false
    iter = 0
    while iter < max_iter
        iter += 1
        SeiK    = Se \ K                       # Sₑ⁻¹ K   (n_y × n)
        KtSeiK  = Symmetric(K' * SeiK)         # KᵀSₑ⁻¹K  (n × n)
        g       = K' * Sei_r .- Sa_inv * (x .- xa_v)   # gradient term

        accepted = false
        for _ in 1:max_linesearch
            H  = Symmetric((1.0 + γ) .* Sa_inv .+ KtSeiK)
            Δx = H \ g
            x_try = x .+ Δx
            F_try = forward(x_try)
            r_try = yv .- F_try
            Sei_rt = Se \ r_try
            dxa_t  = x_try .- xa_v
            J_try  = _oe_cost(r_try, Sei_rt, dxa_t, Sa_inv * dxa_t)

            if J_try < J
                # Rodgers step-size convergence test dᵢ² = Δxᵀ Ŝ⁻¹ Δx.
                d2 = Δx' * (Sa_inv * Δx .+ KtSeiK * Δx)
                x = x_try
                F, K, νout = forward_jac(x)
                resid = yv .- F
                Sei_r = Se \ resid
                J = J_try
                push!(cost_hist, J)
                method === :gauss_newton || (γ = max(γ / γ_factor, γ_min))
                accepted = true
                verbose && println("  iter $iter: J=$(round(J,sigdigits=6)) d²=$(round(d2,sigdigits=3)) γ=$γ")
                d2 < conv_factor * n && (converged = true)
                break
            else
                method === :gauss_newton && break   # no damping to fall back on
                γ *= γ_factor
            end
        end
        (converged || !accepted) && break
    end

    # Diagnostics at the solution.
    SeiK   = Se \ K
    KtSeiK = Symmetric(K' * SeiK)
    S_hat  = Matrix(inv(Symmetric(Sa_inv .+ KtSeiK)))
    A      = S_hat * (K' * SeiK)
    chi2   = resid' * (Se \ resid)

    return RetrievalResult(x, spec, converged, iter, cost_hist,
                           S_hat, A, tr(A), chi2, F, νout)
end
