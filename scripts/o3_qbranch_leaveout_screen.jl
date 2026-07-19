# Does the CO₂ ν₂ Q-branch bias the T retrieval and leak into the 715 spike?
#
# Hypothesis (user): unaccounted spectral structure in the 721 (and 667) Q-branch —
# which carry the majority of the fit's χ² — drags T(p) to a wrong solution, and that
# T-bias shows up as the +3 K residual at 715.5/715.75.
#
# Cheap linearized test (NO re-retrieval): linearize F at the converged joint state x̂
# (K = rJ.K), then re-solve the OE update using only the KEPT channels (Q-branch
# dropped). With a dense scene Se, "dropping" = use inv(Se_KK) on the kept sub-block:
#     x̂_new = xa + [Sa⁻¹ + K_Kᵀ Se_KK⁻¹ K_K]⁻¹ K_Kᵀ Se_KK⁻¹ [ (y−F(x̂))_K + K_K(x̂−xa) ]
# Report ΔT(p) = x̂_new−x̂ on the temperature block, and the PREDICTED residual at the
# spike channels, res_new(j) = (y−F(x̂))_j − K[j,:]·(x̂_new−x̂).
# Sanity: keep=ALL must reproduce x̂ (Δ≈0). If dropping the Q-branch barely moves the
# 715 residual → spike is local, not a Q-branch T-bias; if it collapses → user is right
# (then confirm with a full masked re-retrieval).
#
#   julia --project=. scripts/o3_qbranch_leaveout_screen.jl

using IRSounderLBL
using LinearAlgebra
using Printf
using JLD2: load

# ── Reconstruct the exact joint problem (mirror retrieve_iasi_joint.jl) ──────────────
bl = load("data/iasi_profile_retrieval.jld2")
νobs = bl["ν"]; y = bl["y"]; Se = Matrix(bl["Se"])
J = load("data/iasi_joint.jld2"); rJ = J["rJ"]; blocks = J["blocks"]

base = afgl_atmosphere(:us_standard)
base.vmr[CO2] .*= (432.0*1e-6)/base.vmr[CO2][1]
nlev = n_levels(base)
B_h2o = partial_column_basis(nlev, blocks; taper=:boxcar)
spec  = StateVectorSpec(nlev, [PartialColumns(H2O, B_h2o), ColumnScale(O3)];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0, σ_col=Dict(H2O=>0.5, O3=>0.2), σ_tsfc=5.0)
xa = pack_state(spec, base)

K  = rJ.K                       # 621×55, ∂y/∂x at x̂
x̂  = rJ.x
d  = y .- rJ.y_fit              # residual (y − F(x̂)) at the converged state
Sa_inv = inv(Matrix(Sa))
p  = base.pressure
Trng = spec.temp_range

# Linearized OE solution using only channels in `keep` (Bool mask). Returns x_new.
function relsolve(keep::BitVector)
    K_K   = @view K[keep, :]
    Se_KK = Se[keep, keep]
    F     = cholesky(Symmetric(Se_KK))          # Se_KK⁻¹ via factor
    rhs   = d[keep] .+ K_K * (x̂ .- xa)          # (y−F(x̂))_K + K_K(x̂−xa)
    b     = K_K' * (F \ rhs)
    H     = Sa_inv .+ K_K' * (F \ Matrix(K_K))
    return xa .+ (Symmetric(H) \ b)
end

# Spike + comparison channels.
chans = (715.25, 715.50, 715.75, 716.00, 721.00, 725.25, 725.50, 725.75)
jj = [argmin(abs.(νobs .- c)) for c in chans]

# Q-branch masks to test (drop these from the fit).
masks = (
  ("721 Q only  [719–723]",        (νobs .>= 719) .& (νobs .<= 723)),
  ("667 Q only  [665–668]",        (νobs .>= 665) .& (νobs .<= 668)),
  ("both Q-branches",              ((νobs .>= 719) .& (νobs .<= 723)) .| ((νobs .>= 665) .& (νobs .<= 668))),
)

# Sanity: keep=all reproduces x̂.
x_all = relsolve(trues(length(y)))
@printf("SANITY keep=ALL: max|x_new−x̂| = %.2e  (T block %.2e K)  — should be ~0\n\n",
        maximum(abs.(x_all .- x̂)), maximum(abs.((x_all.-x̂)[Trng])))

println("="^96)
@printf("%-24s %10s | predicted residual at spike channels (K)\n", "drop from fit", "maxΔT(p)")
@printf("%-24s %10s |", "", "")
for c in chans; @printf(" %7.2f", c); end; println()
@printf("%-24s %10s |", "baseline res₀", ""); for j in jj; @printf(" %+7.3f", d[j]); end; println()
println("-"^96)
for (name, qm) in masks
    keep = .!BitVector(qm)
    xn = relsolve(keep)
    Δx = xn .- x̂
    ΔF = K * Δx                       # change in modeled BT at every channel
    res_new = d .- ΔF                 # predicted new residual
    maxdT = maximum(abs.(Δx[Trng]))
    @printf("%-24s %10.3f |", name * "  (n=$(sum(qm)))", maxdT)
    for j in jj; @printf(" %+7.3f", res_new[j]); end; println()
end
println("="^96)

# Detail: for the 721-Q drop, show the ΔT(p) profile where it moves most.
qm = (νobs .>= 719) .& (νobs .<= 723)
Δx = relsolve(.!BitVector(qm)) .- x̂
ΔT = Δx[Trng]
ord = sortperm(abs.(ΔT); rev=true)[1:8]
println("\nDrop 721-Q: largest T(p) shifts (level, pressure, ΔT):")
for i in sort(ord)
    @printf("  lvl %2d  p=%8.2f hPa  ΔT=%+.3f K\n", i, p[i], ΔT[i])
end
@printf("\nInterpretation: if the 715.50/715.75 predicted residual stays ≈ baseline when the\n")
@printf("Q-branch is dropped, the spike is NOT a Q-branch-driven T bias. If it collapses\n")
@printf("toward 0, it is — confirm with a full masked re-retrieval.\n")
