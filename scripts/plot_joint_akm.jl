# Averaging-kernel matrix A = ∂x̂/∂x for the joint T + H₂O + O₃ retrieval.
#
# The joint state (55 elements) is heterogeneous:
#   rows  1–50 : temperature profile T[1..50]     (50 levels, 1013 → 2.5e-5 hPa)
#   rows 51–53 : H₂O partial-column log-scales     (levels 1:5, 6:13, 14:50)
#   row   54   : O₃ column log-scale
#   row   55   : T_sfc
# A[i,j] = ∂x̂_i/∂x_j : how the retrieved element i responds to a unit change in the
# true element j. Diagonal ≈ how much of that element the measurement (vs the prior)
# constrains; tr(A) = DOF for signal.
#
# Produces data/joint_akm.png with two panels:
#   (L) the full 55×55 A as a diverging heatmap, blocks separated + labeled
#   (R) the 50×50 temperature sub-block as classic averaging-kernel ROWS vs pressure
#
#   julia -t auto --project=. scripts/plot_joint_akm.jl

using IRSounderLBL
using JLD2: load
using Printf
using Plots
gr()

J   = load("data/iasi_joint.jld2")
rJ  = J["rJ"]; spec = rJ.spec
A   = rJ.A
lab = state_labels(spec)
n   = size(A, 1)

base = afgl_atmosphere(:us_standard)
p    = base.pressure                     # 50 levels, hPa (level 1 = surface)
nT   = spec.n_levels                     # 50

# Block boundaries in state order: T(1:50) | H₂O(51:53) | O₃(54) | Tsfc(55)
bnds  = [nT + 0.5, nT + 3.5, nT + 4.5]   # separators between blocks
gname = ["T (50 lvl)", "H₂O×3", "O₃", "Tₛ"]
gpos  = [nT/2, nT + 2, nT + 4, nT + 5]

# ── Panel L: full averaging-kernel heatmap ─────────────────────────────────────────
m = maximum(abs, A)
hL = heatmap(1:n, 1:n, A;
             c = :balance, clims = (-m, m), yflip = true, aspect_ratio = :equal,
             xlims = (0.5, n + 0.5), ylims = (0.5, n + 0.5),
             colorbar_title = "  ∂x̂ᵢ/∂xⱼ", framestyle = :box,
             xlabel = "input  (true state xⱼ)", ylabel = "output  (retrieved x̂ᵢ)",
             title = @sprintf("Joint AKM   tr(A)=DOF=%.2f,  H=%.1f bits", rJ.dof, rJ.H))
# block separators + group labels on both axes
for b in bnds
    plot!(hL, [0.5, n + 0.5], [b, b]; c = :black, lw = 0.7, label = "")
    plot!(hL, [b, b], [0.5, n + 0.5]; c = :black, lw = 0.7, label = "")
end
tickpos = Float64[]; tickstr = String[]
for (lp, lab_) in ((1000.0,"1000"),(100.0,"100"),(10.0,"10"),(1.0,"1"),(0.1,"0.1"))
    i = argmin(abs.(p .- lp)); push!(tickpos, i); push!(tickstr, lab_)
end
append!(tickpos, [51, 52, 53, 54, 55])
append!(tickstr, ["H2O₁","H2O₂","H2O₃","O₃","Tₛ"])
plot!(hL; xticks = (tickpos, tickstr), yticks = (tickpos, tickstr),
      xrotation = 60, tickfontsize = 6)
for (gp, gn) in zip(gpos, gname)
    annotate!(hL, gp, -1.5, text(gn, 7, :black, :center))
end

# ── Panel R: temperature averaging-kernel rows vs pressure ─────────────────────────
A_TT   = A[1:nT, 1:nT]
dofT   = sum(diag_ -> diag_, [A_TT[i, i] for i in 1:nT])
# color each row by its target-level pressure (log scale)
lp     = log10.(p)
cnorm(x) = (x - minimum(lp)) / (maximum(lp) - minimum(lp))
hR = plot(; yaxis = (:log10, :flip), ylim = (minimum(p), maximum(p)),
          xlabel = "kernel value  Aᵢⱼ (∂T̂ᵢ/∂Tⱼ)", ylabel = "pressure (hPa)",
          title = @sprintf("Temperature AK rows   tr(A_TT)=%.2f", dofT),
          framestyle = :box, legend = false)
for i in 1:nT
    plot!(hR, A_TT[i, :], p; lw = 1.0, lc = cgrad(:viridis)[cnorm(lp[i])], alpha = 0.8)
end
vline!(hR, [0.0]; c = :gray, lw = 0.5, ls = :dash)

plt = plot(hL, hR; layout = (1, 2), size = (1250, 620),
           left_margin = 6Plots.mm, bottom_margin = 8Plots.mm, top_margin = 4Plots.mm)
savefig(plt, "data/joint_akm.png")
println("wrote data/joint_akm.png")

# ── Numbers to accompany the figure ────────────────────────────────────────────────
println("\nDegrees of freedom for signal (diagonal sums of A):")
@printf("  temperature (T[1:50]) : %.2f\n", sum(A[i,i] for i in 1:nT))
@printf("  H₂O   (3 partial cols): %.2f   diag=%s\n",
        sum(A[i,i] for i in 51:53), string(round.([A[i,i] for i in 51:53]; digits=3)))
@printf("  O₃    (column scale)  : %.3f\n", A[54,54])
@printf("  T_sfc                 : %.3f\n", A[55,55])
@printf("  TOTAL tr(A)           : %.2f  (matches rJ.dof=%.2f)\n", sum(A[i,i] for i in 1:n), rJ.dof)
