# Averaging-kernel row for the O₃ column scale (state element 54) in the joint retrieval.
#
# O₃ is retrieved as a single ColumnScale, so its averaging kernel is ONE row:
#   A[54, :] = ∂(retrieved O₃ column) / ∂(each true state element)
# The self term A[54,54] says how much of the retrieved column comes from the true
# O₃ column; the other entries are interference — how a true error in temperature at
# each level, in H₂O, or in T_sfc leaks into the retrieved O₃ column.
#
# Produces data/o3_akm_row.png:
#   (L) temperature cross-kernel A[54,1:50] vs pressure (T→O₃ leakage per level)
#   (R) the scalar entries (H₂O×3, O₃ self, T_sfc) as labeled bars
#
#   julia --project=. scripts/plot_o3_akm_row.jl

using IRSounderLBL
using JLD2: load
using Printf
using Plots
gr()

J   = load("data/iasi_joint.jld2")
rJ  = J["rJ"]; spec = rJ.spec
A   = rJ.A
nT  = spec.n_levels                      # 50
iO3 = 54
row = A[iO3, :]

base = afgl_atmosphere(:us_standard)
p    = base.pressure                     # hPa, level 1 = surface

# ── Panel L: temperature → O₃ leakage vs pressure ──────────────────────────────────
hL = plot(row[1:nT], p;
          yaxis = (:log10, :flip), ylim = (minimum(p), maximum(p)),
          lw = 2, lc = :darkorange, marker = (:circle, 2.5), legend = false,
          xlabel = "∂(O₃ column) / ∂T[level]   (per K)", ylabel = "pressure (hPa)",
          title = "T → O₃ interference kernel", framestyle = :box)
vline!(hL, [0.0]; c = :gray, lw = 0.6, ls = :dash)

# ── Panel R: scalar entries of the O₃ row ──────────────────────────────────────────
sidx = [51, 52, 53, 54, 55]
snam = ["H₂O₁\n(1:5)", "H₂O₂\n(6:13)", "H₂O₃\n(14:50)", "O₃\n(self)", "T_sfc"]
svals = row[sidx]
cols  = [i == iO3 ? :seagreen : :steelblue for i in sidx]
hR = bar(snam, svals; legend = false, color = cols,
         xlabel = "state element", ylabel = "∂(O₃ column) / ∂xⱼ",
         title = @sprintf("O₃-row entries   self A₅₄,₅₄=%.3f", row[iO3]),
         framestyle = :box, bar_width = 0.6)
hline!(hR, [0.0]; c = :black, lw = 0.6)
for (k, v) in zip(1:length(sidx), svals)
    annotate!(hR, k, v + sign(v) * 0.04 + (v ≈ 0 ? 0.04 : 0),
              text(@sprintf("%.3f", v), 8, :center))
end

plt = plot(hL, hR; layout = (1, 2), size = (1150, 560),
           left_margin = 6Plots.mm, bottom_margin = 8Plots.mm)
savefig(plt, "data/o3_akm_row.png")
println("wrote data/o3_akm_row.png")

@printf("\nO₃ column averaging-kernel row (element 54):\n")
@printf("  self  A[54,54]        = %.4f   (fraction of retrieved column from true O₃)\n", row[iO3])
@printf("  Σ|T leakage| (1:50)   = %.4f   max @ %.1f hPa = %+.4f /K\n",
        sum(abs, row[1:nT]), p[argmax(abs.(row[1:nT]))], row[argmax(abs.(row[1:nT]))])
@printf("  H₂O entries           = %s\n", string(round.(row[51:53]; digits=4)))
@printf("  T_sfc  A[54,55]       = %+.4f\n", row[55])
