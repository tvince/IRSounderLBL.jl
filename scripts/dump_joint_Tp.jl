# Extract retrieved T(p) + posterior σ from the joint retrievals (no re-run):
#   prior = VP_Y lm=5 + fixed O3, T-only (data/iasi_profile_o3.jld2 key rC)
#   VP_Y joint (data/iasi_joint.jld2 rJ) · VP_W joint (data/iasi_joint_vpw.jld2 rJ)
# First guess = AFGL US-standard. Writes data/iasi_joint_Tp.csv.
#   julia --project=. scripts/dump_joint_Tp.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: load

rPrior = load("data/iasi_profile_o3.jld2")["rC"]   # T-only, fixed O3
rVY    = load("data/iasi_joint.jld2")["rJ"]         # VP_Y joint
rVW    = load("data/iasi_joint_vpw.jld2")["rJ"]     # VP_W joint

base = afgl_us_standard_50lev()
p    = base.pressure
Tfg  = base.temperature                              # first guess

Tp(r) = (tr = r.spec.temp_range; it = r.spec.tsfc_index;
         (T = collect(Float64, r.x[tr]), σ = sqrt.(diag(r.S_hat))[tr],
          Tsfc = Float64(r.x[it]), σsfc = sqrt(r.S_hat[it, it])))

pr, vy, vw = Tp(rPrior), Tp(rVY), Tp(rVW)

open("data/iasi_joint_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_fg_K,T_prior_K,sig_prior_K,T_vpy_K,sig_vpy_K,T_vpw_K,sig_vpw_K")
    for i in eachindex(p)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, p[i], Tfg[i], pr.T[i], pr.σ[i], vy.T[i], vy.σ[i], vw.T[i], vw.σ[i])
    end
end
@printf("wrote data/iasi_joint_Tp.csv\n")
@printf("T_sfc:  first-guess %.2f  ·  prior %.2f±%.2f  ·  VP_Y %.2f±%.2f  ·  VP_W %.2f±%.2f K\n",
        Tfg[1], pr.Tsfc, pr.σsfc, vy.Tsfc, vy.σsfc, vw.Tsfc, vw.σsfc)
