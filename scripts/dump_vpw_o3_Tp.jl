# Dump retrieved T(p) for the VP_W+O3 experiment from saved results (no re-run):
# A = VP_Y lm=5 no-O3, B = VP_Y lm=5 +O3, C = VP_W lm=5 +O3.
# Writes data/iasi_profile_vpw_o3_Tp.csv for plot_iasi_profile_vpw_o3_Tp.py.
#
# Run:  julia --project=. scripts/dump_vpw_o3_Tp.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: load

rA = load("data/iasi_profile_lm_iso4.jld2")["rC"]   # VP_Y no-O3
rB = load("data/iasi_profile_o3.jld2")["rC"]        # VP_Y +O3
rC = load("data/iasi_profile_vpw_o3.jld2")["rC"]    # VP_W +O3

base = afgl_us_standard_50lev()
spec = rC.spec
tr, it = spec.temp_range, spec.tsfc_index
σA = sqrt.(diag(rA.S_hat)); σB = sqrt.(diag(rB.S_hat)); σC = sqrt.(diag(rC.S_hat))

open("data/iasi_profile_vpw_o3_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_vpy_noO3_K,T_vpy_O3_K,T_vpw_O3_K,sig_vpy_noO3_K,sig_vpy_O3_K,sig_vpw_O3_K")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], rA.x[lvl], rB.x[lvl], rC.x[lvl], σA[lvl], σB[lvl], σC[lvl])
    end
end
@printf("wrote data/iasi_profile_vpw_o3_Tp.csv\n")
@printf("  T_sfc: VP_Y no-O3 %.2f  VP_Y +O3 %.2f  VP_W +O3 %.2f K\n", rA.x[it], rB.x[it], rC.x[it])
