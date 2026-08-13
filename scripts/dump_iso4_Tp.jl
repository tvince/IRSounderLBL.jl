# Dump the retrieved T(p) profiles for the iso-4 experiment from the saved results
# (no retrieval re-run): A = Voigt iso1-3, B = VP_Y lm=5 iso1-3, C = VP_Y lm=5 iso1-4.
# Writes data/iasi_profile_lm_iso4_Tp.csv for plot_iasi_profile_lm_iso4.py.
#
# Run:  julia --project=. scripts/dump_iso4_Tp.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: load

rA = load("data/iasi_profile_retrieval.jld2")["result"]          # Voigt, iso 1-3
rB = load("data/iasi_profile_lm_experiment_lmc5.jld2")["rC"]     # VP_Y lm=5, iso 1-3
rC = load("data/iasi_profile_lm_iso4.jld2")["rC"]                # VP_Y lm=5, iso 1-4

base = afgl_us_standard_50lev()
spec = rC.spec
tr, it = spec.temp_range, spec.tsfc_index

σA = sqrt.(diag(rA.S_hat)); σB = sqrt.(diag(rB.S_hat)); σC = sqrt.(diag(rC.S_hat))

open("data/iasi_profile_lm_iso4_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_voigt_K,T_iso3_K,T_iso4_K,sig_voigt_K,sig_iso3_K,sig_iso4_K")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], rA.x[lvl], rB.x[lvl], rC.x[lvl], σA[lvl], σB[lvl], σC[lvl])
    end
end
@printf("wrote data/iasi_profile_lm_iso4_Tp.csv\n")
@printf("  T_sfc: Voigt %.2f  iso1-3 %.2f  iso1-4 %.2f K\n", rA.x[it], rB.x[it], rC.x[it])
