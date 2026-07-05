# VP_W (full-matrix line-mixing) retrieval WITH O3, lm_cutoff=5. The 710-730 positive
# residual is a superposition of missing O3 R-branch absorption + VP_Y first-order
# OVER-THINNING of the CO2 inter-line minima flanking the 721 Q-branch. Full-matrix
# VP_W conserves the W-matrix sum rule and should NOT over-thin the wings, so O3 alone
# shouldn't have to over-correct. Does VP_W + O3 clean up 720 without the overshoot?
#
# 3-way: A = VP_Y lm=5, NO O3 (data/iasi_profile_lm_iso4.jld2 rC) ·
#        B = VP_Y lm=5, + O3 (data/iasi_profile_o3.jld2 rC) ·
#        C = VP_W lm=5, + O3 (the only new compute). Same y/Se/Sa/xa, iso 1-4, O3 S>1e-23.
#
# Requires data/o3_645_2760*.par. Run:
#   julia -t auto --project=. scripts/retrieve_iasi_profile_vpw_o3.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const VPY_NOO3  = "data/iasi_profile_lm_iso4.jld2"   # VP_Y lm=5, iso1-4, NO O3 (key rC)
const VPY_O3    = "data/iasi_profile_o3.jld2"        # VP_Y lm=5, iso1-4, + O3 (key rC)
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const REGIONS = ((700.0, 710.0, "O3 ν₂ Q region"), (710.0, 730.0, "LM-overshoot region"))
const ν_PTS = (715.75, 722.75, 723.25)

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

prog("Loading retrieval inputs + prior results …")
bl = load(BASELINE)
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]; Sa = bl["Sa"]; xa = bl["xa"]
rA = load(VPY_NOO3)["rC"]      # VP_Y, no O3
rB = load(VPY_O3)["rC"]        # VP_Y, + O3
@printf("  A VP_Y no-O3 χ²=%.1f · B VP_Y +O3 χ²=%.1f\n", rA.chi2, rB.chi2); flush(stdout)

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

prog("Loading CO₂ (1–4) / H₂O (1–3) / O₃ (1–4, S>$(O3_SMIN)) linelists …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3_full = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3 = HITRANLinelist([l for l in o3_full.lines if l.intensity > O3_SMIN])
@printf("  O₃: %d/%d lines (S>%.0e)\n", length(o3.lines), length(o3_full.lines), O3_SMIN); flush(stdout)
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm_vpw = VPWLineMixing(relmat)   # default whitelist = top-5 bands over 645–800
@printf("  relmat: %d bands; VP_W whitelist = %s\n",
        length(relmat.bands), join(sort(collect(lm_vpw.whitelist)), ", ")); flush(stdout)

prog("Re-screening granule for exact FOV geometry …")
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
_, ychk = measurement(g, ifov)
@assert length(ychk) == length(y) && maximum(abs.(ychk .- y)) < 1e-6 "granule y ≠ baseline y"
@printf("  FOV #%d zen=%.4f°\n", ifov, g.zen[ifov]); flush(stdout)

nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
spec  = StateVectorSpec(nlev, GasSpecies[];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
fm_vpw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
          apply_continuum=true, internal_dnu=0.0025, line_mixing=lm_vpw, lm_cutoff=LM_CUTOFF)

inwin(ν, lo, hi) = (ν .>= lo) .& (ν .<= hi)
rms(v) = sqrt(sum(abs2, v) / length(v))
function report(tag, r)
    res = r.y_fit .- y
    @printf("\n===== %s =====\n", tag)
    @printf("converged=%s iters=%d  χ²=%.1f  χ²/n=%.2f  DOF=%.2f\n",
            r.converged, r.n_iter, r.chi2, r.chi2/length(y), r.dof)
    @printf("full-band: max|F−y|=%.3f K  rms=%.4f K\n", maximum(abs.(res)), rms(res))
    for (lo, hi, name) in REGIONS
        m = inwin(νobs, lo, hi)
        @printf("  %-22s [%.0f,%.0f]: rms=%.4f K  max|res|=%.3f K\n",
                name, lo, hi, rms(res[m]), maximum(abs.(res[m])))
    end
    for νp in ν_PTS
        j = argmin(abs.(νobs .- νp)); @printf("  ν=%.2f residual: %+.3f K\n", νp, res[j])
    end
    return (res=res, rms=rms(res), χ2=r.chi2)
end

dA = report("A: VP_Y lm=5, NO O3", rA)
dB = report("B: VP_Y lm=5, + O3", rB)

prog("Retrieval C: VP_W lm=5 + O3 …")
@time rC = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm_vpw, verbose=true)
dC = report("C: VP_W lm=5, + O3", rC)

@printf("\n########## VP_Y noO3 → VP_Y +O3 → VP_W +O3 ##########\n")
@printf("full-band rms:  %.4f → %.4f → %.4f K\n", dA.rms, dB.rms, dC.rms)
@printf("total χ²     :  %.1f → %.1f → %.1f\n", dA.χ2, dB.χ2, dC.χ2)
mlm = inwin(νobs, 710.0, 730.0)
@printf("710–730 rms  :  %.4f → %.4f → %.4f K\n",
        rms(dA.res[mlm]), rms(dB.res[mlm]), rms(dC.res[mlm]))
for νp in ν_PTS
    j = argmin(abs.(νobs .- νp))
    @printf("ν=%.2f residual:  %+.3f → %+.3f → %+.3f K\n", νp, dA.res[j], dB.res[j], dC.res[j])
end

nedt = sqrt.(diag(Se))
open("data/iasi_profile_vpw_o3_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_vpy_noO3_K,res_vpy_O3_K,res_vpw_O3_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νobs[i], y[i], dA.res[i], dB.res[i], dC.res[i], nedt[i])
    end
end
jldsave("data/iasi_profile_vpw_o3.jld2"; rC=rC, ν=νobs, y=y)
prog("wrote data/iasi_profile_vpw_o3_{fit.csv,jld2}")
