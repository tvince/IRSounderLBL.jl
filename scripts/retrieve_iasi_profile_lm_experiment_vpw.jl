# VP_W (full-matrix line-mixing) EXPERIMENT on the real-IASI T(p) retrieval.
#
# Companion to retrieve_iasi_profile_lm_experiment.jl (VP_Y). Loads BOTH already-
# computed cases — A = plain Voigt (data/iasi_profile_retrieval.jld2) and
# B = VP_Y (data/iasi_profile_lm_experiment.jld2) — and runs ONLY the new case
# C = VP_W on the SAME observed y / Sₑ / Sₐ / xₐ. Compares χ², spectral fit, and
# the CO₂ Q-branch residuals (667 & 721), where full-matrix LM is expected to beat
# first-order VP_Y. Whitelist = default top-5 bands (S(T_REF)·abundance) over
# 645–800; non-whitelisted bands fall back to the VP_Y dispersive term.
#
# Run:  julia -t auto --project=. scripts/retrieve_iasi_profile_lm_experiment_vpw.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const BASELINE = "data/iasi_profile_retrieval.jld2"
const VPY_JLD2 = "data/iasi_profile_lm_experiment.jld2"
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const QBRANCHES = ((667.0, "667 Q (ν₂ fundamental)", 665.0, 669.0),
                   (721.0, "721 Q",                  719.0, 723.0))

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Load the Voigt baseline (A) + VP_Y (B) + the exact retrieval inputs ──────────
prog("Loading Voigt baseline $BASELINE …")
bl = load(BASELINE)
rA   = bl["result"]
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]; Sa = bl["Sa"]; xa = bl["xa"]
@printf("  baseline: %d ch, %d-state, χ²=%.1f, %d iters\n",
        length(y), length(xa), rA.chi2, rA.n_iter); flush(stdout)

prog("Loading VP_Y result $VPY_JLD2 …")
rB = load(VPY_JLD2)["rB"]
@printf("  VP_Y: χ²=%.1f, %d iters\n", rB.chi2, rB.n_iter); flush(stdout)

# ── Rebuild the (deterministic) forward-model context for the VP_W run ───────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

prog("Loading CO₂/H₂O linelists …")
co2 = load_linelist("data/co2_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm_vpw = VPWLineMixing(relmat)   # default whitelist = top-5 bands over the loaded range
@printf("  relmat: %d bands; VP_W whitelist = %s\n",
        length(relmat.bands), join(sort(collect(lm_vpw.whitelist)), ", ")); flush(stdout)

prog("Re-screening granule for exact FOV geometry …")
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
_, ychk = measurement(g, ifov)
@assert length(ychk) == length(y) && maximum(abs.(ychk .- y)) < 1e-6 "granule y ≠ baseline y"
@printf("  FOV #%d zen=%.4f° (baseline y matches to <1e-6 K)\n", ifov, g.zen[ifov]); flush(stdout)

nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
spec  = StateVectorSpec(nlev, GasSpecies[];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
fm_vpw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
          apply_continuum=true, internal_dnu=0.0025, line_mixing=lm_vpw)

# ── Residual diagnostics ────────────────────────────────────────────────────────
inwin(ν, lo, hi) = (ν .>= lo) .& (ν .<= hi)
rms(v) = sqrt(sum(abs2, v) / length(v))

function report(tag, r)
    res = r.y_fit .- y
    @printf("\n===== %s =====\n", tag)
    @printf("converged=%s iters=%d  χ²=%.1f  χ²/n=%.2f  DOF=%.2f  H=%.2f bits\n",
            r.converged, r.n_iter, r.chi2, r.chi2/length(y), r.dof, r.H)
    @printf("spectral fit: max|F−y|=%.3f K  rms=%.4f K\n", maximum(abs.(res)), rms(res))
    qtot = 0.0; qn = 0
    for (νc, name, lo, hi) in QBRANCHES
        m = inwin(νobs, lo, hi)
        @printf("  Q-branch %-24s [%.0f,%.0f]: rms=%.4f K  max|res|=%.3f K  (%d ch)\n",
                name, lo, hi, rms(res[m]), maximum(abs.(res[m])), count(m))
        qtot += sum(abs2, res[m]); qn += count(m)
    end
    σ = sqrt.(diag(Se))
    qmask = falses(length(νobs))
    for (_, _, lo, hi) in QBRANCHES; qmask .|= inwin(νobs, lo, hi); end
    χ2q = sum(abs2, (res./σ)[qmask]); χ2all = sum(abs2, res./σ)
    @printf("  Q-branch χ² share = %.1f%% (%d of %d ch)\n", 100χ2q/χ2all, count(qmask), length(νobs))
    return (res=res, rms=rms(res), qrms=sqrt(qtot/qn), χ2=r.chi2, χ2q_share=χ2q/χ2all)
end

dA = report("A: LM OFF (Voigt, from baseline JLD2)", rA)
dB = report("B: VP_Y (from JLD2)", rB)

# ── (C) VP_W — the only new compute ──────────────────────────────────────────────
prog("Retrieval C: line_mixing = VPWLineMixing(relmat) …")
@time rC = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm_vpw, verbose=true)
dC = report("C: VP_W", rC)

# ── Comparison ──────────────────────────────────────────────────────────────────
@printf("\n########## Voigt → VP_Y → VP_W ##########\n")
@printf("full-band residual rms:  %.4f → %.4f → %.4f K\n", dA.rms, dB.rms, dC.rms)
@printf("Q-branch residual rms :  %.4f → %.4f → %.4f K\n", dA.qrms, dB.qrms, dC.qrms)
@printf("total χ²              :  %.1f → %.1f → %.1f\n", dA.χ2, dB.χ2, dC.χ2)
@printf("Q-branch χ² share     :  %.1f%% → %.1f%% → %.1f%%\n",
        100dA.χ2q_share, 100dB.χ2q_share, 100dC.χ2q_share)

tr = spec.temp_range; it = spec.tsfc_index
@printf("\nretrieved T change (VP_W − VP_Y), levels with |Δ|>0.05 K:\n")
@printf("  T_sfc: %.2f → %.2f K (Δ%+.3f)\n", rB.x[it], rC.x[it], rC.x[it]-rB.x[it])
for (i, lvl) in enumerate(tr)
    dTlev = rC.x[lvl] - rB.x[lvl]
    abs(dTlev) > 0.05 && @printf("  lvl %3d p=%8.2f hPa:  %.2f → %.2f K  (Δ%+.3f)\n",
                                 i, base.pressure[i], rB.x[lvl], rC.x[lvl], dTlev)
end

# ── Dump (3-way) ──────────────────────────────────────────────────────────────────
σA = sqrt.(diag(rA.S_hat)); σB = sqrt.(diag(rB.S_hat)); σC = sqrt.(diag(rC.S_hat))
nedt = sqrt.(diag(Se))
open("data/iasi_profile_lm_experiment_vpw_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_voigt_K,res_vpy_K,res_vpw_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νobs[i], y[i], dA.res[i], dB.res[i], dC.res[i], nedt[i])
    end
end
open("data/iasi_profile_lm_experiment_vpw_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_voigt_K,T_vpy_K,T_vpw_K,sig_voigt_K,sig_vpy_K,sig_vpw_K")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], rA.x[lvl], rB.x[lvl], rC.x[lvl], σA[lvl], σB[lvl], σC[lvl])
    end
end
jldsave("data/iasi_profile_lm_experiment_vpw.jld2"; rC=rC, ν=νobs, y=y)
prog("wrote data/iasi_profile_lm_experiment_vpw_{fit,Tp}.csv + .jld2")
