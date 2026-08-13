# VP_Y line-mixing retrieval with the NEW lm_cutoff fix (lm_cutoff=5 cm⁻¹).
#
# Sibling of retrieve_iasi_profile_lm_experiment.jl. The 692.75 cm⁻¹ residual spike
# was re-diagnosed (2026-07-03) as the iso-1 ν₂ FUNDAMENTAL R-branch inter-line trough
# being over-brightened by far-wing dispersive line-mixing tails evaluated out to the
# full 25 cm⁻¹ Voigt cutoff (84% of the effect from lines 20–40 cm⁻¹ away). The fix is
# a separate `lm_cutoff` kwarg confining the LM dispersive term to the near-resonant
# window; 5 cm⁻¹ preserves 100% of the 667/721 Q-branch mixing while killing the spike.
#
# This drives a full VP_Y retrieval at lm_cutoff=5 and does a 3-way compare against the
# saved Voigt baseline (A) and the earlier VP_Y run at lm_cutoff=25 (B). Distinct
# `_lmc5` outputs; does NOT overwrite the lm=25 VP_Y result.
#
# Run:  julia -t auto --project=. scripts/retrieve_iasi_profile_lm_experiment_lmc5.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const BASELINE = "data/iasi_profile_retrieval.jld2"
const VPY25_JLD2 = "data/iasi_profile_lm_experiment.jld2"   # earlier VP_Y, lm_cutoff=25
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const QBRANCHES = ((667.0, "667 Q (ν₂ fundamental)", 665.0, 669.0),
                   (721.0, "721 Q",                  719.0, 723.0))
const ν_SPIKE = 692.75

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Load Voigt baseline (A) + VP_Y lm=25 (B) + exact retrieval inputs ────────────
prog("Loading Voigt baseline $BASELINE …")
bl = load(BASELINE)
rA   = bl["result"]
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]; Sa = bl["Sa"]; xa = bl["xa"]
@printf("  baseline: %d ch, %d-state, χ²=%.1f, %d iters\n",
        length(y), length(xa), rA.chi2, rA.n_iter); flush(stdout)

prog("Loading VP_Y (lm_cutoff=25) result $VPY25_JLD2 …")
rB = load(VPY25_JLD2)["rB"]
@printf("  VP_Y lm=25: χ²=%.1f, %d iters\n", rB.chi2, rB.n_iter); flush(stdout)

# ── Rebuild the forward-model context ────────────────────────────────────────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

prog("Loading CO₂/H₂O linelists …")
co2 = load_linelist("data/co2_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm_vpy = VPYLineMixing(relmat)
@printf("  relmat: %d bands\n", length(relmat.bands)); flush(stdout)

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
fm_vpy = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
          apply_continuum=true, internal_dnu=0.0025, line_mixing=lm_vpy, lm_cutoff=LM_CUTOFF)

# ── Residual diagnostics ────────────────────────────────────────────────────────
inwin(ν, lo, hi) = (ν .>= lo) .& (ν .<= hi)
rms(v) = sqrt(sum(abs2, v) / length(v))

function report(tag, r)
    res = r.y_fit .- y
    @printf("\n===== %s =====\n", tag)
    @printf("converged=%s iters=%d  χ²=%.1f  χ²/n=%.2f  DOF=%.2f  H=%.2f bits\n",
            r.converged, r.n_iter, r.chi2, r.chi2/length(y), r.dof, r.H)
    @printf("spectral fit: max|F−y|=%.3f K  rms=%.4f K\n", maximum(abs.(res)), rms(res))
    js = argmin(abs.(νobs .- ν_SPIKE))
    @printf("  692.75 R-trough spike residual: %+.3f K\n", res[js])
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
    return (res=res, rms=rms(res), qrms=sqrt(qtot/qn), χ2=r.chi2,
            χ2q_share=χ2q/χ2all, spike=res[js])
end

dA = report("A: LM OFF (Voigt)", rA)
dB = report("B: VP_Y lm_cutoff=25", rB)

# ── (C) VP_Y with lm_cutoff=5 — the only new compute ─────────────────────────────
prog("Retrieval C: VP_Y, lm_cutoff=$LM_CUTOFF …")
@time rC = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm_vpy, verbose=true)
dC = report("C: VP_Y lm_cutoff=$LM_CUTOFF", rC)

# ── Comparison ──────────────────────────────────────────────────────────────────
@printf("\n########## Voigt → VP_Y(lm=25) → VP_Y(lm=5) ##########\n")
@printf("692.75 spike residual:  %+.3f → %+.3f → %+.3f K\n", dA.spike, dB.spike, dC.spike)
@printf("full-band residual rms:  %.4f → %.4f → %.4f K\n", dA.rms, dB.rms, dC.rms)
@printf("Q-branch residual rms :  %.4f → %.4f → %.4f K\n", dA.qrms, dB.qrms, dC.qrms)
@printf("total χ²              :  %.1f → %.1f → %.1f\n", dA.χ2, dB.χ2, dC.χ2)
@printf("Q-branch χ² share     :  %.1f%% → %.1f%% → %.1f%%\n",
        100dA.χ2q_share, 100dB.χ2q_share, 100dC.χ2q_share)

# ── Dump (3-way) ────────────────────────────────────────────────────────────────
σA = sqrt.(diag(rA.S_hat)); σB = sqrt.(diag(rB.S_hat)); σC = sqrt.(diag(rC.S_hat))
nedt = sqrt.(diag(Se))
tr = spec.temp_range
open("data/iasi_profile_lm_experiment_lmc5_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_voigt_K,res_vpy25_K,res_vpy5_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νobs[i], y[i], dA.res[i], dB.res[i], dC.res[i], nedt[i])
    end
end
open("data/iasi_profile_lm_experiment_lmc5_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_voigt_K,T_vpy25_K,T_vpy5_K,sig_voigt_K,sig_vpy25_K,sig_vpy5_K")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], rA.x[lvl], rB.x[lvl], rC.x[lvl], σA[lvl], σB[lvl], σC[lvl])
    end
end
jldsave("data/iasi_profile_lm_experiment_lmc5.jld2"; rC=rC, ν=νobs, y=y)
prog("wrote data/iasi_profile_lm_experiment_lmc5_{fit,Tp}.csv + .jld2")
