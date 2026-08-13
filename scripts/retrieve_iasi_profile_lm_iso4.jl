# VP_Y retrieval INCLUDING CO2 isotopologue 4 (627) — does adding iso-4 collapse the
# 665 cm⁻¹ residual spike?
#
# The 665.0 cm⁻¹ trough (ν₂ Q-branch bandhead) is a −8 K residual present already in the
# Voigt baseline (model too COLD / over-absorbs). Prior diagnosis said iso-4 line-mixing
# is tiny (~0.02 K) and that adding iso-4 *absorption* would push 665 COLDER (worse), but
# that was measured WITHOUT an iso-4 Voigt baseline on disk. Now that data/co2_645_2760_
# iso4.par exists (34,777 lines, 620–2785) and Fix A lets the iso-4 relmat bands act
# consistently, this runs the real test: VP_Y lm_cutoff=5 with iso 1–4 vs the iso 1–3 run.
#
# 3-way: A = Voigt (iso 1–3, saved baseline) · B = VP_Y lm=5 (iso 1–3, saved) ·
#        C = VP_Y lm=5 (iso 1–4, the only new compute). Same y / Sₑ / Sₐ / xₐ / lm_cutoff.
#
# Run:  julia -t auto --project=. scripts/retrieve_iasi_profile_lm_iso4.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const VPY5_JLD2 = "data/iasi_profile_lm_experiment_lmc5.jld2"   # VP_Y lm=5, iso 1–3 (key rC)
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

# The spikes of interest: 665.0 = ν₂ Q-branch bandhead (this experiment); 692.75 = the
# R-branch trough the lm_cutoff fix already cleared (should stay clear).
const ν_665, ν_693 = 665.0, 692.75
const QBRANCHES = ((667.0, "667 Q (ν₂ fundamental)", 664.5, 669.0),
                   (721.0, "721 Q",                  719.0, 723.0))

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Load Voigt baseline (A) + VP_Y lm=5 iso1–3 (B) + exact retrieval inputs ───────
prog("Loading Voigt baseline $BASELINE …")
bl = load(BASELINE)
rA   = bl["result"]
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]; Sa = bl["Sa"]; xa = bl["xa"]
@printf("  baseline: %d ch, %d-state, χ²=%.1f, %d iters\n",
        length(y), length(xa), rA.chi2, rA.n_iter); flush(stdout)

prog("Loading VP_Y lm=5 iso1–3 result $VPY5_JLD2 …")
rB = load(VPY5_JLD2)["rC"]
@printf("  VP_Y lm=5 (iso1–3): χ²=%.1f, %d iters\n", rB.chi2, rB.n_iter); flush(stdout)

# ── Rebuild the forward-model context WITH iso-4 ──────────────────────────────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

prog("Loading CO₂ (iso 1–4) / H₂O (iso 1–3) linelists …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
co2_isos = sort(collect(Set(Int(l.iso_id) for l in co2.lines)))
@printf("  CO₂: %d lines, iso_ids=%s\n", length(co2.lines), join(co2_isos, ",")); flush(stdout)
@assert 4 in co2_isos "iso-4 not present in the loaded CO₂ linelist"
linelists = Dict(CO2 => co2, H2O => h2o)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm_vpy = VPYLineMixing(relmat)
niso4_bands = count(b -> Int(b.isot) == 4, relmat.bands)
@printf("  relmat: %d bands (%d iso-4)\n", length(relmat.bands), niso4_bands); flush(stdout)

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

# ── Residual diagnostics ──────────────────────────────────────────────────────────
inwin(ν, lo, hi) = (ν .>= lo) .& (ν .<= hi)
rms(v) = sqrt(sum(abs2, v) / length(v))

function report(tag, r)
    res = r.y_fit .- y
    @printf("\n===== %s =====\n", tag)
    @printf("converged=%s iters=%d  χ²=%.1f  χ²/n=%.2f  DOF=%.2f  H=%.2f bits\n",
            r.converged, r.n_iter, r.chi2, r.chi2/length(y), r.dof, r.H)
    @printf("spectral fit: max|F−y|=%.3f K  rms=%.4f K\n", maximum(abs.(res)), rms(res))
    j665 = argmin(abs.(νobs .- ν_665)); j693 = argmin(abs.(νobs .- ν_693))
    @printf("  665.0  Q-bandhead spike residual: %+.3f K\n", res[j665])
    @printf("  692.75 R-trough    spike residual: %+.3f K\n", res[j693])
    for (νc, name, lo, hi) in QBRANCHES
        m = inwin(νobs, lo, hi)
        @printf("  Q-branch %-24s [%.1f,%.1f]: rms=%.4f K  max|res|=%.3f K  (%d ch)\n",
                name, lo, hi, rms(res[m]), maximum(abs.(res[m])), count(m))
    end
    return (res=res, rms=rms(res), χ2=r.chi2, s665=res[j665], s693=res[j693])
end

dA = report("A: LM OFF (Voigt, iso 1–3)", rA)
dB = report("B: VP_Y lm=5 (iso 1–3)", rB)

# ── (C) VP_Y lm=5 WITH iso-4 — the only new compute ───────────────────────────────
prog("Retrieval C: VP_Y lm_cutoff=$LM_CUTOFF, iso 1–4 …")
@time rC = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm_vpy, verbose=true)
dC = report("C: VP_Y lm=5 (iso 1–4)", rC)

# ── Comparison ────────────────────────────────────────────────────────────────────
@printf("\n########## Voigt(1–3) → VP_Y lm=5 (1–3) → VP_Y lm=5 (1–4) ##########\n")
@printf("665.0  spike residual:  %+.3f → %+.3f → %+.3f K   (iso-4 Δ = %+.4f K)\n",
        dA.s665, dB.s665, dC.s665, dC.s665 - dB.s665)
@printf("692.75 spike residual:  %+.3f → %+.3f → %+.3f K\n", dA.s693, dB.s693, dC.s693)
@printf("full-band residual rms:  %.4f → %.4f → %.4f K\n", dA.rms, dB.rms, dC.rms)
@printf("total χ²              :  %.1f → %.1f → %.1f\n", dA.χ2, dB.χ2, dC.χ2)

# per-channel iso-4 effect in the 664.5–666.0 bandhead
mb = inwin(νobs, 664.5, 666.0)
@printf("\niso-4 per-channel Δres (C−B) over 664.5–666.0:\n")
for i in findall(mb)
    @printf("  ν=%.3f  B=%+.3f  C=%+.3f  Δ=%+.4f K\n", νobs[i], dB.res[i], dC.res[i], dC.res[i]-dB.res[i])
end
@printf("  max |Δres| in bandhead = %.4f K\n", maximum(abs.(dC.res[mb] .- dB.res[mb])))

# ── Dump ──────────────────────────────────────────────────────────────────────────
nedt = sqrt.(diag(Se))
open("data/iasi_profile_lm_iso4_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_voigt_K,res_vpy5_iso3_K,res_vpy5_iso4_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νobs[i], y[i], dA.res[i], dB.res[i], dC.res[i], nedt[i])
    end
end
jldsave("data/iasi_profile_lm_iso4.jld2"; rC=rC, ν=νobs, y=y)
prog("wrote data/iasi_profile_lm_iso4_fit.csv + .jld2")
