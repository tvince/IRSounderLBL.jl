# VP_Y lm=5 retrieval WITH O3 added to the forward model — does modelling the O3 ν₂
# band (currently absent; CO2+H2O only) change the 645–800 fit, and specifically the
# 710–730 residual structure (shown to be VP_Y line-mixing overshoot, not O3)?
#
# O3 enters as a FIXED absorber at AFGL US-standard climatological VMR (T-only
# retrieval, same treatment as CO2/H2O). 2-way compare: B = VP_Y lm=5 iso1–4, NO O3
# (saved rC from retrieve_iasi_profile_lm_iso4.jl) vs C = same + O3 (the new compute).
#
# Requires data/o3_645_2760*.par (run scripts/fetch_o3.jl first).
# Run:  julia -t auto --project=. scripts/retrieve_iasi_profile_o3.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const NOO3_JLD2 = "data/iasi_profile_lm_iso4.jld2"    # VP_Y lm=5, iso1–4, NO O3 (key rC)
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

# diagnostic windows: O3 ν₂ Q-branch, the LM-overshoot region, and two LM spike points
const REGIONS = ((700.0, 710.0, "O3 ν₂ Q region"),
                 (710.0, 730.0, "LM-overshoot region"))
const ν_PTS = (715.75, 722.75)   # the +4 K / +2.6 K LM-overshoot channels

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Load retrieval inputs + the NO-O3 comparison (B) ──────────────────────────────
prog("Loading Voigt baseline $BASELINE …")
bl = load(BASELINE)
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]; Sa = bl["Sa"]; xa = bl["xa"]
@printf("  %d ch, %d-state\n", length(y), length(xa)); flush(stdout)

prog("Loading NO-O3 VP_Y lm=5 iso1–4 result $NOO3_JLD2 …")
rB = load(NOO3_JLD2)["rC"]
@printf("  no-O3: χ²=%.1f, %d iters\n", rB.chi2, rB.n_iter); flush(stdout)

# ── Forward-model context WITH O3 ─────────────────────────────────────────────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

# O3 is a fixed minor absorber; the retrieval forward runs dptmn=0 (no per-layer
# line rejection, for F/K consistency), so pre-trim the ~29k O3 lines by a static
# intensity floor applied IDENTICALLY to F and K. S>1e-23 keeps 4064 lines (7×
# fewer) and is BT-lossless to max|ΔBT|=0.024 K (validated: scratchpad o3_trim2.jl).
const O3_SMIN = 1e-23
prog("Loading CO₂ (1–4) / H₂O (1–3) / O₃ (1–4) linelists …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3_full = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3 = HITRANLinelist([l for l in o3_full.lines if l.intensity > O3_SMIN])
o3_isos = sort(collect(Set(Int(l.iso_id) for l in o3.lines)))
@printf("  O₃: %d/%d lines (S>%.0e) in %.0f–%.0f, iso_ids=%s (AFGL VMR: sfc %.3g, peak %.3g ppm)\n",
        length(o3.lines), length(o3_full.lines), O3_SMIN, ν_LO-25, ν_HI+25, join(o3_isos, ","),
        base.vmr[O3][1]*1e6, maximum(base.vmr[O3])*1e6); flush(stdout)
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm_vpy = VPYLineMixing(relmat)

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
fm = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
      apply_continuum=true, internal_dnu=0.0025, line_mixing=lm_vpy, lm_cutoff=LM_CUTOFF)

# ── Diagnostics ───────────────────────────────────────────────────────────────────
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

dB = report("B: VP_Y lm=5, iso 1–4, NO O3", rB)

# ── (C) VP_Y lm=5 WITH O3 — the only new compute ──────────────────────────────────
prog("Retrieval C: VP_Y lm=5 + O3 …")
@time rC = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)
dC = report("C: VP_Y lm=5, iso 1–4, + O3", rC)

@printf("\n########## NO O3 → + O3 (both VP_Y lm=5, iso 1–4) ##########\n")
@printf("full-band rms:  %.4f → %.4f K\n", dB.rms, dC.rms)
@printf("total χ²     :  %.1f → %.1f\n", dB.χ2, dC.χ2)
mreg = inwin(νobs, 700.0, 710.0)
@printf("O3 ν₂ region rms (700–710):  %.4f → %.4f K\n", rms(dB.res[mreg]), rms(dC.res[mreg]))
mlm = inwin(νobs, 710.0, 730.0)
@printf("LM region rms (710–730)   :  %.4f → %.4f K  (O3 should barely move this)\n",
        rms(dB.res[mlm]), rms(dC.res[mlm]))
@printf("\nlargest |O3 effect| channels (|res_C − res_B|):\n")
d = abs.(dC.res .- dB.res)
for i in sort(sortperm(d; rev=true)[1:10])
    @printf("  ν=%.3f  noO3=%+.3f  O3=%+.3f  Δ=%+.4f K\n", νobs[i], dB.res[i], dC.res[i], dC.res[i]-dB.res[i])
end

# ── Dump ──────────────────────────────────────────────────────────────────────────
nedt = sqrt.(diag(Se))
open("data/iasi_profile_o3_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_noO3_K,res_O3_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], dB.res[i], dC.res[i], nedt[i])
    end
end
jldsave("data/iasi_profile_o3.jld2"; rC=rC, ν=νobs, y=y)
prog("wrote data/iasi_profile_o3_fit.csv + .jld2")
