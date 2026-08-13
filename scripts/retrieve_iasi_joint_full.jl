# FULL per-level joint retrieval (for posterity): T(50) + T_sfc + H2O(50, full
# profile) + O3(50, full profile) = 151 elements. Unlike retrieve_iasi_joint.jl (which
# uses the reduced-VMR basis: H2O DFS-matched bulk layers + O3 single column scale),
# here BOTH gases are retrieved at every level. The O3 profile is ill-conditioned in the
# 15um band (~1 DOF of real information), so Sa uses a per-level correlation length to
# regularize — the uninformed (upper) levels simply relax back to the prior. This run
# documents what the full-resolution problem does; the reduced-basis driver is the
# production result. Tropical prior (matched to this scene).
#
#   JOINT_BASE=tropical julia -t auto --project=. scripts/retrieve_iasi_joint_full.jl
using IRSounderLBL
using LinearAlgebra: diag, Symmetric
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const LM_METHOD = lowercase(get(ENV, "JOINT_LM", "vpy"))
const TAG       = LM_METHOD == "vpw" ? "_vpw" : ""
const O3_SMIN   = parse(Float64, get(ENV, "O3_SMIN", "1e-23"))
const BASE_ATM  = Symbol(get(ENV, "JOINT_BASE", "tropical"))    # matched prior default
const BASETAG   = BASE_ATM == :us_standard ? "" : "_$(BASE_ATM)"
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const GRANULE   = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const REGIONS = ((700.0, 710.0, "O3 ν₂ Q region"), (710.0, 730.0, "LM-overshoot region"))
const ν_PTS   = (715.75, 722.75, 723.25)

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Inputs (mirror retrieve_iasi_joint.jl) ─────────────────────────────────────────
prog("Loading baseline y / Se …")
bl = load(BASELINE)
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
@printf("  %d channels\n", length(y)); flush(stdout)

base = afgl_atmosphere(BASE_ATM)
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)
@printf("  base atmosphere: AFGL %s (T_sfc=%.1f K)\n", BASE_ATM, base.temperature[1]); flush(stdout)

prog("Loading CO₂(1–4) / H₂O(1–3) / O₃(1–4, S>$(O3_SMIN)) linelists …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3_full = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3 = HITRANLinelist([l for l in o3_full.lines if l.intensity > O3_SMIN])
@printf("  O₃: %d/%d lines (S>%.0e)\n", length(o3.lines), length(o3_full.lines), O3_SMIN); flush(stdout)
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = LM_METHOD == "vpw" ? VPWLineMixing(relmat) : VPYLineMixing(relmat)

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
fm = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
      apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF)

# ── Full-profile joint state: T + T_sfc + H2O(50) + O3(50) ─────────────────────────
spec = StateVectorSpec(nlev, [H2O, O3];
                       include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0,
              σ_vmr=Dict(H2O=>0.5, O3=>0.2), L_vmr=Dict(H2O=>1.0, O3=>1.0), σ_tsfc=5.0)
xa = pack_state(spec, base)
@printf("  full joint state: n=%d  (T %d + T_sfc + H₂O %d + O₃ %d), Sa L_vmr=1 lev\n",
        spec.n, nlev, nlev, nlev); flush(stdout)

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

prog("Full-profile joint retrieval ($(uppercase(LM_METHOD)): T + H₂O(50) + O₃(50)) …")
@time rF = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)
dF = report("FULL JOINT $(uppercase(LM_METHOD)): T + H₂O(50) + O₃(50)", rF)

# Per-species information content (AKM diagonal) — shows the O3 ill-conditioning.
dA = diag(rF.A)
rH = vmr_range(spec, H2O); rO = vmr_range(spec, O3)
@printf("\nDOF budget: total=%.2f  T=%.2f  T_sfc=%.2f  H₂O=%.2f  O₃=%.2f\n",
        rF.dof, sum(dA[spec.temp_range]), dA[spec.tsfc_index], sum(dA[rH]), sum(dA[rO]))

# FullProfile packs ABSOLUTE log-VMR, so exp(x) is the retrieved VMR itself (not a
# scale relative to prior — that convention is only for ColumnScale/PartialColumns).
h2o_vmr = exp.(rF.x[rH]); o3_vmr = exp.(rF.x[rO])
xa_h2o  = exp.(xa[rH]);   xa_o3  = exp.(xa[rO])          # prior VMR for the ratio
@printf("H₂O VMR range: %.2e–%.2e  ·  O₃ VMR range: %.2e–%.2e (mol/mol)\n",
        minimum(h2o_vmr), maximum(h2o_vmr), minimum(o3_vmr), maximum(o3_vmr))
@printf("retrieved/prior ratio: H₂O %.2f–%.2f×  ·  O₃ %.2f–%.2f×\n",
        minimum(h2o_vmr ./ xa_h2o), maximum(h2o_vmr ./ xa_h2o),
        minimum(o3_vmr ./ xa_o3), maximum(o3_vmr ./ xa_o3))

# ── Dump ───────────────────────────────────────────────────────────────────────────
nedt = sqrt.(diag(Se))
open("data/iasi_joint_full$(TAG)$(BASETAG)_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_full_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], dF.res[i], nedt[i])
    end
end
open("data/iasi_joint_full$(TAG)$(BASETAG)_profiles.csv", "w") do io
    println(io, "level,pressure,T_K,T_prior,H2O_vmr,H2O_prior,O3_vmr,O3_prior,H2O_ratio,O3_ratio,H2O_dof,O3_dof,T_dof")
    for i in 1:nlev
        @printf(io, "%d,%.5f,%.4f,%.4f,%.6e,%.6e,%.6e,%.6e,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                i, base.pressure[i], rF.x[i], base.temperature[i],
                h2o_vmr[i], xa_h2o[i], o3_vmr[i], xa_o3[i],
                h2o_vmr[i]/xa_h2o[i], o3_vmr[i]/xa_o3[i],
                dA[rH[i]], dA[rO[i]], dA[spec.temp_range[i]])
    end
end
jldsave("data/iasi_joint_full$(TAG)$(BASETAG).jld2"; rF=rF, ν=νobs, y=y,
        h2o_vmr=h2o_vmr, o3_vmr=o3_vmr)
prog("wrote data/iasi_joint_full$(TAG)$(BASETAG)_{fit.csv,profiles.csv,jld2}")
