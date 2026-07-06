# Joint retrieval of temperature + H2O + total-column O3 on IASI FOV #1 (645–800),
# VP_Y lm=5, CO2 iso 1–4. Two design choices, both from the reduced-VMR basis work:
#
#   • O3  → ColumnScale(O3):    a single multiplicative total-column scale
#                               (VMR = VMR_ref·exp(θ)); 1 state element, 1 Jacobian
#                               column. The fixed climatological O3 overshot the
#                               722–723 residual, so we retrieve the *amount*.
#   • H2O → PartialColumns(O3): "bulk layers" whose block edges are placed by the
#                               DFS diagonal helper. A cheap linear pilot (one
#                               Jacobian at the prior) gives the H2O averaging-kernel
#                               diagonal; dfs_partition cuts it into ~1-DOF blocks so
#                               we retrieve H2O only where the measurement informs it.
#   • T   → full profile (50 levels) + T_sfc, as before. ε fixed 0.98.
#
# Same y / Se as the T-only baseline (data/iasi_profile_retrieval.jld2); Sa and xa
# are rebuilt for the joint state. Compare against the best prior fit: VP_Y lm=5 + O3
# with FIXED climatological O3, T-only (data/iasi_profile_o3.jld2 key rC).
#
# Requires data/{co2,h2o,o3}_645_2760*.par. Run:
#   julia -t auto --project=. scripts/retrieve_iasi_joint.jl

using IRSounderLBL
using LinearAlgebra: diag, Symmetric
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const LM_METHOD = lowercase(get(ENV, "JOINT_LM", "vpy"))   # "vpy" (default) or "vpw"
const TAG = LM_METHOD == "vpw" ? "_vpw" : ""
const O3_SMIN   = 1e-23
const TARGET_DFS = 1.0                     # ≈ one DOF per H2O bulk layer
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const PRIOR_O3  = "data/iasi_profile_o3.jld2"   # VP_Y lm=5, +O3 (FIXED VMR), T-only (key rC)
const GRANULE   = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const REGIONS = ((700.0, 710.0, "O3 ν₂ Q region"), (710.0, 730.0, "LM-overshoot region"))
const ν_PTS   = (715.75, 722.75, 723.25)

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Inputs ────────────────────────────────────────────────────────────────────────
prog("Loading baseline y / Se …")
bl = load(BASELINE)
νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
@printf("  %d channels\n", length(y)); flush(stdout)

rPrior = load(PRIOR_O3)["rC"]              # best prior: fixed-VMR O3, T-only
@printf("  prior (fixed-O3, T-only): χ²=%.1f  rms=%.4f K\n",
        rPrior.chi2, sqrt(sum(abs2, rPrior.y_fit .- y)/length(y))); flush(stdout)

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

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
@printf("  line mixing: %s\n", LM_METHOD == "vpw" ? "VP_W (full matrix)" : "VP_Y (first-order)"); flush(stdout)

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

# ── Pilot: linear DFS estimate for H2O (one Jacobian at the prior, no iteration) ─────
# Retrieve H2O at FULL resolution *only* to measure where the information is. The
# averaging kernel at the prior, A = (Sₐ⁻¹ + KᵀSₑ⁻¹K)⁻¹ KᵀSₑ⁻¹K, gives per-level DOF.
prog("Pilot: full-profile H₂O Jacobian at the prior → DFS diagonal …")
pilot_spec = StateVectorSpec(nlev, [H2O, ColumnScale(O3)];
                             include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa_pilot = build_sa(pilot_spec, base; σ_T=5.0, L_T=1.0,
                    σ_vmr=Dict(H2O=>0.5), L_vmr=Dict(H2O=>1.0),
                    σ_col=Dict(O3=>0.2), σ_tsfc=5.0)
jac = analytic_jacobian(base, linelists, pilot_spec;
                        T_sfc=base.temperature[1], ε_sfc=ε_SEA, fm...)
K = jac.K
SeiK = Se \ K
M    = Symmetric(K' * SeiK)                       # KᵀSₑ⁻¹K
A    = (Matrix(M) + Matrix(inv(Sa_pilot))) \ Matrix(M)
dfs  = diag(A)
r_h2o = vmr_range(pilot_spec, H2O)
dfs_h2o = dfs[r_h2o]
@printf("  total DOF=%.2f  ·  H₂O DOF=%.2f  ·  O₃ DOF=%.2f  ·  T DOF=%.2f\n",
        sum(dfs), sum(dfs_h2o), dfs[vmr_range(pilot_spec, O3)][1], sum(dfs[pilot_spec.temp_range]));
flush(stdout)

blocks = dfs_partition(dfs_h2o; target_dfs=TARGET_DFS)
@printf("  dfs_partition(target=%.1f) → %d H₂O bulk layers:\n", TARGET_DFS, length(blocks))
for (m, blk) in enumerate(blocks)
    @printf("    layer %d: levels %2d–%2d  p=%7.1f–%6.1f hPa  ΣDOF=%.2f\n",
            m, first(blk), last(blk), base.pressure[first(blk)], base.pressure[last(blk)],
            sum(dfs_h2o[blk]))
end
flush(stdout)

# ── Final joint retrieval: T(full) + H2O(bulk layers) + O3(column) ──────────────────
B_h2o = partial_column_basis(nlev, blocks; taper=:boxcar)
spec  = StateVectorSpec(nlev, [PartialColumns(H2O, B_h2o), ColumnScale(O3)];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0,
              σ_col=Dict(H2O=>0.5, O3=>0.2), σ_tsfc=5.0)
xa = pack_state(spec, base)
@printf("  joint state: n=%d  (T %d + T_sfc + H₂O %d layers + O₃ column)\n",
        spec.n, nlev, length(blocks)); flush(stdout)

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

dPrior = report("PRIOR: VP_Y lm=5 + fixed O3, T-only", rPrior)

prog("Final joint retrieval ($(uppercase(LM_METHOD)): T + H₂O bulk layers + O₃ column) …")
@time rJ = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)
dJ = report("JOINT $(uppercase(LM_METHOD)): T + H₂O($(length(blocks)) layers) + O₃ column", rJ)

# Retrieved reduced params: O3 column scale and per-layer H2O scales.
o3_scale = exp(rJ.x[vmr_range(spec, O3)][1])
h2o_scales = exp.(rJ.x[vmr_range(spec, H2O)])
@printf("\nRetrieved O₃ total-column scale: %.3f× climatology\n", o3_scale)
@printf("Retrieved H₂O bulk-layer scales (× climatology):\n")
for (m, blk) in enumerate(blocks)
    @printf("  layer %d (%.0f–%.0f hPa): %.3f×\n",
            m, base.pressure[first(blk)], base.pressure[last(blk)], h2o_scales[m])
end

@printf("\n########## PRIOR (fixed O3, T-only) → JOINT (T+H₂O+O₃col) ##########\n")
@printf("full-band rms:  %.4f → %.4f K\n", dPrior.rms, dJ.rms)
@printf("total χ²     :  %.1f → %.1f\n", dPrior.χ2, dJ.χ2)
for (lo, hi, name) in REGIONS
    m = inwin(νobs, lo, hi)
    @printf("%-22s rms:  %.4f → %.4f K\n", name, rms(dPrior.res[m]), rms(dJ.res[m]))
end

# ── Dump ────────────────────────────────────────────────────────────────────────────
nedt = sqrt.(diag(Se))
open("data/iasi_joint$(TAG)_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_prior_K,res_joint_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], dPrior.res[i], dJ.res[i], nedt[i])
    end
end
jldsave("data/iasi_joint$(TAG).jld2"; rJ=rJ, ν=νobs, y=y, blocks=blocks,
        dfs_h2o=dfs_h2o, o3_scale=o3_scale, h2o_scales=h2o_scales)
prog("wrote data/iasi_joint$(TAG)_{fit.csv,jld2}")
