# Joint retrieval of T + H2O + O3, with BOTH H2O and O3 as DFS-partitioned bulk layers.
# Variant of retrieve_iasi_joint.jl: instead of a single O3 ColumnScale, O3 is given the
# same bulk-layer treatment as H2O — a full-profile pilot Jacobian → averaging-kernel
# diagonal → dfs_partition → PartialColumns. Tests whether resolving O3 vertically shrinks
# the 710-730 residual (see scripts/pilot_o3_dfs.jl: loosening O3 σ→0.5 lifts O3 profile
# DOF 1.14→1.69, opening a real 2nd (upper-strat) layer).
#
# One pilot Jacobian serves both species (H2O + O3 both full profiles). Compares against
#   • the fixed-O3 T-only prior  (data/iasi_profile_o3.jld2 key rC)
#   • the O3-COLUMN joint         (data/iasi_joint.jld2 key rJ) — the current best fit
#
#   julia -t auto --project=. scripts/retrieve_iasi_joint_o3layers.jl

using IRSounderLBL
using LinearAlgebra: diag, Symmetric, inv
using Printf
using JLD2: jldsave, load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN   = 1e-23
const σ_H2O, σ_O3 = 0.5, 0.5              # loosened O3 prior (log-scale, ±65% 1σ)
const TARGET_DFS = 1.0
const BASELINE  = "data/iasi_profile_retrieval.jld2"
const PRIOR_O3  = "data/iasi_profile_o3.jld2"   # fixed-O3, T-only (key rC)
const JOINT_COL = "data/iasi_joint.jld2"        # O3-column joint (key rJ)
const GRANULE   = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const REGIONS = ((700.0, 710.0, "O3 ν₂ Q region"), (710.0, 730.0, "LM-overshoot region"))
const ν_PTS   = (715.75, 722.75, 723.25)

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Inputs ──────────────────────────────────────────────────────────────────────────
prog("Loading baseline y / Se …")
bl = load(BASELINE); νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
@printf("  %d channels\n", length(y)); flush(stdout)

rPrior = load(PRIOR_O3)["rC"]
rCol   = load(JOINT_COL)["rJ"]
@printf("  prior  (fixed-O3, T-only): χ²=%.1f  rms=%.4f K\n",
        rPrior.chi2, sqrt(sum(abs2, rPrior.y_fit .- y)/length(y)))
@printf("  O3-col joint (reference) : χ²=%.1f  rms=%.4f K\n",
        rCol.chi2, sqrt(sum(abs2, rCol.y_fit .- y)/length(y))); flush(stdout)

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

prog("Loading CO₂(1–4) / H₂O(1–3) / O₃(1–4, S>$(O3_SMIN)) linelists …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > O3_SMIN])
@printf("  O₃: %d/%d lines (S>%.0e)\n", length(o3.lines), length(o3f.lines), O3_SMIN); flush(stdout)
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

prog("Loading HITRAN relaxation matrix (645–800) …")
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)

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

# ── Pilot: ONE Jacobian with H2O AND O3 full profiles → DFS diagonals for both ────────
prog("Pilot: full-profile H₂O+O₃ Jacobian at the prior → DFS diagonals …")
pilot_spec = StateVectorSpec(nlev, [H2O, O3];
                             include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa_pilot = build_sa(pilot_spec, base; σ_T=5.0, L_T=1.0,
                    σ_vmr=Dict(H2O=>σ_H2O, O3=>σ_O3), L_vmr=Dict(H2O=>1.0, O3=>1.0),
                    σ_tsfc=5.0)
jac = analytic_jacobian(base, linelists, pilot_spec;
                        T_sfc=base.temperature[1], ε_sfc=ε_SEA, fm...)
K = jac.K
M = Symmetric(K' * (Se \ K))
A = (Matrix(M) + Matrix(inv(Sa_pilot))) \ Matrix(M)
d = diag(A)
dfs_h2o = d[vmr_range(pilot_spec, H2O)]
dfs_o3  = d[vmr_range(pilot_spec, O3)]
@printf("  total DOF=%.2f · H₂O DOF=%.2f · O₃ DOF=%.2f · T DOF=%.2f\n",
        sum(d), sum(dfs_h2o), sum(dfs_o3), sum(d[pilot_spec.temp_range])); flush(stdout)

blk_h2o = dfs_partition(dfs_h2o; target_dfs=TARGET_DFS)
blk_o3  = dfs_partition(dfs_o3;  target_dfs=TARGET_DFS)
report_blocks(name, blocks, dfs) = begin
    @printf("  %s → %d bulk layers:\n", name, length(blocks))
    for (m, b) in enumerate(blocks)
        @printf("    layer %d: levels %2d–%2d  p=%8.2f–%7.2f hPa  ΣDOF=%.2f\n",
                m, first(b), last(b), base.pressure[first(b)], base.pressure[last(b)], sum(dfs[b]))
    end
end
report_blocks("H₂O", blk_h2o, dfs_h2o)
report_blocks("O₃",  blk_o3,  dfs_o3); flush(stdout)

# ── Final joint retrieval: T(full) + H2O(bulk) + O3(bulk) ─────────────────────────────
B_h2o = partial_column_basis(nlev, blk_h2o; taper=:boxcar)
B_o3  = partial_column_basis(nlev, blk_o3;  taper=:boxcar)
spec  = StateVectorSpec(nlev, [PartialColumns(H2O, B_h2o), PartialColumns(O3, B_o3)];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0,
              σ_col=Dict(H2O=>σ_H2O, O3=>σ_O3), σ_tsfc=5.0)
xa = pack_state(spec, base)
@printf("  joint state: n=%d  (T %d + T_sfc + H₂O %d layers + O₃ %d layers)\n",
        spec.n, nlev, length(blk_h2o), length(blk_o3)); flush(stdout)

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
dCol   = report("REF: O3-COLUMN joint (T + H₂O bulk + O₃ column)", rCol)

prog("Final joint retrieval (T + H₂O bulk + O₃ BULK LAYERS) …")
@time rJ = optimal_estimation(y, spec, base, linelists;
                              xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)
dJ = report("JOINT: T + H₂O($(length(blk_h2o))) + O₃($(length(blk_o3)) BULK LAYERS)", rJ)

o3_scales  = exp.(rJ.x[vmr_range(spec, O3)])
h2o_scales = exp.(rJ.x[vmr_range(spec, H2O)])
@printf("\nRetrieved O₃ bulk-layer scales (× climatology):\n")
for (m, b) in enumerate(blk_o3)
    @printf("  layer %d (%.1f–%.1f hPa): %.3f×\n",
            m, base.pressure[first(b)], base.pressure[last(b)], o3_scales[m])
end
@printf("Retrieved H₂O bulk-layer scales (× climatology):\n")
for (m, b) in enumerate(blk_h2o)
    @printf("  layer %d (%.0f–%.0f hPa): %.3f×\n",
            m, base.pressure[first(b)], base.pressure[last(b)], h2o_scales[m])
end

@printf("\n########## O3-COLUMN joint → O3-BULK-LAYERS joint ##########\n")
@printf("full-band rms:  %.4f → %.4f K\n", dCol.rms, dJ.rms)
@printf("total χ²     :  %.1f → %.1f\n", dCol.χ2, dJ.χ2)
for (lo, hi, name) in REGIONS
    m = inwin(νobs, lo, hi)
    @printf("%-22s rms:  %.4f → %.4f K\n", name, rms(dCol.res[m]), rms(dJ.res[m]))
end
for νp in ν_PTS
    j = argmin(abs.(νobs .- νp))
    @printf("ν=%.2f:  %+.3f → %+.3f K\n", νp, dCol.res[j], dJ.res[j])
end

# ── Dump ──────────────────────────────────────────────────────────────────────────────
nedt = sqrt.(diag(Se))
open("data/iasi_joint_o3layers_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,res_prior_K,res_o3col_K,res_o3layers_K,nedt_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νobs[i], y[i], dPrior.res[i], dCol.res[i], dJ.res[i], nedt[i])
    end
end
jldsave("data/iasi_joint_o3layers.jld2"; rJ=rJ, ν=νobs, y=y,
        blk_h2o=blk_h2o, blk_o3=blk_o3, dfs_h2o=dfs_h2o, dfs_o3=dfs_o3,
        o3_scales=o3_scales, h2o_scales=h2o_scales)
prog("wrote data/iasi_joint_o3layers_{fit.csv,jld2}")
