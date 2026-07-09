# How many O3 layers does IASI FOV #1 actually support? Linear DFS pilot with O3 as a
# FULL PROFILE (a single ColumnScale caps at 1 DOF by construction, so it can't answer
# "do we need more layers"). One analytic Jacobian at the prior; then sweep the O3 prior
# σ (0.2 current → 0.5 loosened) cheaply by rebuilding Sa only. Reports total O3 DFS and
# the dfs_partition(target=1) block layout for each σ.
#
#   julia -t auto --project=. scripts/pilot_o3_dfs.jl

using IRSounderLBL
using LinearAlgebra: diag, Symmetric, inv
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const BASELINE = "data/iasi_profile_retrieval.jld2"
const GRANULE  = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const σ_SWEEP  = (0.2, 0.5)          # O3 prior log-scale std: current vs loosened

bl = load(BASELINE); νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
@printf("%d channels\n", length(y)); flush(stdout)

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > O3_SMIN])
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)

g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fm = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
      apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF)

# O3 as a FULL PROFILE alongside H2O full profile — one Jacobian serves the whole sweep.
pilot_spec = StateVectorSpec(nlev, [H2O, O3];
                             include_temperature=true, include_tsfc=true, include_emissivity=false)
@printf("Computing pilot Jacobian (O3 full profile) …\n"); flush(stdout)
jac = analytic_jacobian(base, linelists, pilot_spec;
                        T_sfc=base.temperature[1], ε_sfc=ε_SEA, fm...)
K = jac.K
M = Symmetric(K' * (Se \ K))                      # KᵀSₑ⁻¹K — reused across σ
r_o3  = vmr_range(pilot_spec, O3)
r_h2o = vmr_range(pilot_spec, H2O)

for σo3 in σ_SWEEP
    Sa = build_sa(pilot_spec, base; σ_T=5.0, L_T=1.0,
                  σ_vmr=Dict(H2O=>0.5, O3=>σo3), L_vmr=Dict(H2O=>1.0, O3=>1.0),
                  σ_tsfc=5.0)
    A = (Matrix(M) + Matrix(inv(Sa))) \ Matrix(M)
    d = diag(A)
    dfs_o3 = d[r_o3]
    blocks = dfs_partition(dfs_o3; target_dfs=1.0)
    @printf("\n===== O3 prior σ = %.2f (log-scale, ±%.0f%% 1σ column) =====\n",
            σo3, 100*(exp(σo3)-1))
    @printf("  total DOF=%.2f · O3 profile DOF=%.2f · H2O DOF=%.2f · T DOF=%.2f\n",
            sum(d), sum(dfs_o3), sum(d[r_h2o]), sum(d[pilot_spec.temp_range])); flush(stdout)
    @printf("  dfs_partition(target=1) → %d O3 bulk layers:\n", length(blocks))
    for (m, blk) in enumerate(blocks)
        @printf("    layer %d: levels %2d–%2d  p=%8.2f–%7.2f hPa  ΣDOF=%.2f\n",
                m, first(blk), last(blk), base.pressure[first(blk)], base.pressure[last(blk)],
                sum(dfs_o3[blk]))
    end
end
