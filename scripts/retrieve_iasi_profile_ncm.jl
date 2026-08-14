# Temperature-PROFILE optimal-estimation retrieval on a real IASI L1C FOV, using
# the EUMETSAT-published IASI L1C **Noise Covariance Matrix (NCM)** product as Sₑ
# (EO:EUM:DAT:1099) instead of the analytically-modelled scene covariance.
#
# This is the end-to-end verification of the host-noise-covariance path:
#   read_iasi_ncm → subset_channels(→ retrieval grid) → to_bt(scene BT) → Se.
# Retrieval runs in BT space (observable=:bt) on a 0.0005 cm⁻¹ internal grid.
#
# State : T at every level + T_sfc.  ε fixed 0.98, CO₂ 432 ppm, H₂O AFGL fixed.
#         Sₐ = build_sa matern52 in log-p, σ_T=5 K, L_T=1, σ_Tsfc=5 K.
#         Sₑ = EUMETSAT NCM, radiance→BT-linearized at the observed scene.
#
# Run:  IASI_NCM=/path/to/IASI_NCM_1C_*.nc \
#       julia -t auto --project=. scripts/retrieve_iasi_profile_ncm.jl [ν_lo ν_hi] [granule]

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave

const ν_LO  = length(ARGS) >= 2 ? parse(Float64, ARGS[1]) : 645.0
const ν_HI  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
const GRANULE = length(ARGS) >= 3 ? ARGS[3] :
    "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const NCM_PATH = get(ENV, "IASI_NCM",
    "data/iasi_ncm/IASI_NCM_1C_M03_PN01_20250827145836Z.nc")
const CO2_PPM = 432.0
const ε_SEA   = 0.98
const INTERNAL_DNU = 0.0005
const SAVE_JLD2 = get(ENV, "SAVE_JLD2", "1") != "0"

const T0 = time()
prog(msg) = (@printf("[%7.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── 1. Read granule, screen for a clear, glint-free, near-nadir FOV ─────────────
prog("Reading $(basename(GRANULE)) over $(ν_LO)–$(ν_HI) cm⁻¹ …")
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
nfov(g) > 0 || error("no FOV passed clear/glint/zenith screening")
ifov = argmin(cloud_fraction(g))
νobs, y = measurement(g, ifov)
@printf("  %d FOVs; using #%d lat=%.2f lon=%.2f zen=%.1f° cld=%.1f%%;  BT %.1f…%.1f K (%d ch)\n",
        nfov(g), ifov, g.lat[ifov], g.lon[ifov], g.zen[ifov], cloud_fraction(g)[ifov],
        minimum(y), maximum(y), length(y))
flush(stdout)

# ── 2. Base atmosphere ──────────────────────────────────────────────────────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

# ── 3. Forward-model context (0.0005 cm⁻¹ internal grid) ─────────────────────────
prog("Loading CO₂ (iso 1–4) / H₂O (iso 1–3) linelists ($(ν_LO-25)–$(ν_HI+25) cm⁻¹) …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
sounder = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fm    = (sounder=sounder, geom=geom, with_ils=true, apodization=:gaussian,
         apply_continuum=true, internal_dnu=INTERNAL_DNU)

# ── 4. State spec + priors:  T(p) profile + T_sfc ───────────────────────────────
spec = StateVectorSpec(nlev, GasSpecies[];
                       include_temperature=true, include_tsfc=true, include_emissivity=false)
xa = pack_state(spec, base)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0, σ_tsfc=5.0, kernel=:matern52)

# ── 4b. Sₑ from the EUMETSAT NCM (the point of this run) ─────────────────────────
prog("Loading IASI NCM $(basename(NCM_PATH)) …")
isfile(NCM_PATH) || error("NCM file not found: $(NCM_PATH)  (set IASI_NCM=…)")
nc_full = read_iasi_ncm(NCM_PATH; ν_window=(ν_LO, ν_HI))     # windowed hyperslab read
nc_sub  = subset_channels(nc_full, νobs)                     # align to retrieval channels
nc_bt   = to_bt(nc_sub, y)                                   # radiance → BT at the scene
Se = measurement_covariance(nc_bt)                           # Symmetric K² matrix
σ_ncm = sqrt.([nc_bt.cov[i,i] for i in eachindex(νobs)])
@printf("  state: %s (n=%d);  Sₐ: σ_T=5 K L_T=1 (C²), σ_Tsfc=5 K\n", string(spec), spec.n)
@printf("  Sₑ: EUMETSAT NCM → BT σ %.3f…%.3f K (median %.3f); neighbour corr %.3f\n",
        minimum(σ_ncm), maximum(σ_ncm), sort(σ_ncm)[cld(length(σ_ncm),2)],
        nc_bt.cov[1,2]/sqrt(nc_bt.cov[1,1]*nc_bt.cov[2,2]))
flush(stdout)

# ── 5. Retrieve (BT space) ───────────────────────────────────────────────────────
prog("Running optimal estimation (profile n=$(spec.n), $(length(y)) ch, internal_dnu=$(INTERNAL_DNU)) …")
prog("  (0.0005 cm⁻¹ grid ⇒ the first Jacobian is heavy — expect a long quiet pause before 'iter 1')")
@time r = optimal_estimation(y, spec, base, linelists;
                             xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA,
                             observable=:bt, fm_kwargs=fm, verbose=true)
prog("optimal estimation returned (converged=$(r.converged), $(r.n_iter) iters)")

# ── 6. Report ───────────────────────────────────────────────────────────────────
@printf("\nconverged=%s iters=%d  χ²=%.1f (n_y=%d)  DOF=%.2f  H=%.2f bits\n",
        r.converged, r.n_iter, r.chi2, length(y), r.dof, r.H)
@printf("spectral fit: max|F−y|=%.2f K  rms=%.3f K\n",
        maximum(abs.(r.y_fit .- y)), sqrt(sum(abs2, r.y_fit .- y)/length(y)))

σpost = sqrt.(max.(diag(r.S_hat), 0.0))
tr = spec.temp_range
it = spec.tsfc_index
@printf("\nT_sfc: prior=%.2f → %.2f K (Δ%+.2f)  σ_post=%.2f K  AKdiag=%.2f\n",
        xa[it], r.x[it], r.x[it]-xa[it], σpost[it], r.A[it,it])
println("\n  level   p[hPa]    T_prior   T_retr    Δ      σ_post  AKdiag")
for (i, lvl) in enumerate(tr)
    @printf("  %3d  %9.2f  %7.2f  %7.2f  %+6.2f  %6.2f  %5.2f\n",
            i, base.pressure[i], xa[lvl], r.x[lvl], r.x[lvl]-xa[lvl], σpost[lvl], r.A[lvl,lvl])
end
@printf("\nprofile DOF (Σ AKdiag over T-levels) = %.2f of %d levels\n",
        sum(r.A[l,l] for l in tr), nlev)

# ── 7. Dump for plotting ────────────────────────────────────────────────────────
open("data/iasi_profile_ncm_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_prior_K,T_retr_K,sigma_post_K,AK_diag")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], xa[lvl], r.x[lvl], σpost[lvl], r.A[lvl,lvl])
    end
end
open("data/iasi_profile_ncm_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,bt_model_K,residual_K,sigma_ncm_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], r.y_fit[i], r.y_fit[i]-y[i], σ_ncm[i])
    end
end
println("\nwrote data/iasi_profile_ncm_{Tp,fit}.csv")

if SAVE_JLD2
    jldsave("data/iasi_profile_ncm.jld2"; result=r, K=r.K, Se=Matrix(Se), Sa=Sa,
            xa=xa, ν=νobs, y=y, sigma_ncm=σ_ncm)
    println("wrote data/iasi_profile_ncm.jld2 (RetrievalResult + K, Se, Sa, xa, ν, y)")
end
