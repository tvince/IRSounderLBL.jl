# Temperature-PROFILE optimal-estimation retrieval on a real IASI L1C FOV.
#
# Graduates the proof-of-life (T_sfc-only, scripts/retrieve_iasi_fov.jl) to the
# real sounder problem: retrieve the full T(p) profile + T_sfc from the CO₂ ν₂
# band (645–800 cm⁻¹), whose channels span the surface (window edges) up to the
# upper stratosphere (opaque Q-branch at 667 cm⁻¹).
#
# State : T at every level + T_sfc.  ε fixed 0.98, CO₂ fixed 432 ppm, H₂O/other
#         gases fixed at AFGL.  Sₐ = build_sa C²-smooth (matern52) vertical
#         correlation in log-p (the thesis §3.3 constraint), σ_T=5 K, L_T=1 scale
#         height; σ_Tsfc=5 K.  Sₑ = apodized IASI covariance, 0.3 K NEdT.
#
# Run:  julia -t auto --project=. scripts/retrieve_iasi_profile.jl [ν_lo ν_hi] [granule]

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: jldsave

const ν_LO  = length(ARGS) >= 2 ? parse(Float64, ARGS[1]) : 645.0
const ν_HI  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
const GRANULE = length(ARGS) >= 3 ? ARGS[3] :
    "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const CO2_PPM = 432.0
const ε_SEA   = 0.98
# Full RetrievalResult+K/Se/Sa dump (for channel-selection/DFS reuse). On by
# default; `SAVE_JLD2=0 julia … retrieve_iasi_profile.jl` skips it (the CSVs
# always write). Env var, not a positional ARG, to leave `[ν_lo ν_hi] [granule]`.
const SAVE_JLD2 = get(ENV, "SAVE_JLD2", "1") != "0"

# Timestamped progress print that flushes immediately — Julia block-buffers stdout
# to a file/pipe, so without the flush a tailed background log shows nothing until
# the run ends. `optimal_estimation(verbose=true)` streams per-iteration J/d²/γ
# through the same flushed path.
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

# ── 3. Forward-model context ────────────────────────────────────────────────────
# CO₂ iso 1–4: iso-4 (627) supplies the ν₂ Q-branch bandhead near 665 cm⁻¹ that,
# when missing, leaves a ~−8.7 K residual spike there (see scripts/retrieve_iasi_
# profile_lm_iso4.jl). H₂O stays iso 1–3.
prog("Loading CO₂ (iso 1–4) / H₂O (iso 1–3) linelists ($(ν_LO-25)–$(ν_HI+25) cm⁻¹) …")
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fm    = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
         apply_continuum=true, internal_dnu=0.0025)

# ── 4. State spec + priors:  T(p) profile + T_sfc ───────────────────────────────
spec = StateVectorSpec(nlev, GasSpecies[];
                       include_temperature=true, include_tsfc=true, include_emissivity=false)
xa = pack_state(spec, base)
Sa = build_sa(spec, base; σ_T=5.0, L_T=1.0, σ_tsfc=5.0, kernel=:matern52)
# Sₑ diagonal = scene-specific NEΔT (thesis Eq. 2.3) from the observed BT scene,
# anchored to IASI NEΔT_280K≈0.25 K; off-diagonals = apodization (§3.6).
const NEDT_280 = 0.25
σ_scene = scene_nedt(νobs, y; nedt_280K=NEDT_280, T_ref=280.0)
Se = scene_measurement_covariance(νobs, y; nedt_280K=NEDT_280, T_ref=280.0)
@printf("  state: %s (n=%d);  Sₐ: σ_T=5 K L_T=1 (C²), σ_Tsfc=5 K\n", string(spec), spec.n)
@printf("  Sₑ: scene NEΔT (Eq 2.3, NEΔT₂₈₀=%.2f K) → σ_BT %.2f…%.2f K  (median %.2f) across band\n",
        NEDT_280, minimum(σ_scene), maximum(σ_scene),
        sort(σ_scene)[cld(length(σ_scene),2)])
flush(stdout)

# ── 5. Retrieve ─────────────────────────────────────────────────────────────────
prog("Running optimal estimation (profile, n=$(spec.n) state, $(length(y)) ch) …")
prog("  (the first Jacobian is the heaviest step — expect a quiet minutes-long " *
     "pause before 'iter 1'; per-iteration J/d²/γ stream below as they land)")
@time r = optimal_estimation(y, spec, base, linelists;
                             xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)
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
open("data/iasi_profile_retrieval_Tp.csv", "w") do io
    println(io, "level,p_hPa,T_prior_K,T_retr_K,sigma_post_K,AK_diag")
    for (i, lvl) in enumerate(tr)
        @printf(io, "%d,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                i, base.pressure[i], xa[lvl], r.x[lvl], σpost[lvl], r.A[lvl,lvl])
    end
end
open("data/iasi_profile_retrieval_fit.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,bt_model_K,residual_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], r.y_fit[i], r.y_fit[i]-y[i])
    end
end
println("\nwrote data/iasi_profile_retrieval_{Tp,fit}.csv")

# Full retrieval object for later analysis without re-running the (minutes-long)
# forward model: `result` carries the Jacobian K at the solution plus A/G/Ŝ/DOF/H;
# Se/Sa/xa let the linear diagnostics be regenerated (or extended to channel
# selection / DFS studies, backlog #5/#6) under different assumptions. Opt out
# with SAVE_JLD2=0.
if SAVE_JLD2
    jldsave("data/iasi_profile_retrieval.jld2"; result=r, K=r.K, Se=Se, Sa=Sa,
            xa=xa, ν=νobs, y=y)
    println("wrote data/iasi_profile_retrieval.jld2 (RetrievalResult + K, Se, Sa, xa, ν, y)")
else
    println("SAVE_JLD2=0 → skipped data/iasi_profile_retrieval.jld2")
end
