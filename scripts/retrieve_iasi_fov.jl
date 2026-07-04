# Proof-of-life optimal-estimation retrieval on a real IASI L1C FOV.
#
# Retrieve ONLY the surface temperature T_sfc (spec.n = 1) from a small window
# microwindow where the surface emission dominates. Emissivity is fixed at 0.98
# (sea surface) — NOT retrieved: over a single window ε and T_sfc are degenerate
# (we measured posterior corr = −0.999, ε averaging-kernel ~0.09), so freeing ε
# only inflates T_sfc's uncertainty without improving the fit.
# Fixed: the whole T(p) profile, pressure/altitude grid, and all gases — in
# particular CO₂ at 432 ppm in the lower column (AFGL upper taper kept).
# Window: 820–840 cm⁻¹ (12 µm atmospheric window, past the CO₂ ν₂ band edge and
# below the O₃ ν₂ band we don't model → a clean surface thermometer).
#
# Run:  julia -t auto --project=. scripts/retrieve_iasi_fov.jl [granule_path]

using IRSounderLBL
using LinearAlgebra: diag
using Printf

const GRANULE = length(ARGS) >= 1 ? ARGS[1] :
    "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

const ν_LO, ν_HI = 820.0, 840.0          # small window microwindow (cm⁻¹)
const CO2_PPM    = 432.0                  # fixed lower-column CO₂

# ── 1. Read granule, screen for a clear, glint-free, near-nadir FOV ─────────────
println("Reading $(basename(GRANULE)) over $(ν_LO)–$(ν_HI) cm⁻¹ …")
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0),       # near-clear (AVHRR cloud fraction <5%)
                  sralim=(15.0, 180.0),    # outside the 15° sun-glint cone
                  zenlim=(0.0, 25.0),      # near-nadir → simple airmass
                  max_fov=400)
nfov(g) > 0 || error("no FOV passed the clear/glint/zenith screening")
ifov = argmin(cloud_fraction(g))           # clearest of the pool
νobs, y = measurement(g, ifov)
@printf("  %d candidate FOVs; using #%d  lat=%.2f lon=%.2f  zen=%.1f° sza=%.1f° φᵣ=%.1f° cld=%.1f%%\n",
        nfov(g), ifov, g.lat[ifov], g.lon[ifov], g.zen[ifov], g.sza[ifov],
        solar_reflection_angle(g)[ifov], cloud_fraction(g)[ifov])
@printf("  measured BT: %.1f … %.1f K  (%d channels)\n", minimum(y), maximum(y), length(y))

# ── 2. Base atmosphere: AFGL US-Standard, CO₂ → 432 ppm lower column ────────────
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]    # scale so surface = 432 ppm
nlev = n_levels(base)
@printf("  AFGL profile: %d levels, %.1f–%.4g hPa, surface T=%.1f K, CO₂=%.0f ppm\n",
        nlev, base.pressure[1], base.pressure[end], base.temperature[1],
        base.vmr[CO2][1]*1e6)

# ── 3. Forward-model context (geometry, instrument, absorbers) ──────────────────
co2 = load_linelist("data/co2_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fm    = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian, apply_continuum=true,
         internal_dnu=0.0025)   # ≪0.01 K BT error, ≫ faster than the 0.001 validation grid

# ── 4. State spec + priors:  retrieve ONLY T_sfc (ε fixed at 0.98) ──────────────
const ε_SEA = 0.98                                   # fixed sea-surface emissivity
spec = StateVectorSpec(nlev, GasSpecies[];          # no retrieved gases (CO₂ fixed)
                       include_temperature=false,   # T(p) profile fixed at AFGL
                       include_tsfc=true,
                       include_emissivity=false)     # ε NOT retrieved (degenerate)
xa = pack_state(spec, base)                          # prior = AFGL surface T
Sa = build_sa(spec, base; σ_tsfc=5.0)                # 1×1: σ(T_sfc) = 5 K
Se = apodized_measurement_covariance(νobs, 0.3)      # 0.3 K NEdT, IASI apodization
@printf("  state: %s (n=%d);  prior σ_Tsfc=5 K, Se=0.3 K;  xa(T_sfc)=%.1f K, ε fixed=%.2f\n",
        string(spec), spec.n, xa[spec.tsfc_index], ε_SEA)

# ── 5. Retrieve ────────────────────────────────────────────────────────────────
println("Running optimal estimation …")
@time r = optimal_estimation(y, spec, base, linelists;
                             xa=xa, Sa=Sa, Se=Se, ε_fixed=ε_SEA, fm_kwargs=fm, verbose=true)

# ── 6. Report ────────────────────────────────────────────────────────────────────
println("\n", r)
@printf("converged=%s  iters=%d  χ²=%.1f (n_y=%d)  DOF=%.2f  H=%.2f bits\n",
        r.converged, r.n_iter, r.chi2, length(y), r.dof, r.H)
@printf("spectral fit: max|F−y|=%.2f K  rms=%.3f K\n",
        maximum(abs.(r.y_fit .- y)), sqrt(sum(abs2, r.y_fit .- y)/length(y)))

σpost = sqrt.(max.(diag(r.S_hat), 0.0))
it = spec.tsfc_index
@printf("\nT_sfc: prior=%.2f K  retrieved=%.2f K  Δ=%+.2f K  σ_post=%.2f K  AKdiag=%.2f  (ε fixed=%.2f)\n",
        xa[it], r.x[it], r.x[it]-xa[it], σpost[it], r.A[it,it], ε_SEA)

# ── 7. Dump observed vs modelled BT for plotting ────────────────────────────────
open("data/iasi_retrieval_fov.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,bt_model_K,residual_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", νobs[i], y[i], r.y_fit[i], r.y_fit[i]-y[i])
    end
end
println("\nwrote data/iasi_retrieval_fov.csv")
