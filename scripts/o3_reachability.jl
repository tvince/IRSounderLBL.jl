# Is the 710-730 residual REACHABLE by an O3 profile change? Test of the hypothesis
# "the O3 prior is too far off / we under-parameterize O3, so the dipole is unretrieved
# shape, not a forward-model error."
#
# Method: linearize O3 as a FULL 50-level profile at the RETRIEVED joint state (where the
# residual actually lives). Take the O3 Jacobian block K_O3 (channels × levels) over
# 710-730, whiten by scene NEΔT, and project the residual onto O3's column space:
#   • how many O3 modes carry signal above noise (singular value > 1) in this window?
#   • what fraction of the 710-730 residual can an O3 profile change CANCEL —
#       (a) using only those physical (s>1) modes, and (b) unconstrained?
#   • the per-level Jacobian SIGN at 715.75 vs 722.75 (does any level flip → dipole
#       reachable) and the peak-weighting pressure of each line.
# If physical O3 modes explain most of the residual → hypothesis SUPPORTED (need more O3
# vertical freedom). If even unconstrained O3 can't → it's lineshape/forward-model.
#
#   julia -t auto --project=. scripts/o3_reachability.jl

using IRSounderLBL
using LinearAlgebra: diag, svd, Diagonal
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const BASELINE = "data/iasi_profile_retrieval.jld2"
const JOINT    = "data/iasi_joint.jld2"              # production 1e-23 joint (retrieved state)
const FITCSV   = "data/iasi_joint_fit.csv"           # residual res_joint_K
const GRANULE  = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const ν_PTS = (715.75, 722.75, 723.25)

# ── Inputs: y / Se and the retrieved joint state ────────────────────────────────────
bl = load(BASELINE); νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
nedt = sqrt.(diag(Se))
rJ = load(JOINT)["rJ"]

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)
prof, T_sfc, _ = unpack_state(rJ.spec, rJ.x, base)   # RETRIEVED atmosphere
@printf("Retrieved state: T_sfc=%.2f K, O3 column scale (from rJ) applied in prof\n", T_sfc)

# residual F−y (production joint) from the fit CSV, aligned to νobs
fit = Dict{Float64,Float64}()
for (i, line) in enumerate(eachline(FITCSV))
    i == 1 && continue
    p = split(line, ","); fit[parse(Float64, p[1])] = parse(Float64, p[4])  # res_joint_K
end
res = [fit[round(ν, digits=4)] for ν in round.(νobs, digits=4)]

# ── Linelists / LM / geometry (match the production joint) ──────────────────────────
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
iasi = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom = ViewingGeometry(g.zen[ifov])
fm = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
      apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF)

# ── One Jacobian at the retrieved state, O3 as a FULL 50-level profile ───────────────
spec = StateVectorSpec(nlev, [H2O, O3];
                       include_temperature=true, include_tsfc=true, include_emissivity=false)
jac = analytic_jacobian(prof, linelists, spec; T_sfc=T_sfc, ε_sfc=ε_SEA, fm...)
K = jac.K
o3r = vmr_range(spec, O3)
K_o3 = K[:, o3r]                       # (nchan × 50)  ∂BT/∂(O3 param) per level
@printf("O3 Jacobian block: %d channels × %d levels\n", size(K_o3, 1), size(K_o3, 2))

# ── Reachability projection over 710-730 ────────────────────────────────────────────
win = (νobs .>= 710.0) .& (νobs .<= 730.0)
Kw  = K_o3[win, :] ./ nedt[win]        # NEΔT-whitened O3 Jacobian in the window
rw  = res[win] ./ nedt[win]            # whitened residual (σ units)
rms(v) = sqrt(sum(abs2, v)/length(v))

F = svd(Kw)
s = F.S
ninfo = count(>(1.0), s)               # O3 modes above noise = info DOF in-window
@printf("\n── O3 information content in 710-730 (%d channels) ──\n", count(win))
@printf("  singular values (σ units), top 8: %s\n",
        join((@sprintf("%.2f", x) for x in s[1:min(8,length(s))]), "  "))
@printf("  O3 modes with s>1 (signal above noise): %d\n", ninfo)

# Fraction of the whitened residual that lies in O3's column space (cancellable by an O3
# profile change). Project r onto span of the top-k left singular vectors.
proj_frac(k) = (k == 0 ? 0.0 : sum(abs2, F.U[:, 1:k]' * rw) / sum(abs2, rw))
@printf("\n── Fraction of the 710-730 residual an O3 profile change can CANCEL ──\n")
for k in (ninfo, min(ninfo+2, length(s)), length(s))
    tag = k == ninfo ? "physical modes (s>1)" : (k == length(s) ? "unconstrained (all modes)" : "s>1 +2")
    @printf("  top-%2d %-26s: %.1f%% of residual variance\n", k, tag, 100*proj_frac(k))
end
@printf("  (whitened residual RMS in-window: %.2f σ)\n", rms(rw))

# Magnitude of the O3 profile change the physical-mode fit requires (log-VMR RMS)
δ = F.V[:, 1:ninfo] * (Diagonal(1 ./ s[1:ninfo]) * (F.U[:, 1:ninfo]' * rw))
@printf("  O3 log-VMR perturbation to achieve it: RMS %.2f, max|Δ| %.2f  (>~1 = unphysical)\n",
        rms(δ), maximum(abs.(δ)))

# ── Per-level sign at 715.75 vs 722.75 (the Jacobian-sign test) ──────────────────────
j715 = argmin(abs.(νobs .- 715.75)); j722 = argmin(abs.(νobs .- 722.75))
r715 = K_o3[j715, :]; r722 = K_o3[j722, :]
flips = [ℓ for ℓ in 1:nlev if sign(r715[ℓ]) * sign(r722[ℓ]) < 0 && abs(r715[ℓ]) > 0.01*maximum(abs.(r715)) && abs(r722[ℓ]) > 0.01*maximum(abs.(r722))]
@printf("\n── Jacobian-sign test: per-level ∂BT/∂O3 at 715.75 vs 722.75 ──\n")
@printf("  peak-weighting level 715.75: %2d (p=%.1f hPa) · 722.75: %2d (p=%.1f hPa)\n",
        argmax(abs.(r715)), base.pressure[argmax(abs.(r715))],
        argmax(abs.(r722)), base.pressure[argmax(abs.(r722))])
@printf("  levels where 715.75 and 722.75 have OPPOSITE sign (>1%% of peak): %d\n", length(flips))
if !isempty(flips)
    @printf("    e.g. level %d (p=%.1f hPa): K715=%+.2e  K722=%+.2e\n",
            flips[1], base.pressure[flips[1]], r715[flips[1]], r722[flips[1]])
end

@printf("\nVERDICT: ")
if proj_frac(ninfo) > 0.5
    @printf("O3 profile change explains %.0f%% of the 710-730 residual with %d physical modes\n",
            100*proj_frac(ninfo), ninfo)
    @printf("  → hypothesis SUPPORTED: it's under-parameterized O3 shape, not lineshape.\n")
else
    @printf("even O3's full column space cancels only %.0f%% (physical) / %.0f%% (unconstrained)\n",
            100*proj_frac(ninfo), 100*proj_frac(length(s)))
    @printf("  → hypothesis NOT supported: the residual is largely OUTSIDE O3's reach = lineshape/FM.\n")
end
