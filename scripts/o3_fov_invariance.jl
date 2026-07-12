# Is the 715.5-715.75 spike SYSTEMATIC (instrument spectroscopy / model artifact / IASI
# channel) or ATMOSPHERIC (high-altitude T read through the strongest O3 line cores)? Fork
# test: hold the MODEL fixed at the FOV#1 retrieved state, sweep the N clearest FOVs (only
# geometry + each FOV's own obs change), high-pass the residual (raw − boxcar-smooth) to
# isolate the sharp 2-channel component, and report it at 715.50/715.75.
#   • sharp residual CONSTANT across FOVs → SYSTEMATIC (model fixed ⇒ obs sharp feature is
#     atmosphere-independent, or the model itself carries the artifact) = instrument/spectro.
#   • sharp residual VARIES with scene → ATMOSPHERIC (obs strong-line core depth changes with
#     each FOV's real upper-strat T; fixed model can't track it).
# Also reports the OBS-only high-passed sharp value (is the measurement itself spiky there?).
#
#   julia -t auto --project=. scripts/o3_fov_invariance.jl

using IRSounderLBL
using LinearAlgebra: diag
using Printf
using JLD2: load
using Statistics: mean, std

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const NFOV = 8
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const JOINT   = "data/iasi_joint.jld2"
const ν_SPIKE = (715.50, 715.75)

# Fixed model = FOV#1 retrieved state
rJ = load(JOINT)["rJ"]
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
prof, T_sfc, _ = unpack_state(rJ.spec, rJ.x, base)

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > O3_SMIN])
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)

g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
cf = cloud_fraction(g)
order = sortperm(cf)[1:min(NFOV, length(cf))]     # clearest FOVs
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)

# high-pass: subtract a boxcar-smoothed spectrum (±HALF cm⁻¹ ⇒ removes smooth, keeps 2-ch spike)
const HALF = 4                                    # ±4 channels ≈ ±1 cm⁻¹
function highpass(v)
    n = length(v); s = similar(v)
    for i in 1:n
        lo = max(1, i-HALF); hi = min(n, i+HALF)
        s[i] = v[i] - mean(@view v[lo:hi])
    end
    return s
end

@printf("Fixed model = FOV#1 retrieved state (T_sfc=%.2f). Sweeping %d clearest FOVs.\n\n",
        T_sfc, length(order))
@printf("%5s %7s  %10s %10s   %10s %10s\n",
        "FOV", "cloud%", "resSharp715.50", "resSharp715.75", "obsSharp715.50", "obsSharp715.75")
νcol = nothing; jsp = Int[]
rs = [Float64[] for _ in ν_SPIKE]; os = [Float64[] for _ in ν_SPIKE]
for ifov in order
    geom = ViewingGeometry(g.zen[ifov])
    fmkw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
            apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF,
            T_sfc=T_sfc, ε_sfc=ε_SEA)
    νg, _, model = iasi_forward_model(prof, linelists; fmkw...)
    νv = νg isa WavenumberGrid ? νg.ν : collect(Float64, νg)
    νobs, obs = measurement(g, ifov)
    if νcol === nothing
        global νcol = νv; global jsp = [argmin(abs.(νv .- νs)) for νs in ν_SPIKE]
    end
    resid = model .- obs
    rsh = highpass(resid); osh = highpass(obs)
    @printf("%5d %6.2f%%  %+10.3f %+10.3f   %+10.3f %+10.3f\n",
            ifov, 100*cf[ifov], rsh[jsp[1]], rsh[jsp[2]], osh[jsp[1]], osh[jsp[2]])
    for k in 1:length(ν_SPIKE)
        push!(rs[k], rsh[jsp[k]]); push!(os[k], osh[jsp[k]])
    end
end

@printf("\n── across %d FOVs (mean ± std) ──\n", length(order))
for k in 1:length(ν_SPIKE)
    @printf("  ν=%.2f  resSharp = %+.3f ± %.3f K   obsSharp = %+.3f ± %.3f K\n",
            ν_SPIKE[k], mean(rs[k]), std(rs[k]), mean(os[k]), std(os[k]))
end
@printf("\nINTERPRETATION: std ≪ |mean| ⇒ SYSTEMATIC (instrument/model). std ≳ |mean| ⇒ ATMOSPHERIC.\n")
