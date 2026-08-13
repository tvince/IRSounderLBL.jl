# Width analog of o3_reachability.jl. The lineshape audit found the widths have real room
# (γ_air 2-5%, some 10-20%; n_air 5-10%) and a J-dependence separating the two clusters
# (715 mid-J broader, 722-723 high-J narrower). Question: can a PHYSICALLY-SMALL γ_air
# change cancel the 710-730 residual — and does it need to be J-dependent (the dipole
# signature) rather than a uniform scale (which, like O3 amount, should move both the same
# way)?
#
# Build two width-perturbation BT-sensitivity vectors by finite difference at the RETRIEVED
# state: (1) UNIFORM γ_air scale; (2) J-DEPENDENT slope via E'' (widen high-E''/high-J,
# narrow low-E''/low-J). Whiten by NEΔT, project the residual onto span{v_uniform, v_slope}
# over 710-730. Report fraction explained and the implied γ_air change (must be within
# ~2-5% to be physical).
#
#   julia -t auto --project=. scripts/o3_width_reachability.jl

using IRSounderLBL
using LinearAlgebra: diag, svd, Diagonal, qr
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const BASELINE = "data/iasi_profile_retrieval.jld2"
const JOINT    = "data/iasi_joint.jld2"
const FITCSV   = "data/iasi_joint_fit.csv"
const GRANULE  = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const ν_PTS = (715.75, 722.75, 723.25)

bl = load(BASELINE); νobs = bl["ν"]; y = bl["y"]; Se = bl["Se"]
nedt = sqrt.(diag(Se))
rJ = load(JOINT)["rJ"]
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
prof, T_sfc, _ = unpack_state(rJ.spec, rJ.x, base)

fit = Dict{Float64,Float64}()
for (i, line) in enumerate(eachline(FITCSV))
    i == 1 && continue
    p = split(line, ","); fit[parse(Float64, p[1])] = parse(Float64, p[4])
end
res = [fit[round(ν, digits=4)] for ν in round.(νobs, digits=4)]

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > O3_SMIN])
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom = ViewingGeometry(g.zen[ifov])
fmkw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
        apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF,
        T_sfc=T_sfc, ε_sfc=ε_SEA)

# γ_air-perturbed O3 linelist: air_broad *= factor(line)
scaleγ(ll, fac) = HITRANLinelist([HITRANLine(l.mol_id, l.iso_id, l.wavenumber, l.intensity,
    l.a_coeff, Float32(l.air_broad * fac(l)), l.self_broad, l.lower_energy,
    l.temp_depend, l.pressure_shift) for l in ll.lines])
bt(o3ll) = iasi_forward_model(prof, Dict(CO2=>co2, H2O=>h2o, O3=>o3ll); fmkw...)[3]

# E'' centering for the J-dependent (slope) perturbation, over the residual window
Ewin = [l.lower_energy for l in o3.lines if 710.0 <= l.wavenumber <= 730.0]
Ē = sum(Ewin)/length(Ewin); Espan = maximum(Ewin) - minimum(Ewin)
pslope(l) = (l.lower_energy - Ē) / Espan          # ~[-0.5,0.5] across the window

const ε = 0.05                                    # 5% FD step
@printf("Baseline forward + 2 width-perturbed forwards …\n"); flush(stdout)
b0 = bt(o3)
bU = bt(scaleγ(o3, l -> 1.0 + ε))                 # uniform +5%
bS = bt(scaleγ(o3, l -> 1.0 + ε * pslope(l)))     # J-dependent slope, ±2.5% edge-to-edge·2.5

vU = (bU .- b0) ./ ε                              # ∂BT / ∂(uniform Δγ/γ)
vS = (bS .- b0) ./ ε                              # ∂BT / ∂(slope amplitude)

@printf("\n── BT sensitivity at the residual points (K per unit perturbation) ──\n")
@printf("  %10s %12s %12s %12s\n", "ν", "residual", "∂BT/uniform", "∂BT/slope")
for νp in ν_PTS
    j = argmin(abs.(νobs .- νp))
    @printf("  %10.2f %+12.3f %+12.3f %+12.3f\n", νobs[j], res[j], vU[j], vS[j])
end

# Whitened projection of the residual onto span{vU, vS} over 710-730
win = (νobs .>= 710.0) .& (νobs .<= 730.0)
rw = res[win] ./ nedt[win]
G  = hcat(vU[win] ./ nedt[win], vS[win] ./ nedt[win])
Q  = Matrix(qr(G).Q)
proj = Q * (Q' * rw)
rms(v) = sqrt(sum(abs2, v)/length(v))
explained = sum(abs2, proj) / sum(abs2, rw)
# least-squares amplitudes (coef .* perturbation units)
coef = G \ rw
@printf("\n── Width reachability over 710-730 (%d channels) ──\n", count(win))
@printf("  residual explained by {uniform γ, J-slope γ}: %.1f%%\n", 100*explained)
@printf("  uniform-only: %.1f%%   ·   slope-only: %.1f%%\n",
        100*sum(abs2, (qU=Matrix(qr(G[:,1:1]).Q); qU*(qU'*rw)))/sum(abs2, rw),
        100*sum(abs2, (qS=Matrix(qr(G[:,2:2]).Q); qS*(qS'*rw)))/sum(abs2, rw))
@printf("  fitted amplitudes: uniform Δγ/γ = %+.3f (%.1f%%)   slope amp = %+.3f\n",
        coef[1], 100*coef[1], coef[2])
@printf("  → implied γ_air change: uniform %.1f%%, plus edge-to-edge J-slope of ~%.1f%%\n",
        100*abs(coef[1]), 100*abs(coef[2]))
@printf("  (HITRAN γ_air uncertainty here is 2-5%%, up to 10-20%% for ~8%% of intensity)\n")
@printf("  whitened residual RMS in-window: %.2f σ ; after width fit: %.2f σ\n",
        rms(rw), rms(rw .- proj))
