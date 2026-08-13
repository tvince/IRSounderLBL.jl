# Is the S>1e-23 O3 trim BT-lossless in the 710-730 residual window? Forward BT at the
# retrieved column-joint atmosphere with three O3 line sets: production trim (S>1e-23),
# near-full (S>1e-25), full (all iso 1-4). Diff at the residual points + 710-730 RMS.
using IRSounderLBL
const IRS = IRSounderLBL
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const ν_PTS = (715.75, 722.75, 723.25)

rJ = load("data/iasi_joint.jld2")["rJ"]
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
prof, T_sfc, _ = unpack_state(rJ.spec, rJ.x, base)

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
trim(s) = HITRANLinelist([l for l in o3f.lines if l.intensity > s])
o3_prod = trim(1e-23)      # production
o3_fine = trim(1e-25)      # near-full
o3_full = o3f              # everything (iso 1-4, all strengths)
@printf("O3 lines: prod(S>1e-23)=%d · fine(S>1e-25)=%d · full=%d\n",
        length(o3_prod.lines), length(o3_fine.lines), length(o3_full.lines))

relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fmkw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
        apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF,
        T_sfc=T_sfc, ε_sfc=ε_SEA)

bt(o3) = iasi_forward_model(prof, Dict(CO2=>co2, H2O=>h2o, O3=>o3); fmkw...)[3]
νg_raw, _, _ = iasi_forward_model(prof, Dict(CO2=>co2); fmkw...)
νg = νg_raw isa WavenumberGrid ? νg_raw.ν : collect(Float64, νg_raw)
b_prod = bt(o3_prod); b_fine = bt(o3_fine); b_full = bt(o3_full)

win = (νg .>= 710) .& (νg .<= 730)
rms(v) = sqrt(sum(abs2, v)/length(v))
@printf("\n710-730 BT change from RESTORING dropped O3 lines:\n")
@printf("  prod→fine (add S∈[1e-25,1e-23]): rms=%.4f K  max|Δ|=%.4f K\n",
        rms((b_fine.-b_prod)[win]), maximum(abs.((b_fine.-b_prod)[win])))
@printf("  prod→full (add everything)     : rms=%.4f K  max|Δ|=%.4f K\n",
        rms((b_full.-b_prod)[win]), maximum(abs.((b_full.-b_prod)[win])))
@printf("\nAt the residual points (ΔBT = full − prod; NEGATIVE = trim was UNDER-absorbing):\n")
for νp in ν_PTS
    j = argmin(abs.(νg .- νp))
    @printf("  ν=%.2f:  fine−prod=%+.4f K   full−prod=%+.4f K\n",
            νg[j], b_fine[j]-b_prod[j], b_full[j]-b_prod[j])
end
open("data/o3_trim_bt.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_prod_K,d_fine_K,d_full_K")
    for j in eachindex(νg)
        710.0 <= νg[j] <= 730.0 || continue
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", νg[j], b_prod[j], b_fine[j]-b_prod[j], b_full[j]-b_prod[j])
    end
end
println("\nwrote data/o3_trim_bt.csv")
