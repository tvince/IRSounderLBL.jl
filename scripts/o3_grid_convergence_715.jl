# Grid-resolution convergence of the 715.5/715.75 spike.
#
# The joint retrieval (retrieve_iasi_joint.jl) ran at internal_dnu=0.0025 cm⁻¹.
# 715.5 is the steepest spectral edge in the band, where a coarse internal grid
# would show its worst sampling error. This reconstructs the retrieved joint state
# and runs ONE forward model at each of {0.0025, 0.001, 0.0005}, all other forward
# settings identical, to see whether the +2 K spike is grid-converged.
#
#   julia -t auto --project=. scripts/o3_grid_convergence_715.jl

using IRSounderLBL
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const DNUS = (0.0025, 0.001, 0.0005)

T0 = time()
prog(msg) = (@printf("[%6.1fs] %s\n", time() - T0, msg); flush(stdout))

# ── Inputs (mirror the driver) ─────────────────────────────────────────────────────
prog("Loading baseline y …")
bl = load("data/iasi_profile_retrieval.jld2")
νobs = bl["ν"]; y = bl["y"]

prog("Base atmosphere + linelists + relmat …")
base = afgl_atmosphere(:us_standard)
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
nlev = n_levels(base)

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3_full = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3 = HITRANLinelist([l for l in o3_full.lines if l.intensity > 1e-23])
linelists = Dict(CO2 => co2, H2O => h2o, O3 => o3)

relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)

# ── Reconstruct the retrieved joint state ──────────────────────────────────────────
prog("Reconstructing retrieved joint state from data/iasi_joint.jld2 …")
J = load("data/iasi_joint.jld2")
rJ = J["rJ"]; blocks = J["blocks"]
B_h2o = partial_column_basis(nlev, blocks; taper=:boxcar)
spec  = StateVectorSpec(nlev, [PartialColumns(H2O, B_h2o), ColumnScale(O3)];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
prof, T_sfc, _ = unpack_state(spec, rJ.x, base)
@printf("  O₃ scale=%.3f×  H₂O scales=%s  T_sfc=%.2f K\n",
        exp(rJ.x[vmr_range(spec,O3)][1]),
        string(round.(exp.(rJ.x[vmr_range(spec,H2O)]); digits=3)), T_sfc); flush(stdout)

nchan = round(Int, (ν_HI - ν_LO)/0.25) + 1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
geom = ViewingGeometry(g.zen[ifov])

# ── Forward at each resolution ──────────────────────────────────────────────────────
chans = (715.25, 715.50, 715.75, 716.00)
jidx  = [argmin(abs.(νobs .- c)) for c in chans]
BTs = Dict{Float64, Vector{Float64}}()
for dnu in DNUS
    prog("Forward @ internal_dnu=$(dnu) …")
    @time _, _, BT = iasi_forward_model(prof, linelists; iasi=iasi, geom=geom,
                        T_sfc=T_sfc, ε_sfc=ε_SEA, internal_dnu=dnu,
                        with_ils=true, apodization=:gaussian, apply_continuum=true,
                        line_mixing=lm, lm_cutoff=LM_CUTOFF)
    BTs[dnu] = BT
end

# ── Report ──────────────────────────────────────────────────────────────────────────
println("\n===== BT (K) at the spike channels vs internal_dnu =====")
@printf("%-9s %10s %10s %10s %10s\n", "chan", "obs", "dnu=2.5e-3", "1e-3", "5e-4")
for (c, j) in zip(chans, jidx)
    @printf("%-9.2f %10.4f %10.4f %10.4f %10.4f\n",
            c, y[j], BTs[0.0025][j], BTs[0.001][j], BTs[0.0005][j])
end
println("\n===== residual (model − obs), and grid deltas =====")
@printf("%-9s %10s %10s %10s | %12s %12s\n",
        "chan", "res@2.5e-3", "res@1e-3", "res@5e-4", "Δ(1e-3−2.5)", "Δ(5e-4−1e-3)")
for (c, j) in zip(chans, jidx)
    r25 = BTs[0.0025][j]-y[j]; r10 = BTs[0.001][j]-y[j]; r05 = BTs[0.0005][j]-y[j]
    @printf("%-9.2f %10.4f %10.4f %10.4f | %+12.4f %+12.4f\n",
            c, r25, r10, r05, BTs[0.001][j]-BTs[0.0025][j], BTs[0.0005][j]-BTs[0.001][j])
end

# Band-wide grid sensitivity for context (max |ΔBT| over 710-730).
w = (νobs .>= 710) .& (νobs .<= 730)
@printf("\n710–730 max|ΔBT|: (1e-3 − 2.5e-3)=%.4f K   (5e-4 − 1e-3)=%.4f K\n",
        maximum(abs.(BTs[0.001][w] .- BTs[0.0025][w])),
        maximum(abs.(BTs[0.0005][w] .- BTs[0.001][w])))
prog("done")
