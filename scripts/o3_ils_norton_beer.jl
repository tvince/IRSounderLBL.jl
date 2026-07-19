# ILS-shape sensitivity probe for the 715.5/715.75 spike.
#
# CAVEAT (see project_reduced_vmr_joint): the IASI ILS is fixed by design =
# sinc(finite 2 cm OPD) ⊗ applied Gaussian (fwhm 0.5). Norton-Beer is a DIFFERENT
# apodization IASI does not use, so this does NOT test "is our ILS right" — it is a
# deliberately-wrong shape swap to measure how much the spike can move under a gross
# ILS-shape change at fixed resolution. If it barely moves, no ILS-shape error can
# produce the +3 K spike (shape comes off the board); if it moves a lot, ILS shape is
# at least a sensitive lever.
#
# Reconstructs the retrieved joint state, runs the forward once per apodization.
#
#   julia -t auto --project=. scripts/o3_ils_norton_beer.jl

using IRSounderLBL
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const DNU = 0.001           # converged production grid on this steep edge
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const APODS = (:gaussian, :norton_beer_weak, :norton_beer_medium, :norton_beer_strong)

T0 = time(); prog(m) = (@printf("[%6.1fs] %s\n", time()-T0, m); flush(stdout))

prog("Loading baseline y + inputs …")
bl = load("data/iasi_profile_retrieval.jld2"); νobs = bl["ν"]; y = bl["y"]
base = afgl_atmosphere(:us_standard)
base.vmr[CO2] .*= (CO2_PPM*1e-6)/base.vmr[CO2][1]
nlev = n_levels(base)
co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > 1e-23])
linelists = Dict(CO2=>co2, H2O=>h2o, O3=>o3)
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", ν_LO, ν_HI; stot_min=0.0)
lm = VPYLineMixing(relmat)

prog("Reconstructing retrieved joint state …")
J = load("data/iasi_joint.jld2"); rJ = J["rJ"]; blocks = J["blocks"]
B_h2o = partial_column_basis(nlev, blocks; taper=:boxcar)
spec  = StateVectorSpec(nlev, [PartialColumns(H2O, B_h2o), ColumnScale(O3)];
                        include_temperature=true, include_tsfc=true, include_emissivity=false)
prof, T_sfc, _ = unpack_state(spec, rJ.x, base)

nchan = round(Int, (ν_HI-ν_LO)/0.25)+1
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0,5.0), sralim=(15.0,180.0), zenlim=(0.0,25.0), max_fov=400)
geom = ViewingGeometry(g.zen[argmin(cloud_fraction(g))])

fwd(apod) = iasi_forward_model(prof, linelists; iasi=iasi, geom=geom, T_sfc=T_sfc, ε_sfc=ε_SEA,
              internal_dnu=DNU, with_ils=true, apodization=apod, apply_continuum=true,
              line_mixing=lm, lm_cutoff=LM_CUTOFF)[3]

chans = (715.25, 715.50, 715.75, 716.00, 725.25, 725.50, 725.75, 726.25)
jj = [argmin(abs.(νobs .- c)) for c in chans]

BT = Dict{Symbol,Vector{Float64}}()
for a in APODS
    prog("Forward @ apodization=$(a) …")
    BT[a] = fwd(a)
end

println("\n" * "="^100)
println("BT (K) at spike channels vs apodization  (obs, then each ILS shape)")
println("="^100)
@printf("%-9s %9s", "chan", "obs"); for a in APODS; @printf(" %14s", String(a)); end; println()
for (c,j) in zip(chans,jj)
    @printf("%-9.2f %9.4f", c, y[j])
    for a in APODS; @printf(" %14.4f", BT[a][j]); end; println()
end

println("\n" * "="^100)
println("residual (model − obs) vs apodization, and Δ from the Gaussian (true IASI ILS)")
println("="^100)
@printf("%-9s", "chan"); for a in APODS; @printf(" %11s", String(a)); end
for a in APODS[2:end]; @printf(" %13s", "Δ("*String(a)[13:end]*"−G)"); end; println()
for (c,j) in zip(chans,jj)
    @printf("%-9.2f", c)
    for a in APODS; @printf(" %+11.4f", BT[a][j]-y[j]); end
    for a in APODS[2:end]; @printf(" %+13.4f", BT[a][j]-BT[:gaussian][j]); end
    println()
end

# Band-wide magnitude of the shape swap, for context.
w = (νobs .>= 710) .& (νobs .<= 730)
println()
for a in APODS[2:end]
    @printf("710–730 max|Δ(%s − Gaussian)| = %.3f K\n", String(a),
            maximum(abs.(BT[a][w] .- BT[:gaussian][w])))
end
prog("done")
