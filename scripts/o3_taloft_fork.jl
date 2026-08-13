# Upper-stratosphere T fork for the strong-O3-core residual family.
#
# The steepness test (scripts/o3_steepness_test.py) showed the 715.5/715.75 (+10σ) and
# 724.75-726.5 (+5-7σ) residuals are a family localized to the STRONGEST O3 line cores,
# all POSITIVE = model under-absorbs / runs too warm where those cores reach τ≈1 (high,
# cold ~1-50 hPa). Two survivors predict OPPOSITE signs:
#   • warm-T bias at the core-emitting altitude  → + residual  (THIS test)
#   • SDV/Dicke core narrowing                    → − residual  (wrong sign; separate test)
#
# Reconstruct the retrieved joint state, then perturb T by a fixed ΔT in several
# upper-atmosphere PRESSURE BANDS, re-run the forward, and measure ∂BT/∂T at the spike
# channels. Decisive number: how many K of cooling would cancel the +3 K spike, and is
# that within the retrieval's plausible T-aloft uncertainty (σ_T=5 K prior)?
#
#   julia -t auto --project=. scripts/o3_taloft_fork.jl

using IRSounderLBL
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const DNU = 0.0025          # differences of BT are grid-insensitive (spike is converged)
const ΔT  = -2.0            # cooling perturbation (the direction that would REDUCE a + residual)
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
# Upper-atmosphere pressure bands (hPa) to perturb, plus an all-above-100 lump.
const BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
               (20.0, 50.0), (50.0, 100.0), (0.0, 100.0))

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

fwd(p) = iasi_forward_model(p, linelists; iasi=iasi, geom=geom, T_sfc=T_sfc, ε_sfc=ε_SEA,
            internal_dnu=DNU, with_ils=true, apodization=:gaussian, apply_continuum=true,
            line_mixing=lm, lm_cutoff=LM_CUTOFF)[3]

prog("Baseline forward …")
BT0 = fwd(prof)

chans = (715.25, 715.50, 715.75, 725.25, 725.50, 725.75, 726.25)
jj = [argmin(abs.(νobs .- c)) for c in chans]
res0 = [BT0[j]-y[j] for j in jj]

# Perturb T by ΔT within each pressure band; measure ΔBT and per-K sensitivity.
println("\nRetrieved T(p) at the bands of interest:")
for (lo,hi) in BANDS[1:6]
    lv = findall(p -> lo < p <= hi, base.pressure)
    isempty(lv) && continue
    @printf("  %5.1f–%5.1f hPa: levels %s  T=%.1f–%.1f K\n", lo, hi,
            string(first(lv))*"-"*string(last(lv)),
            minimum(prof.temperature[lv]), maximum(prof.temperature[lv]))
end

results = Dict{Tuple{Float64,Float64}, Vector{Float64}}()
for (lo,hi) in BANDS
    lv = findall(p -> lo < p <= hi, base.pressure)
    isempty(lv) && continue
    Tp = copy(prof.temperature); Tp[lv] .+= ΔT
    pp = AtmosphericProfile(copy(base.pressure), Tp, copy(base.altitude), deepcopy(prof.vmr))
    prog(@sprintf("Forward: %+.0f K in %.0f–%.0f hPa (%d levels) …", ΔT, lo, hi, length(lv)))
    BTp = fwd(pp)
    results[(lo,hi)] = [BTp[j]-BT0[j] for j in jj]   # ΔBT from the perturbation
end

# ── Report ──────────────────────────────────────────────────────────────────────────
println("\n" * "="^92)
@printf("∂BT/∂T from %+.0f K cooling in each band — ΔBT per channel (K), and K-to-cancel the residual\n", ΔT)
println("="^92)
@printf("%-9s %8s", "band(hPa)", "")
for c in chans; @printf(" %8.2f", c); end; println()
@printf("%-9s %8s", "baseline", "res₀→"); for r in res0; @printf(" %+8.3f", r); end; println()
for (lo,hi) in BANDS
    haskey(results,(lo,hi)) || continue
    @printf("%-4.0f-%-4.0f %8s", lo, hi, "ΔBT")
    for d in results[(lo,hi)]; @printf(" %+8.3f", d); end; println()
    # sensitivity per K = ΔBT/ΔT ; K-to-cancel = res₀ / (per-K sens)
    @printf("%-9s %8s", "", "∂/∂T")
    for d in results[(lo,hi)]; @printf(" %+8.3f", d/ΔT); end; println()
end

# Decisive summary for the two lead spike channels.
println("\n" * "="^92)
println("VERDICT — how much upper-strat cooling to cancel the +3 K spike (715.5) & 715.75?")
println("="^92)
band_all = (0.0, 100.0)
for (c, r0, k) in zip(chans, res0, 1:length(chans))
    (c in (715.50, 715.75)) || continue
    sens = results[band_all][k] / ΔT             # K(BT) per K(T) over the whole 0-100 hPa column
    need = sens != 0 ? r0 / sens : Inf
    @printf("  %.2f: res₀=%+.3f K, ∂BT/∂T(0-100hPa)=%.3f K/K ⇒ need ΔT=%+.1f K to cancel\n",
            c, r0, sens, -need)
end
prog("done")
