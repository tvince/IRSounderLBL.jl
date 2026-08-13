# Line culprit inventory for the 710–730 opposite-sign O3 residual (joint retrieval).
#
# Two diagnostics at IASI FOV #1:
#   (A) TEMPERATURE-ADJUSTED LINE CENSUS — for each target channel (715.75 wants MORE
#       O3, 722.75/723.25 want LESS), list the strongest lines per species/isotope
#       within ±1.0 cm⁻¹, with S(296K), S adjusted to a representative T, E'' and
#       broadening. Shows WHAT physically sits at each residual feature.
#   (B) SPECIES-TOGGLE BT ATTRIBUTION — run the forward model at the RETRIEVED joint
#       atmosphere with CO2-only, +O3, +H2O toggles; the per-species ΔBT across
#       710–730 shows which species produces the absorption at 715.75 vs 722.75.
#
#   julia -t auto --project=. scripts/inventory_715.jl

using IRSounderLBL
const IRS = IRSounderLBL
using Printf
using JLD2: load

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const LM_CUTOFF = 5.0
const O3_SMIN = 1e-23
const c2 = 1.4387769             # cm·K  (second radiation constant)
const T_REF = 296.0
const T_CENSUS = 240.0           # representative weighting-function temperature, 710–730 band
const TARGETS = (715.75, 722.75, 723.25)
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

# HITRAN S(T) from S(296): TIPS ratio × Boltzmann(E'') × stimulated-emission factor.
function S_at(l, T)
    qr = IRS.Q_ratio(Int(l.mol_id), Int(l.iso_id), T)      # Q(296)/Q(T)
    boltz = exp(-c2 * l.lower_energy * (1/T - 1/T_REF))
    stim  = (1 - exp(-c2 * l.wavenumber / T)) / (1 - exp(-c2 * l.wavenumber / T_REF))
    return l.intensity * qr * boltz * stim
end

spname(s) = IRS.SPECIES_NAME[s]

println("="^78)
println("(A)  TEMPERATURE-ADJUSTED LINE CENSUS  (S adjusted 296 K → $(Int(T_CENSUS)) K)")
println("="^78)

lls = Dict(
    CO2 => load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25),
    H2O => load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25),
    O3  => load_linelist("data/o3_645_2760",  1:4; ν_min=ν_LO-25, ν_max=ν_HI+25),
)

for νt in TARGETS
    @printf("\n── target ν = %.2f cm⁻¹  (±1.0 cm⁻¹ window) ──────────────────────\n", νt)
    rows = NamedTuple[]
    for (sp, ll) in lls
        for l in ll.lines
            abs(l.wavenumber - νt) <= 1.0 || continue
            push!(rows, (sp=sp, iso=Int(l.iso_id), ν=l.wavenumber,
                         S296=l.intensity, ST=S_at(l, T_CENSUS),
                         E=l.lower_energy, air=Float64(l.air_broad)))
        end
    end
    isempty(rows) && (println("   (no lines within ±1.0 cm⁻¹)"); continue)
    sort!(rows, by=r->-r.ST)
    @printf("   %-5s iso  %9s   %10s %10s  %9s  %6s\n",
            "spec", "ν", "S(296)", "S($(Int(T_CENSUS)))", "E''", "γair")
    for r in rows[1:min(10, length(rows))]
        @printf("   %-5s  %d  %9.4f   %.3e %.3e  %8.2f  %.4f\n",
                spname(r.sp), r.iso, r.ν, r.S296, r.ST, r.E, r.air)
    end
    # per-species Σ S(T) + intensity-weighted mean E'' — "who dominates & at what altitude"
    for sp in (CO2, O3, H2O)
        srows = [r for r in rows if r.sp === sp]
        tot = sum(r.ST for r in srows; init=0.0)
        Ebar = tot > 0 ? sum(r.ST * r.E for r in srows) / tot : 0.0
        @printf("     Σ S(%d) %-4s = %.3e   (%d lines)   ⟨E''⟩_S = %6.1f cm⁻¹\n",
                Int(T_CENSUS), spname(sp), tot, length(srows), Ebar)
    end
end

# ── (B) species-toggle BT attribution at the retrieved joint atmosphere ─────────────
if get(ENV, "SKIP_B", "") != ""
    println("\n(B) skipped (SKIP_B set)")
    exit()
end
println("\n", "="^78)
println("(B)  SPECIES-TOGGLE BT ATTRIBUTION  at the retrieved joint atmosphere")
println("="^78)

rJ = load("data/iasi_joint.jld2")["rJ"]                       # VP_Y joint solution
base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM * 1e-6) / base.vmr[CO2][1]
prof, T_sfc, _ = unpack_state(rJ.spec, rJ.x, base)            # retrieved profile
@printf("retrieved T_sfc=%.2f K  ·  O3 col scale=%.3f×\n",
        T_sfc, exp(rJ.x[vmr_range(rJ.spec, O3)][1]))

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
iasi  = IASIInstrument(ν_LO, ν_HI, 0.25, nchan, 2.0, 0.5)
geom  = ViewingGeometry(g.zen[ifov])
fmkw = (iasi=iasi, geom=geom, with_ils=true, apodization=:gaussian,
        apply_continuum=true, internal_dnu=0.0025, line_mixing=lm, lm_cutoff=LM_CUTOFF,
        T_sfc=T_sfc, ε_sfc=ε_SEA)

run_bt(lldict) = iasi_forward_model(prof, lldict; fmkw...)[3]

νg_raw, _, _ = iasi_forward_model(prof, Dict(CO2=>co2); fmkw...)   # grid
νg = νg_raw isa WavenumberGrid ? νg_raw.ν : collect(Float64, νg_raw)
bt_co2   = run_bt(Dict(CO2=>co2))
bt_co2o3 = run_bt(Dict(CO2=>co2, O3=>o3))
bt_co2h2 = run_bt(Dict(CO2=>co2, H2O=>h2o))
bt_all   = run_bt(Dict(CO2=>co2, H2O=>h2o, O3=>o3))

@printf("\n%-9s %8s %8s %8s %8s %9s %9s\n",
        "ν", "BT_all", "ΔO3", "ΔH2O", "ΔLM?", "|CO2only", "")
println("  (ΔO3 = BT drop from adding O3; ΔH2O likewise; negative = absorption)")
for νt in TARGETS
    j = argmin(abs.(νg .- νt))
    dO3  = bt_co2o3[j] - bt_co2[j]
    dH2O = bt_co2h2[j] - bt_co2[j]
    @printf("ν=%.2f  all=%.3f  ΔO3=%+.3f K  ΔH2O=%+.3f K  (CO2-only=%.3f)\n",
            νg[j], bt_all[j], dO3, dH2O, bt_co2[j])
end

# full 710–730 dump for the plot
open("data/inventory_715.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_all_K,bt_co2_K,dO3_K,dH2O_K")
    for j in eachindex(νg)
        710.0 <= νg[j] <= 730.0 || continue
        @printf(io, "%.4f,%.4f,%.4f,%.4f,%.4f\n",
                νg[j], bt_all[j], bt_co2[j], bt_co2o3[j]-bt_co2[j], bt_co2h2[j]-bt_co2[j])
    end
end
println("\nwrote data/inventory_715.csv (710–730 per-species ΔBT)")
