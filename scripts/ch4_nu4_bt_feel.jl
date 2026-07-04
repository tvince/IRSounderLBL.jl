# Scoping: how much can CH4 ν₄ line mixing move IASI BT for an Earth scene?
# We don't have the CH4 relaxation matrix yet, so we BOUND the LM effect:
# line mixing only redistributes intensity WITHIN the ν₄ Q-branch, so its BT
# effect is at most a fraction of CH4's own ν₄ Q-branch BT signature. This run
# measures that signature (CH4 on vs off) in a realistic IASI Earth scene, and
# locates/sizes the Q-branch feature so we know which channels & what ceiling.
using IRSounderLBL, Printf
const CUTOFF = 25.0

function load_sp(fnbase, lo, hi)
    L = HITRANLine[]
    for suf in ("", "_iso2", "_iso3")
        p = joinpath("data", "$(fnbase)$(suf).par"); isfile(p) || continue
        ll = try
            load_hitran_par(p; ν_min=lo-CUTOFF, ν_max=hi+CUTOFF)
        catch                       # empty in-window ⇒ skip this isotope file
            continue
        end
        append!(L, ll.lines)
    end
    HITRANLinelist(L)
end

lo, hi = 1240.0, 1360.0           # ν₄ Q-branch ~1306 cm⁻¹, IASI band 2
prof = afgl_us_standard_50lev()

lls = Dict{GasSpecies, HITRANLinelist}(
    CH4 => load_sp("ch4_1200_1800", lo, hi),
    H2O => load_sp("h2o_645_2760",  lo, hi),
    N2O => load_sp("n2o_1200_2310", lo, hi),
    CO2 => load_sp("co2_645_2760",  lo, hi),
)
@printf("lines: CH4 %d  H2O %d  N2O %d  CO2 %d\n",
        length(lls[CH4].lines), length(lls[H2O].lines),
        length(lls[N2O].lines), length(lls[CO2].lines))

n_ch = round(Int, (hi - lo) / 0.25) + 1          # IASI L1C 0.25 cm⁻¹ sampling
iasi = IASIInstrument(lo, hi, 0.25, n_ch, 2.0, 0.5)
common = (iasi=iasi, cutoff=CUTOFF, apply_continuum=true,
          with_ils=true, apodization=:gaussian, internal_dnu=0.001)

println("running full scene (all species)...");      flush(stdout)
ν, _, BT_all = iasi_forward_model(prof, lls; common...)

println("running scene without CH4...");              flush(stdout)
lls_noch4 = Dict(k => v for (k, v) in lls if k != CH4)
_, _, BT_noch4 = iasi_forward_model(prof, lls_noch4; common...)

dBT = BT_all .- BT_noch4          # CH4's total BT contribution (negative = cooling)
νv  = ν isa WavenumberGrid ? collect(ν.ν) : collect(ν)
ν   = νv

# Q-branch window 1300-1312; report depth + where.
qmask = (ν .>= 1300.0) .& (ν .<= 1312.0)
imax  = argmax(abs.(dBT))
iqmax = findfirst(qmask)[1] - 1 + argmax(abs.(dBT[qmask]))

@printf("\n=== CH4 ν₄ total BT signature (Earth scene, CH4 on vs off) ===\n")
@printf("band 1240-1360:  max|ΔBT_CH4| = %.2f K at %.2f cm⁻¹\n", maximum(abs.(dBT)), ν[imax])
@printf("Q-branch 1300-1312:  max|ΔBT_CH4| = %.2f K at %.2f cm⁻¹\n",
        maximum(abs.(dBT[qmask])), ν[iqmax])
@printf("Q-branch mean BT-with-CH4 = %.1f K (scene brightness there)\n", sum(BT_all[qmask])/count(qmask))

# Dump the Q-branch BT-with-CH4 and ΔBT so we can eyeball the feature shape.
open("data/ch4_nu4_bt_feel.csv", "w") do io
    println(io, "nu,BT_all,BT_noCH4,dBT_CH4")
    for i in eachindex(ν)
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", ν[i], BT_all[i], BT_noch4[i], dBT[i])
    end
end
println("\nwrote data/ch4_nu4_bt_feel.csv")
