# Which species owns the 715.5/715.75 feature — O₃ or CO₂?
#
# At 715 cm⁻¹ we sit in BOTH the O₃ ν₂ R-branch and the CO₂ ν₂ R-branch. CO₂'s
# column is ~1150× O₃'s, so a much weaker CO₂ line (per molecule) can still dominate
# the optical depth. Rank the lines in 714–717 cm⁻¹ by HITRAN intensity AND by an
# abundance-weighted proxy S·N_col (the real driver of nadir τ).
#
#   julia --project=. scripts/o3_vs_co2_715_inventory.jl

using IRSounderLBL
using Printf

const ν_LO, ν_HI = 645.0, 800.0
# Full-atmosphere column densities (molec/cm²), order-of-magnitude:
#   N_air ≈ 2.15e25 ;  CO₂ = 432 ppm ⇒ 9.3e21 ;  O₃ ≈ 300 DU = 8.07e18
const NCOL = Dict(:CO2 => 432e-6 * 2.15e25, :O3 => 300 * 2.69e16, :H2O => 0.01 * 2.15e25)

co2 = load_linelist("data/co2_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3f = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
o3  = HITRANLinelist([l for l in o3f.lines if l.intensity > 1e-23])

function window(ll, lo, hi)
    [l for l in ll.lines if lo <= l.wavenumber <= hi]
end

for (lo, hi) in ((715.0, 716.0), (714.0, 717.0))
    println("\n" * "="^92)
    @printf("Lines in %.1f–%.1f cm⁻¹, ranked by abundance-weighted S·N_col  (the τ driver)\n", lo, hi)
    println("="^92)
    rows = Tuple{Float64,String,Float64,Float64}[]   # (ν, sp, S, S·N)
    for (nm, ll, key) in (("CO₂", co2, :CO2), ("O₃", o3, :O3), ("H₂O", h2o, :H2O))
        for l in window(ll, lo, hi)
            push!(rows, (l.wavenumber, nm, l.intensity, l.intensity * NCOL[key]))
        end
    end
    sort!(rows, by = r -> -r[4])
    @printf("%-10s %-5s %14s %14s %10s\n", "ν (cm⁻¹)", "sp", "S (HITRAN)", "S·N_col", "share%")
    tot = sum(r[4] for r in rows)
    for r in rows[1:min(end, 14)]
        @printf("%-10.4f %-5s %14.3e %14.3e %9.1f\n", r[1], r[2], r[3], r[4], 100*r[4]/tot)
    end
    # per-species totals in the window
    println("  ---- window totals (Σ S·N_col) ----")
    for nm in ("CO₂", "O₃", "H₂O")
        s = sum((r[4] for r in rows if r[2]==nm); init=0.0)
        @printf("    %-4s : %.3e  (%.1f%%)\n", nm, s, 100*s/tot)
    end
end
