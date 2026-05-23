"""
Export Julia BT with H2O MT-CKD continuum for comparison with ARTS.

Same setup as julia_bt_export.jl but with APPLY_CONTINUUM = true.
Output: data/julia_bt_cont.csv  (do NOT overwrite julia_bt_645_800.csv).

Run with:
  julia --project -t auto scripts/julia_bt_export_cont.jl
"""

using IRSounderLBL
using Printf

const WITH_ILS        = false
const APPLY_CONTINUUM = true
const CUTOFF          = 25.0
const HRF             = 4

t0 = time()

function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || (println("  SKIP missing: $fname"); continue)
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(all_lines)
end

println("Loading HITRAN linelists...")
linelists = Dict{GasSpecies, HITRANLinelist}(
    CO2 => load_multi(1:3,  645.0, 2760.0, "co2_645_2760"),
    H2O => load_multi(1:3,  645.0, 2760.0, "h2o_645_2760"),
    O3  => load_multi(1:3,  980.0, 1090.0, "o3_980_1090"),
    N2O => load_multi(1:3, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => load_multi(1:3, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => load_multi(1:3, 2000.0, 2280.0, "co_2000_2280"),
)
for (sp, ll) in sort(collect(linelists), by=x->string(x[1]))
    @printf("  %-4s: %d lines\n", string(sp), length(ll))
end

prof = afgl_us_standard_50lev()

println("Running forward model (continuum ON)...")
t1 = time()
ν_iasi, _, BT_iasi = iasi_forward_model(prof, linelists;
    high_res_factor  = HRF,
    cutoff           = CUTOFF,
    apply_continuum  = APPLY_CONTINUUM,
    with_ils         = WITH_ILS)
@printf("  τ + RTE done in %.1f s\n", time() - t1)

@printf("BT range: %.1f – %.1f K  (n=%d)\n",
        minimum(BT_iasi), maximum(BT_iasi), length(BT_iasi))

outpath = joinpath("data", "julia_bt_cont.csv")
open(outpath, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_iasi.ν, BT_iasi)
        write(f, "$(round(ν, digits=4)),$(round(bt, digits=6))\n")
    end
end
println("Saved → $outpath  ($(length(BT_iasi)) channels)")
@printf("Total: %.1f s  (%d threads)\n", time() - t0, Threads.nthreads())
