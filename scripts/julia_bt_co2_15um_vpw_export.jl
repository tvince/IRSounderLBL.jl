"""
Export Julia BT (645–800 cm⁻¹) with CO2 full-matrix line mixing (VP_W).

Matches arts_validation_cont_lm.py configuration:
  - H2O MT-CKD continuum ON
  - CO2 VP_W on whitelisted bands (top-5 per isotope in `ISOTOPES`),
    VP_Y dispersive perturbation on all remaining bands.
  - HITRAN Voigt baseline always evaluated on the full CO2 linelist.
  - No ILS, 0.25 cm⁻¹ output grid.

Run with:
  julia --project -t auto scripts/julia_bt_co2_15um_vpw_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN   = 645.0
const NU_MAX   = 800.0
const DNU_OUT  = 0.25
const CUTOFF   = 25.0
const HRF      = 50          # internal Δν = 0.25 / 50 = 0.005 cm⁻¹
const LM_DIR   = "data/Line-mixing_HITRAN2020/data_new"
const ISOTOPES = [1, 2]      # which CO2 isotopologues get VP_W treatment
const N_TOP    = 5           # top-N bands per isotope
const OUT_CSV  = "data/julia_bt_co2_15um_vpw_iso12.csv"

t0 = time()

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU_OUT) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU_OUT, n_ch, 2.0, 0.5)

println("Loading HITRAN relmat (VP_W, $(NU_MIN)–$(NU_MAX) cm⁻¹)...")
@time relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
println("  $(length(relmat.bands)) LM bands loaded")

println("Loading HITRAN linelists...")
function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(all_lines)
end

ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
ll_h2o = load_multi(1:3, NU_MIN, NU_MAX, "h2o_645_2760")
@printf("  CO2: %d lines   H2O: %d lines\n", length(ll_co2), length(ll_h2o))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2, H2O => ll_h2o)
prof = afgl_us_standard_50lev()

lm = VPWLineMixing(relmat; n_top=N_TOP, isotopes=ISOTOPES)
println("VPW whitelist ($(length(lm.whitelist)) bands, isotopes=$(ISOTOPES), n_top=$(N_TOP)):")
for name in sort!(collect(lm.whitelist))
    println("  ", name)
end

println("Running iasi_forward_model with VPWLineMixing …")
t1 = time()
ν_out, _, BT_out = iasi_forward_model(prof, linelists;
                                       iasi            = iasi,
                                       high_res_factor = HRF,
                                       cutoff          = CUTOFF,
                                       apply_continuum = true,
                                       with_ils        = false,
                                       line_mixing     = lm)
@printf("  forward model: %.1f s\n", time() - t1)

open(OUT_CSV, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out.ν, BT_out)
        write(f, "$(round(ν, digits=4)),$(round(bt, digits=6))\n")
    end
end
@printf("Saved %d channels → %s\n", length(BT_out), OUT_CSV)
@printf("BT range: %.1f – %.1f K\n", minimum(BT_out), maximum(BT_out))
@printf("Total: %.1f s  (%d threads)\n", time() - t0, Threads.nthreads())
