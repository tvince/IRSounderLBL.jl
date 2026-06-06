"""
Export Julia BT for the 4.3 µm CO₂ band WITH VP_Y line mixing ON (cont-OFF),
to test whether CO₂ line mixing closes the +7 K / −6.6 K LBL residual at the
2364 cm⁻¹ ν₃ Q-branch / band head seen vs LBLRTM (see project memory:
project_lblrtm_chi_disabled / project_lblrtm_inputs_verified).

Identical grid + atmosphere + linelist to julia_bt_43um_export.jl
(DNU=0.005, hrf=1, CO₂ iso 1-3, AFGL US Std 50lev, no ILS, cont-OFF) so the
output differences directly against julia_bt_43um_contOFF.csv (LM off) and the
LBLRTM cont-OFF spectrum.

  -> data/lblrtm/julia_bt_43um_contOFF_lm.csv

Run with:
  julia --project -t auto scripts/julia_bt_43um_lm_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2000.0
const NU_MAX  = 2500.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const LM_DIR  = "data/Line-mixing_HITRAN2020/data_new"

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

println("Loading HITRAN relmat (VP_Y, $(NU_MIN)–$(NU_MAX) cm⁻¹)…")
@time relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
println("  $(length(relmat.bands)) LM bands loaded")

function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

println("Loading CO2 linelist (iso 1–3)…")
ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
@printf("  CO2: %d lines\n", length(ll_co2))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)
prof = afgl_us_standard_50lev()

println("\nForward model: cont-OFF, VP_Y line mixing ON …")
t = time()
ν_out, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = 1,
                                   cutoff          = CUTOFF,
                                   apply_continuum = false,
                                   continua        = (),
                                   with_ils        = false,
                                   line_mixing     = VPYLineMixing(relmat))
@printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
        time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))

outfile = joinpath(OUTDIR, "julia_bt_43um_contOFF_lm.csv")
open(outfile, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out.ν, BT)
        write(f, "$(round(ν, digits=5)),$(round(bt, digits=6))\n")
    end
end
println("  wrote ", outfile)
println("\nDone.")
