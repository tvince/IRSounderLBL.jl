"""
Export Julia BT spectra for the 4.3 µm CO₂ band, matching the LBLRTM 4.3 µm
validation run (CO₂ ν₃ fundamental, 2000–2500 cm⁻¹).

Mirrors the LBLRTM TAPE5/TAPE3 setup:
  - CO₂ only, all 3 isotopologues (CO2-only TAPE3, 61,748 lines, COUPLED=0)
  - line mixing OFF (pure Voigt)
  - AFGL US Standard 50-level profile, nadir from TOA, blackbody surface
  - no ILS, fine monochromatic grid Δν = 0.005 cm⁻¹ (high_res_factor = 1)

Two runs, to mirror the two LBLRTM configs:
  - cont-OFF        (apply_continuum=false)        -> julia_bt_43um_contOFF.csv
  - CO₂-continuum   (continua=(:co2,) only)         -> julia_bt_43um_co2cont.csv
The second isolates the MT-CKD CO₂ continuum (with the XFACCO2 + bandhead
corrections) — matching LBLRTM ICNTNM=6 with XCO2C=1 — so the N₂/O₂ continua
(different datasets between the two codes near 2330 cm⁻¹) don't contaminate.

Run with:
  julia --project -t auto scripts/julia_bt_43um_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2000.0
const NU_MAX  = 2500.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

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

function run_and_write(outfile; apply_cont, continua)
    @printf("\nForward model: apply_continuum=%s continua=%s …\n", apply_cont, continua)
    t = time()
    ν_out, _, BT = iasi_forward_model(prof, linelists;
                                       iasi            = iasi,
                                       high_res_factor = 1,
                                       cutoff          = CUTOFF,
                                       apply_continuum = apply_cont,
                                       continua        = continua,
                                       with_ils        = false,
                                       line_mixing     = nothing)
    @printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
            time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))
    open(outfile, "w") do f
        write(f, "nu_cm1,BT_K\n")
        for (ν, bt) in zip(ν_out.ν, BT)
            write(f, "$(round(ν, digits=5)),$(round(bt, digits=6))\n")
        end
    end
    println("  wrote ", outfile)
end

run_and_write(joinpath(OUTDIR, "julia_bt_43um_contOFF.csv");
              apply_cont=false, continua=())
run_and_write(joinpath(OUTDIR, "julia_bt_43um_co2cont.csv");
              apply_cont=true,  continua=(:co2,))
println("\nDone.")
