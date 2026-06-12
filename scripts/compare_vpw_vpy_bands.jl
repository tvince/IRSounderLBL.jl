# Compare full-matrix VP_W against first-order VP_Y at IASI BT level over the
# 15 µm (645–800) and 4.3 µm (2000–2500) CO2 bands. cont-OFF, ILS-OFF, CO2 iso 1–3.
#
# ΔBT = BT(VP_W) − BT(VP_Y). The VP_W run gives whitelisted bands the
# eigendecomposition and ALL OTHER bands VP_Y (line_mixing.jl:126), so this
# difference isolates full-matrix vs first-order in the strong (whitelisted)
# bands; everything else cancels exactly.
#
# Output: data/vpw_vpy_<band>.csv  (nu, BT_VPY, BT_VPW, dBT)
# Run: julia --project -t auto scripts/compare_vpw_vpy_bands.jl
using IRSounderLBL, Printf, Statistics

const LM_DIR = "data/Line-mixing_HITRAN2020/data_new"
const CUTOFF = 25.0

function load_co2(ν_min, ν_max)
    all_lines = HITRANLine[]
    for iso in 1:3
        fname = iso == 1 ? "co2_645_2760.par" : "co2_645_2760_iso$(iso).par"
        fpath = joinpath("data", fname); isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min-CUTOFF, ν_max=ν_max+CUTOFF)
        append!(all_lines, ll.lines)
    end
    HITRANLinelist(all_lines)
end

const prof = afgl_us_standard_50lev()

function run_band(name, ν_min, ν_max, dnu_out)
    n_ch = round(Int, (ν_max-ν_min)/dnu_out) + 1
    iasi = IASIInstrument(ν_min, ν_max, dnu_out, n_ch, 2.0, 0.5)
    relmat = load_hitran_relmat(LM_DIR, ν_min, ν_max; stot_min=0.0)
    vpw = VPWLineMixing(relmat; ν_window=(ν_min, ν_max))
    @printf("\n========== %s band (%.0f–%.0f cm⁻¹) ==========\n", name, ν_min, ν_max)
    @printf("LM bands loaded: %d   VP_W whitelist (%d bands): %s\n",
            length(relmat.bands), length(vpw.whitelist), join(sort(collect(vpw.whitelist)), ", "))
    ll = load_co2(ν_min, ν_max)
    lls = Dict{GasSpecies, HITRANLinelist}(CO2 => ll)
    # internal_dnu=0.005: a VP_W−VP_Y *difference* is smooth; the validated 0.001
    # grid is unnecessary here and ~5× slower. Both runs share the grid so the
    # difference is robust.
    common = (iasi=iasi, cutoff=CUTOFF, apply_continuum=false, with_ils=false, internal_dnu=0.005)

    t=time(); νY,_,BTY = iasi_forward_model(prof, lls; common..., line_mixing=VPYLineMixing(relmat)); @printf("VP_Y: %.1fs\n", time()-t)
    t=time(); νW,_,BTW = iasi_forward_model(prof, lls; common..., line_mixing=vpw);                    @printf("VP_W: %.1fs\n", time()-t)

    d = BTW .- BTY
    @printf("RMS ΔBT(VPW−VPY) = %.4f K    max|ΔBT| = %.4f K    mean ΔBT = %+.4f K\n",
            sqrt(mean(d.^2)), maximum(abs, d), mean(d))
    nmov = count(>(0.05), abs.(d))
    @printf("channels with |ΔBT| > 0.05 K: %d / %d\n", nmov, length(d))

    println("top 20 movers (ν, ΔBT, BT_VPY):")
    for i in sortperm(abs.(d), rev=true)[1:min(20, length(d))]
        @printf("  ν=%8.2f   ΔBT=%+.4f K   BT_VPY=%6.2f K\n", νY.ν[i], d[i], BTY[i])
    end

    open("data/vpw_vpy_$(name).csv", "w") do f
        write(f, "nu_cm1,BT_VPY,BT_VPW,dBT\n")
        for i in eachindex(d)
            write(f, "$(round(νY.ν[i],digits=4)),$(round(BTY[i],digits=5)),$(round(BTW[i],digits=5)),$(round(d[i],digits=6))\n")
        end
    end
    @printf("saved → data/vpw_vpy_%s.csv\n", name)
end

run_band("15um", 645.0, 800.0, 0.05)
run_band("4um",  2000.0, 2500.0, 0.05)
println("\ndone.")
