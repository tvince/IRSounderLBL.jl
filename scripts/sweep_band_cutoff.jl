# #4 band-strength cutoff: find the largest min_band_strength keeping the LM
# brightness-temperature change under 1 mK (conservative: monochromatic, no ILS,
# which upper-bounds the ILS-convolved product difference).
using IRSounderLBL, Printf
const CUTOFF=25.0
const LM_DIR="data/Line-mixing_HITRAN2020/data_new"
function load_co2(lo,hi)
    L=HITRANLine[]
    for iso in 1:3
        fn = iso==1 ? "co2_645_2760.par" : "co2_645_2760_iso$(iso).par"
        p=joinpath("data",fn); isfile(p)||continue
        append!(L, load_hitran_par(p; ν_min=lo-CUTOFF, ν_max=hi+CUTOFF).lines)
    end
    HITRANLinelist(L)
end
const prof = afgl_us_standard_50lev()
const THRS = (1e-28,1e-27,1e-26,1e-25)

function sweep(name,lo,hi)
    relmat = load_hitran_relmat(LM_DIR, lo, hi; stot_min=0.0)
    ll = load_co2(lo,hi); lls = Dict{GasSpecies,HITRANLinelist}(CO2=>ll)
    n_ch = round(Int,(hi-lo)/0.05)+1
    iasi = IASIInstrument(lo,hi,0.05,n_ch,2.0,0.5)
    common = (iasi=iasi, cutoff=CUTOFF, apply_continuum=false, with_ils=false, internal_dnu=0.005)
    nb = count(b->Int(b.li)<=8, relmat.bands)
    @printf("\n=== %s (%d coupled bands) ===\n", name, nb)
    iasi_forward_model(prof, lls; common..., line_mixing=VPYLineMixing(relmat))  # warmup
    local BT0
    t0 = @elapsed begin; _,_,BT0 = iasi_forward_model(prof, lls; common..., line_mixing=VPYLineMixing(relmat)); end
    @printf("  baseline (thr=0): %.2f s\n", t0)
    for thr in THRS
        local BT
        t = @elapsed begin; _,_,BT = iasi_forward_model(prof, lls; common..., line_mixing=VPYLineMixing(relmat; min_band_strength=thr)); end
        kept = count(b->Int(b.li)<=8 && IRSounderLBL._band_eff_strength(b)>=thr, relmat.bands)
        d = maximum(abs.(BT .- BT0))
        @printf("  thr=%.0e: kept %3d/%d  max|ΔBT|=%.4f mK  %s  %.2f s (%.2fx)\n",
                thr, kept, nb, 1000*d, d < 1e-3 ? "PASS" : "FAIL", t, t0/t)
        flush(stdout)
    end
end
sweep("15um", 645.0, 800.0)
sweep("4um",  2000.0, 2500.0)
println("\nsweep done.")
