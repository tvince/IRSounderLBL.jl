"""
Spectral-grid convergence test, with IASI-NG in mind.

Question: is the 0.005 cm⁻¹ internal monochromatic grid fine enough — for IASI
(apodized ILS FWHM ~0.5 cm⁻¹, sampling 0.25) AND for IASI-NG (~2× resolution:
ILS FWHM ~0.25 cm⁻¹, sampling 0.125, longer OPD)?

Two things are measured, on small windows around the two LBLRTM worst points
(4.3 µm 2335.9 cm⁻¹, 15 µm 656.5 cm⁻¹), CO₂-only cont-OFF, :cim default:

1. MONOCHROMATIC point-value invariance. Julia σ(ν) is a pointwise Voigt sum, so
   BT at an existing grid point should NOT depend on grid spacing. Confirming this
   shows the monochromatic worst-point spikes are a comparison/interp artifact,
   not Julia under-sampling at that point.

2. ILS-CONVOLVED self-convergence. The grid DOES matter for the integral the ILS
   performs over the line cores. For each sensor, convolve at internal grids
   0.005 / 0.0025 / 0.001 / 0.0005 cm⁻¹ and compare each to the finest (reference)
   on the channel grid. If 0.005 is within tolerance of 0.0005, it is fine enough.

Run with:
  julia --project -t auto scripts/grid_convergence_iasing.jl
"""

using IRSounderLBL
using Printf

const CUTOFF = 25.0
const OUTDIR = "data/lblrtm"

rms(x) = sqrt(sum(abs2, x) / length(x))

prof = afgl_us_standard_50lev()

load_co2(νlo, νhi) = Dict{GasSpecies,HITRANLinelist}(
    CO2 => load_linelist(joinpath("data", "co2_645_2760"), 1:3;
                         ν_min = νlo - CUTOFF, ν_max = νhi + CUTOFF))

# (name, output Δν, opd_max, fwhm_gauss)
sensors = [("IASI   ", 0.25,  2.0, 0.5),
           ("IASI-NG", 0.125, 4.0, 0.25)]
grids = [0.005, 0.0025, 0.001, 0.0005]

# (label, ν_lo, ν_hi, ν_worst, inner_lo, inner_hi)
windows = [("4.3um  core 2335.9", 2326.0, 2346.0, 2335.9, 2331.0, 2341.0),
           ("15um   core 656.5",   646.0,  666.0,  656.5,  651.0,  661.0)]

for (wname, νlo, νhi, νworst, ilo, ihi) in windows
    println("\n" * "="^72)
    println("WINDOW: ", wname, "   (", νlo, "–", νhi, " cm⁻¹)")
    println("="^72)
    ll = load_co2(νlo, νhi)
    @printf("  CO2 lines (±%g cutoff): %d\n", CUTOFF, length(ll[CO2]))

    # ── 1. monochromatic point-value invariance at the worst point ──────────
    println("\n  [1] Monochromatic BT at ν=$(νworst) cm⁻¹ vs grid (with_ils=false):")
    for g in (0.005, 0.001, 0.0005)
        n = round(Int, (νhi - νlo) / g) + 1
        inst = IASIInstrument(νlo, νhi, g, n, 2.0, 0.5)
        νo, _, BT = iasi_forward_model(prof, ll; iasi = inst, high_res_factor = 1,
            cutoff = CUTOFF, apply_continuum = false, continua = (),
            with_ils = false, line_mixing = nothing)
        i = argmin(abs.(νo.ν .- νworst))
        @printf("        Δν=%.4f : BT(%.4f) = %.5f K\n", g, νo.ν[i], BT[i])
    end

    # ── 2. ILS-convolved self-convergence, per sensor ───────────────────────
    for (sname, outΔν, opd, fw) in sensors
        nch = round(Int, (νhi - νlo) / outΔν) + 1
        runs = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
        for g in grids
            hrf  = round(Int, outΔν / g)
            inst = IASIInstrument(νlo, νhi, outΔν, nch, opd, fw)
            νo, _, BT = iasi_forward_model(prof, ll; iasi = inst, high_res_factor = hrf,
                cutoff = CUTOFF, apply_continuum = false, continua = (),
                with_ils = true, line_mixing = nothing)
            push!(runs, (g, collect(νo.ν), BT))
        end
        νref, BTref = runs[end][2], runs[end][3]          # finest grid = reference
        mask = (νref .>= ilo) .& (νref .<= ihi)           # inner window (avoid edges)
        @printf("\n  [2] %s  (ILS FWHM=%.3f, Δν_out=%.3f) ILS-convolved BT vs finest grid %.4f:\n",
                strip(sname), fw, outΔν, grids[end])
        @printf("        %-9s %-12s %-12s\n", "Δν_int", "RMS ΔBT (K)", "max|ΔBT| (K)")
        for (g, νo, BT) in runs
            d = BT[mask] .- BTref[mask]
            flag = g == grids[end] ? "  (ref)" : ""
            @printf("        %.4f    %10.5f   %10.5f%s\n", g, rms(d), maximum(abs.(d)), flag)
        end
    end
end

println("\nDone.")
