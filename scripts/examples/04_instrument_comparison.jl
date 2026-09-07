# Example 4 — the same atmosphere seen by four different sounders
#
# The forward model is instrument-agnostic: an instrument is just a `Sounder`
# describing a spectral grid and an instrument line shape. Swapping IASI for
# CrIS changes one argument, not the radiative transfer. This example runs one
# atmosphere through four missions and shows what the instrument design does to
# the measurement.
#
# All four are compared over 700–800 cm⁻¹, the range every one of them covers,
# so the only differences are sampling, optical path difference and apodization.
#
# Note what this is a comparison OF: each mission's L1 product as delivered, not
# the instruments stripped to their optics. CrIS and MTG-IRS ship unapodized, and
# are run that way here, but CrIS users conventionally apply Hamming before
# analysis. Pass `apodization = :hamming` to `forward_model` to see that version.
#
# Data needed: the default 15 µm line-list set (see 01_forward_spectrum.jl).
# Run:  julia --project=. -t auto scripts/examples/04_instrument_comparison.jl
# Time: ~2 min on an M1 Pro with 6 threads.

using IRSounderLBL
using Printf

linelists = try
    default_linelists()
catch err
    println(sprint(showerror, err))
    println("\nRun `data_status()` to see what is present and what is missing.")
    exit(1)
end

prof = afgl_us_standard_50lev()

# Mission specifications, from the named constructors. `Sounder` carries the
# spectral sampling Δν, the maximum optical path difference (which sets how
# narrow a feature the sinc ILS can resolve), and the Gaussian apodization FWHM.
missions = [
    ("IASI",     IASIInstrument()),
    ("IASI-NG",  IASINGInstrument()),
    ("CrIS FSR", CrISInstrument(band = :lwir)),
    ("MTG-IRS",  MTGIRSInstrument(band = :lwir)),
]

# Restrict each to the common 700–800 cm⁻¹ window, preserving its own Δν,
# OPD and apodization.
ν_lo, ν_hi = 700.0, 800.0
println("Instrument specifications (full mission), compared over 700–800 cm⁻¹\n")
println("  mission     Δν (cm⁻¹)   OPD (cm)   apod. FWHM   full range (cm⁻¹)")
for (name, s) in missions
    @printf("  %-10s  %8.3f  %9.1f   %9.2f    %.0f–%.0f\n",
            name, s.Δν, s.opd_max, s.fwhm_gauss, s.ν_min, s.ν_max)
end

results = Dict{String, Any}()
println("\nRunning the forward model for each ...")
for (name, s) in missions
    sub = Sounder(ν_min = ν_lo, ν_max = ν_hi, Δν = s.Δν,
                  opd_max = s.opd_max, fwhm_gauss = s.fwhm_gauss)
    t0 = time()
    grid, _, BT = forward_model(prof, linelists; sounder = sub)
    @printf("  %-10s %4d channels in %5.1f s\n", name, length(grid.ν), time() - t0)
    results[name] = (ν = grid.ν, BT = BT)
end

# ── What the instrument design buys you ─────────────────────────────────────
# Sampling density is the obvious difference: IASI-NG puts 801 channels where
# CrIS puts 161. But spectral *contrast* is governed here by apodization rather
# than by optical path difference. CrIS and MTG-IRS are unapodized (pure sinc
# ILS) and so retain the most peak-to-peak range despite having the shortest
# OPD, while IASI's 0.5 cm⁻¹ Gaussian self-apodization smooths its spectrum the
# most. That is a deliberate trade: apodization suppresses sinc ringing and
# keeps inter-channel noise correlation short-ranged (see example 3's Se), at
# the cost of resolution.
#
# CrIS and MTG-IRS report identical numbers because their LWIR specifications
# coincide — both 0.625 cm⁻¹ sampling, 0.8 cm OPD, unapodized. Over this window
# they are the same instrument.
#
# The contrast ranking below is therefore a statement about product conventions
# rather than about which instrument is sharper. Apodize CrIS with Hamming, as
# its users do, and it stops being the high-contrast one: over this window its
# BT range falls 67.93 K → 64.47 K, below both IASI (64.90 K) and IASI-NG
# (65.92 K). The cost is modest — about 5% — because at 0.625 cm⁻¹ over a dense
# CO₂ band the channel-to-channel structure is mostly real absorption, not sinc
# ringing; the side-lobe suppression is what you are buying, not resolution.
println("\nWhat each instrument resolves over 700–800 cm⁻¹")
println("  mission     channels   BT range (K)   coldest (K)   std (K)")
for (name, _) in missions
    r  = results[name]
    bt = r.BT
    @printf("  %-10s %8d   %10.2f   %11.2f   %7.2f\n",
            name, length(bt), maximum(bt) - minimum(bt), minimum(bt),
            sqrt(sum(abs2, bt .- sum(bt) / length(bt)) / length(bt)))
end

# ── Reading the same feature at four resolutions ────────────────────────────
# One absorption feature, as each instrument records it. The spread is around a
# kelvin — small, but it is pure instrument: the atmosphere and the line-by-line
# calculation behind these four numbers were identical. Comparing radiances
# across missions means accounting for this, which is exactly what a common
# forward model with a swappable `Sounder` lets you do.
println("\nA single CO₂ feature near 720 cm⁻¹, as each instrument records it")
println("  mission      nearest ν    BT (K)")
for (name, _) in missions
    r = results[name]
    i = argmin(abs.(r.ν .- 720.0))
    @printf("  %-10s  %9.3f  %8.2f\n", name, r.ν[i], r.BT[i])
end

println("\nThe radiative transfer was identical in all four runs — only the")
println("`Sounder` changed. Presets exist for IASI, IASI-NG, CrIS and MTG-IRS;")
println("for anything else, construct a `Sounder` with its Δν, OPD and apodization.")
