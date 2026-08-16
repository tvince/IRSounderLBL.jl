# Example 1 — simulate a brightness-temperature spectrum
#
# The starting point for everything else: take an atmospheric profile, a set of
# HITRAN line lists and an instrument specification, and produce the spectrum a
# nadir-viewing sounder would measure.
#
# Data needed: the default 15 µm line-list set. One-time setup —
#
#   using IRSounderLBL
#   download_data()             # CIA tables, no key needed
#   download_data(:linelists)   # needs a free HITRAN_API_KEY
#
# Run:  julia --project=. -t auto scripts/examples/01_forward_spectrum.jl
# Time: ~40 s on an M1 Pro with 6 threads.

using IRSounderLBL
using Printf

# ── Inputs ──────────────────────────────────────────────────────────────────
# `default_linelists()` loads the CO₂ (iso 1–4) and H₂O (iso 1–3) lines covering
# 620–825 cm⁻¹ — the 15 µm CO₂ ν₂ band plus the ±25 cm⁻¹ far-wing margin the
# line-shape cutoff needs.
linelists = try
    default_linelists()
catch err
    println(sprint(showerror, err))
    println("\nRun `data_status()` to see what is present and what is missing.")
    exit(1)
end

# One of the six AFGL reference atmospheres bundled with the package: 50 levels
# of pressure, temperature, altitude and the standard gas profiles.
prof = afgl_us_standard_50lev()

# IASI L1C: 0.25 cm⁻¹ sampling, 2 cm maximum optical path difference, and the
# 0.5 cm⁻¹ Gaussian self-apodization applied in L1C processing. Restricted here
# to the window the default line lists cover.
sounder = Sounder(ν_min = 645.0, ν_max = 800.0, Δν = 0.25,
                  opd_max = 2.0, fwhm_gauss = 0.5)

@printf("Atmosphere : AFGL US Standard, %d levels, surface %.1f K\n",
        length(prof.pressure), prof.temperature[1])
@printf("Instrument : %.0f–%.0f cm⁻¹, Δν = %.2f cm⁻¹, %d channels\n",
        sounder.ν_min, sounder.ν_max, sounder.Δν, sounder.n_channels)
@printf("Species    : %s\n", join(sort(string.(keys(linelists))), ", "))

# ── Forward model ───────────────────────────────────────────────────────────
# Profile → Curtis–Godson layers → line-by-line cross sections (+ continuum and
# CIA) → optical depth → Schwarzschild RTE → ILS convolution. Returns the
# channel wavenumbers, the radiances and the brightness temperatures.
println("\nRunning the forward model ...")
t0 = time()
grid, R, BT = forward_model(prof, linelists; sounder)
@printf("  done in %.1f s\n", time() - t0)

# `grid` is a `WavenumberGrid`; its channel wavenumbers are the `.ν` field.
ν = grid.ν

# ── What the spectrum shows ─────────────────────────────────────────────────
# A channel's brightness temperature is roughly the physical temperature of the
# altitude where its optical depth reaches unity. Transparent channels see down
# to the warm surface; opaque ones stop higher up. That spread over a single
# absorption band is what makes temperature sounding possible.
#
# For reference, this profile has a 288.2 K surface, an isothermal 216.7 K
# tropopause from 11–20 km, and temperature rising again above it (226.5 K at
# 30 km) as ozone heats the stratosphere.
i_cold = argmin(BT)
i_warm = argmax(BT)

@printf("\nBrightness temperature over %.0f–%.0f cm⁻¹\n", sounder.ν_min, sounder.ν_max)
@printf("  coldest  %7.2f K  at %.2f cm⁻¹   (peaks near the 216.7 K tropopause)\n",
        BT[i_cold], ν[i_cold])
@printf("  warmest  %7.2f K  at %.2f cm⁻¹   (window — sees the 288.2 K surface)\n",
        BT[i_warm], ν[i_warm])
@printf("  mean     %7.2f K   spread %.2f K\n",
        sum(BT) / length(BT), BT[i_warm] - BT[i_cold])

# Landmarks across the band. Note the Q-branch: it is the *most* opaque point in
# the band, yet it reads several K warmer than the channels flanking it. Being
# most opaque, it stops highest — up in the stratosphere, where temperature has
# started increasing with height again. Opacity does not mean cold; it means
# high, and what that is worth depends on the profile above.
println("\nLandmark channels")
println("  ν (cm⁻¹)     BT (K)   sounds")
for (ν_t, what) in ((648.0,  "P-branch wing — tropopause"),
                    (667.0,  "ν₂ Q-branch — most opaque, mid-stratosphere"),
                    (700.0,  "R-branch — upper troposphere"),
                    (740.0,  "band edge — mid-troposphere"),
                    (795.0,  "window — near-surface"))
    i = argmin(abs.(ν .- ν_t))
    @printf("  %8.2f  %8.2f   %s\n", ν[i], BT[i], what)
end

# Sanity: every channel should land in a physically plausible range.
n_bad = count(b -> !(180.0 <= b <= 330.0), BT)
if n_bad > 0
    @printf("\nWARNING: %d channels outside 180–330 K\n", n_bad)
    exit(1)
end
println("\nAll channels physically plausible.")
println("\nNext: 02_weighting_functions.jl — where each channel actually gets its signal.")
