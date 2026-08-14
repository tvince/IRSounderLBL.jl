# First-light check: run the forward model over the 15 µm CO₂ ν₂ band and
# sanity-check the brightness temperatures. Uses the default line-list set that
# `download_data(:linelists)` installs (CO₂ iso 1–4 + H₂O iso 1–3, 620–825 cm⁻¹).
#
# One-time setup (see README "Getting the data"):
#
#   using IRSounderLBL
#   download_data()              # HITRAN CIA tables — no key needed
#   download_data(:linelists)    # line lists — needs a free HITRAN_API_KEY
#
# Then:
#
#   julia --project=. -t auto smoke_test.jl

using IRSounderLBL

println("Loading the default line-list set ...")
linelists = try
    default_linelists()   # errors with fetch instructions if files are missing
catch err
    println()
    println(sprint(showerror, err))
    println("\nRun `data_status()` for a full checklist of present/missing data.")
    exit(1)
end
for (sp, ll) in linelists
    println("  $(sp): $(length(ll.lines)) lines")
end

prof = afgl_us_standard_50lev()
println("  US Standard Atmosphere: $(length(prof.pressure)) levels, T_sfc=$(prof.temperature[1]) K")

# IASI L1C spec (Δν 0.25 cm⁻¹, OPD 2 cm, Gaussian FWHM 0.5 cm⁻¹), restricted to
# the 645–800 cm⁻¹ window the default line lists cover (with the ±25 cm⁻¹ margin).
sounder = Sounder(ν_min = 645.0, ν_max = 800.0, Δν = 0.25,
                  opd_max = 2.0, fwhm_gauss = 0.5)

println("\nRunning forward_model (645–800 cm⁻¹, CO₂ + H₂O) ...")
t0 = time()
ν, R, BT = forward_model(prof, linelists; sounder)
elapsed = time() - t0
println("  Done in $(round(elapsed; digits=1)) s")

println("\nBrightness temperature over 645–800 cm⁻¹:")
println("  min  = $(round(minimum(BT); digits=2)) K")
println("  max  = $(round(maximum(BT); digits=2)) K")
println("  mean = $(round(sum(BT)/length(BT); digits=2)) K")

# Quick sanity checks
n_ok  = count(200 .<= BT .<= 320)
n_bad = length(BT) - n_ok
println("\n  Channels in [200,320] K: $n_ok / $(length(BT))  (bad: $n_bad)")
if n_bad > 0
    println("  WARNING: $n_bad channels outside physical range!")
    exit(1)
else
    println("  OK: all channels in physical range")
end
