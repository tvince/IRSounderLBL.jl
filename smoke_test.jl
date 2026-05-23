using IRSounderLBL

println("Loading CO₂ lines from data/co2_645_700.par ...")
ll = load_hitran_par("data/co2_645_700.par"; ν_min=645.0, ν_max=700.0)
println("  Loaded $(length(ll.lines)) lines")

prof = us_standard_atmosphere()
println("  US Standard Atmosphere: $(length(prof.pressure)) levels, T_sfc=$(prof.temperature[1]) K")

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll)

println("\nRunning iasi_forward_model (645–700 cm⁻¹ CO₂ only) ...")
t0 = time()
ν, R, BT = iasi_forward_model(prof, linelists)
elapsed = time() - t0
println("  Done in $(round(elapsed; digits=1)) s")

# Report stats for the 645–700 window
mask = (ν.ν .>= 645.0) .& (ν.ν .<= 700.0)
BT_window = BT[mask]
println("\nBrightness Temperature in 645–700 cm⁻¹:")
println("  min  = $(round(minimum(BT_window); digits=2)) K")
println("  max  = $(round(maximum(BT_window); digits=2)) K")
println("  mean = $(round(sum(BT_window)/length(BT_window); digits=2)) K")

# Quick sanity checks
n_ok  = count(200 .<= BT_window .<= 320)
n_bad = length(BT_window) - n_ok
println("\n  Channels in [200,320] K: $n_ok / $(length(BT_window))  (bad: $n_bad)")
if n_bad > 0
    println("  WARNING: $n_bad channels outside physical range!")
else
    println("  OK: all channels in physical range")
end
