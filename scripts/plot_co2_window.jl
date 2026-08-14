using IRSounderLBL
using Plots
using Printf

# Spectral range of interest
ν_lo, ν_hi = 645.0, 875.0

# Load cached CO₂ and H₂O lines; trim to plot range + line-wing buffer
CUTOFF = 25.0
S_MIN  = 1e-23

function load_species(files, ν_min, ν_max)
    lines = HITRANLine[]
    for f in files
        fpath = joinpath("data", f)
        isfile(fpath) || error("Missing cached file: $fpath — run plot_iasi_spectrum.jl first")
        ll = load_hitran_par(fpath; ν_min = ν_min - CUTOFF, ν_max = ν_max + CUTOFF)
        append!(lines, ll.lines)
    end
    filt = filter(l -> l.intensity >= S_MIN, lines)
    return HITRANLinelist(filt)
end

println("Loading lines for $(ν_lo)–$(ν_hi) cm⁻¹ ...")
co2 = load_species(["co2_645_2760.par",     "co2_645_2760_iso2.par", "co2_645_2760_iso3.par"],
                   ν_lo, ν_hi)
h2o = load_species(["h2o_645_2760.par",     "h2o_645_2760_iso2.par", "h2o_645_2760_iso3.par"],
                   ν_lo, ν_hi)
@printf("  CO₂: %d lines,  H₂O: %d lines\n", length(co2), length(h2o))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => co2, H2O => h2o)
prof      = us_standard_atmosphere()

println("Running forward model ...")
t0 = time()
ν_iasi, _, BT = iasi_forward_model(prof, linelists; high_res_factor=4)
@printf("  Done in %.1f s\n", time() - t0)

# Restrict to the plot window
mask   = (ν_iasi.ν .>= ν_lo) .& (ν_iasi.ν .<= ν_hi)
ν_plot = ν_iasi.ν[mask]
BT_plot = BT[mask]

@printf("BT range %.1f–%.1f cm⁻¹:  min=%.1f K  max=%.1f K\n",
        ν_lo, ν_hi, minimum(BT_plot), maximum(BT_plot))

p = plot(
    ν_plot, BT_plot;
    xlabel     = "Wavenumber (cm⁻¹)",
    ylabel     = "Brightness Temperature (K)",
    title      = "CO₂ 15 µm band + window — US Standard Atmosphere",
    label      = nothing,
    lw         = 0.6,
    color      = :darkblue,
    dpi        = 200,
    size       = (900, 440),
    xlims      = (ν_lo, ν_hi),
    grid       = true,
    gridalpha  = 0.3,
    framestyle = :box,
    background_color = :white,
)

hline!(p, [prof.temperature[1]]; lc=:firebrick, ls=:dash, lw=0.8,
       label="T_sfc = $(round(Int, prof.temperature[1])) K")

# Band annotations
vline!(p, [667.0]; lc=:gray, lw=0.6, ls=:dot, label=nothing)
annotate!(p, 667, minimum(BT_plot) + 8, text("Q-branch\n667 cm⁻¹", :center, 7, :gray))

vline!(p, [720.0]; lc=:gray, lw=0.6, ls=:dot, label=nothing)
annotate!(p, 720, minimum(BT_plot) + 8, text("P/R\nwings", :center, 7, :gray))

savefig(p, "co2_window_check.png")
println("Saved → co2_window_check.png")
