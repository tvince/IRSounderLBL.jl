#!/usr/bin/env julia
# Quick-look summary of an IASI L1C granule — validates the EPS-native byte
# offsets on a REAL file and gives an eyeball of geometry, cloud, glint, and BT.
#
# Usage:
#   julia --project=. scripts/inspect_iasi_l1c.jl [path.nat] [ν_lo ν_hi]
#
# With no path it reads the first *.nat under data/iasi_l1c/. A narrow default
# wavenumber window (700–760 cm⁻¹) keeps memory bounded: a full orbit is ~91k
# FOVs, so the full 8461-channel spectra would be several GB. Pass an explicit
# (ν_lo, ν_hi) to widen it; geometry/cloud/glint are read for every FOV regardless.
using IRSounderLBL
using Printf, Statistics

# ── locate the granule ────────────────────────────────────────────────────────
path = if length(ARGS) >= 1 && isfile(ARGS[1])
    ARGS[1]
else
    dir = joinpath(@__DIR__, "..", "data", "iasi_l1c")
    nats = isdir(dir) ? filter(f -> endswith(lowercase(f), ".nat"), readdir(dir; join=true)) : String[]
    if isempty(nats)
        println("No .nat granule found.")
        println("  Put an IASI L1C file in data/iasi_l1c/, or pass a path:")
        println("    julia --project=. scripts/inspect_iasi_l1c.jl <file.nat> [ν_lo ν_hi]")
        exit(0)
    end
    first(sort(nats))
end

wnolim = length(ARGS) >= 3 ? (parse(Float64, ARGS[2]), parse(Float64, ARGS[3])) : (700.0, 760.0)

println("Reading: ", path)
@printf("  size on disk: %.2f GB\n", filesize(path) / 2^30)
println("  window:  $(wnolim[1])–$(wnolim[2]) cm⁻¹  (geometry/cloud/glint read for all FOVs)")

t = @elapsed g = read_iasi_l1c(path; wnolim=wnolim, bright=true)   # bright=true → BT [K]
@printf("  read %d FOVs in %.1fs\n\n", nfov(g), t)

# ── metadata ────────────────────────────────────────────────────────────────
for k in ("INSTRUMENT_ID", "SPACECRAFT_ID", "SENSING_START", "SENSING_END", "ORBIT_START")
    haskey(g.mph, k) && @printf("  %-15s %s\n", k, g.mph[k])
end
println()

n = nfov(g)
if n == 0
    println("No FOVs passed the read; nothing to summarise.")
    exit(0)
end

rng(v) = (minimum(v), median(v), maximum(v))
@printf("  lat        %7.2f … %7.2f  (median %6.2f)\n", minimum(g.lat), maximum(g.lat), median(g.lat))
@printf("  lon        %7.2f … %7.2f  (median %6.2f)\n", minimum(g.lon), maximum(g.lon), median(g.lon))
lo, md, hi = rng(g.zen); @printf("  sat zenith %7.2f … %7.2f  (median %6.2f)°\n", lo, hi, md)
lo, md, hi = rng(g.sza); @printf("  sol zenith %7.2f … %7.2f  (median %6.2f)°\n", lo, hi, md)
lo, md, hi = rng(g.sra); @printf("  glint φᵣ   %7.2f … %7.2f  (median %6.2f)°\n", lo, hi, md)
nglint = count(<(15.0), g.sra)
@printf("  → %d FOVs (%.1f%%) within the 15° sun-glint cone\n", nglint, 100nglint/n)
@printf("  land frac  %.0f%% mean,  cloud-free (cld<5%%): %d FOVs (%.1f%%)\n\n",
        mean(g.lnd), count(<(5.0), g.cld), 100count(<(5.0), g.cld)/n)

# ── AVHRR cloud-fraction histogram ────────────────────────────────────────────
println("  AVHRR cloud fraction histogram (% of FOVs per 10% bin):")
edges = 0:10:100
for b in 1:10
    lo_e, hi_e = edges[b], edges[b+1]
    c = count(x -> (lo_e <= x < hi_e) || (b == 10 && x == 100.0), g.cld)
    bar = repeat("█", round(Int, 40 * c / n))
    @printf("    %3d–%-3d%% | %-40s %5.1f%%\n", lo_e, hi_e, bar, 100c/n)
end
println()

# ── brightness temperature over the window ────────────────────────────────────
bt = vec(g.spc)                       # all channels × FOVs (BT, K)
@printf("  BT over %.0f–%.0f cm⁻¹:  %.1f … %.1f K  (median %.1f K)\n",
        wnolim[1], wnolim[2], minimum(bt), maximum(bt), median(bt))
if !(100 < minimum(bt) && maximum(bt) < 400)
    @warn "BT outside 100–400 K — MDR offsets / radiance scaling may not match this product version."
else
    println("  ✓ BT in physical range — EPS-native offsets look correct for this granule.")
end
