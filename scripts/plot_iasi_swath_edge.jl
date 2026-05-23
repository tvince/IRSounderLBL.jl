"""
Run iasi_forward_model at nadir and at the swath-edge zenith angle
(IASI scan = 48.33° → local θ ≈ 57.43° at 817 km MetOp altitude), plot
both spectra and the difference. Demonstrates the off-nadir mechanism
now that `scan_angle_to_local_zenith` is wired up.

Usage:
  julia --project=. -t auto scripts/plot_iasi_swath_edge.jl

Outputs:
  data/iasi_swath_edge_compare.png
  data/iasi_swath_edge_compare.csv   (ν, BT_nadir, BT_edge)
"""

using RadiativeTransfer
using DelimitedFiles
using Plots
using Printf

const ROOT = dirname(@__DIR__)

# ── Load linelists (reuses cached HITRAN .par files from data/) ─────────────
function load_iso13(mol_id, ν_min, ν_max, base)
    lines = HITRANLine[]
    for iso in 1:3
        fname = iso == 1 ? "$(base).par" : "$(base)_iso$(iso).par"
        fpath = joinpath(ROOT, "data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=Float64(ν_min), ν_max=Float64(ν_max))
        append!(lines, ll.lines)
    end
    return isempty(lines) ?
        HITRANLinelist(HITRANLine[], Set{Int}(), Float64(ν_min), Float64(ν_max)) :
        HITRANLinelist(lines)
end

println("Loading HITRAN catalogues...")
linelists = Dict{GasSpecies, HITRANLinelist}(
    CO2 => load_iso13(2, 645.0, 2760.0, "co2_645_2760"),
    H2O => load_iso13(1, 645.0, 2760.0, "h2o_645_2760"),
    O3  => load_iso13(3, 980.0, 1090.0, "o3_980_1090"),
    N2O => load_iso13(4, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => load_iso13(6, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => load_iso13(5, 2000.0, 2280.0, "co_2000_2280"),
)
for (sp, ll) in linelists
    @printf "  %-4s  %6d lines\n" string(sp) length(ll)
end

prof = afgl_us_standard_50lev()

# Filter weak lines for speed (1e-23 cm/molec @ 296 K; same threshold as plot_iasi_spectrum.jl)
S_MIN = 1e-23
linelists_f = Dict{GasSpecies, HITRANLinelist}()
for (sp, ll) in linelists
    linelists_f[sp] = filter_linelist(ll, S_MIN)
end
println("\nAfter S_min=$(S_MIN) filter:")
for (sp, ll) in linelists_f
    @printf "  %-4s  %6d lines\n" string(sp) length(ll)
end

# ── Run both geometries ─────────────────────────────────────────────────────
nadir = nadir_geometry()
edge  = ViewingGeometry(scan_angle_to_local_zenith(48.33))
@printf "\nNadir geometry: θ=%.2f°, μ=%.4f, airmass=%.3f\n" nadir.zenith_angle nadir.μ airmass_factor(nadir)
@printf "Edge  geometry: θ=%.2f°, μ=%.4f, airmass=%.3f\n" edge.zenith_angle edge.μ airmass_factor(edge)

println("\nRunning forward model — nadir...")
t0 = time()
ν_iasi, _, BT_nadir = iasi_forward_model(prof, linelists_f;
                                          geom=nadir,
                                          high_res_factor=2,
                                          cutoff=25.0)
@printf "  done in %.1f s\n" (time() - t0)

println("Running forward model — swath edge...")
t0 = time()
_, _, BT_edge = iasi_forward_model(prof, linelists_f;
                                    geom=edge,
                                    high_res_factor=2,
                                    cutoff=25.0)
@printf "  done in %.1f s\n" (time() - t0)

ΔBT = BT_edge .- BT_nadir
@printf "\nΔBT (edge − nadir):  min=%+.2f K  max=%+.2f K  mean=%+.3f K\n" minimum(ΔBT) maximum(ΔBT) sum(ΔBT)/length(ΔBT)

# ── Save CSV ─────────────────────────────────────────────────────────────────
out_csv = joinpath(ROOT, "data", "iasi_swath_edge_compare.csv")
open(out_csv, "w") do f
    write(f, "nu_cm1,BT_nadir,BT_edge\n")
    for i in eachindex(ν_iasi.ν)
        @printf f "%.4f,%.4f,%.4f\n" ν_iasi.ν[i] BT_nadir[i] BT_edge[i]
    end
end
println("Saved → $out_csv")

# ── Plot: two-panel (spectra overlay + ΔBT) ─────────────────────────────────
plt = plot(layout=grid(2, 1; heights=[0.65, 0.35]), size=(1400, 800), dpi=200,
           link=:x, framestyle=:box, background_color=:white,
           left_margin=8Plots.mm, bottom_margin=6Plots.mm)

plot!(plt[1], ν_iasi.ν, BT_nadir; label="Nadir (θ=0°, AMF=1.00)",
      color=:steelblue, lw=0.6,
      ylabel="Brightness Temperature (K)",
      title="IASI simulation — AFGL US Standard, nadir vs swath edge",
      ylims=(180, 305), grid=true, gridalpha=0.3, legend=:bottomright)
plot!(plt[1], ν_iasi.ν, BT_edge; label=@sprintf("Swath edge (θ=%.1f°, AMF=%.2f)", edge.zenith_angle, airmass_factor(edge)),
      color=:darkorange, lw=0.6, alpha=0.85)

plot!(plt[2], ν_iasi.ν, ΔBT; label=nothing,
      color=:darkgreen, lw=0.5,
      xlabel="Wavenumber (cm⁻¹)", ylabel="ΔBT = edge − nadir (K)",
      xlims=(645, 2760), grid=true, gridalpha=0.3)
hline!(plt[2], [0]; color=:black, ls=:dot, lw=0.5, label=nothing)

out_png = joinpath(ROOT, "data", "iasi_swath_edge_compare.png")
savefig(plt, out_png)
println("Saved → $out_png")
