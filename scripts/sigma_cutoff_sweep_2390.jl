"""
Isolate the 2390.235 cm⁻¹ band-head cross-section excess: is Julia's +47 % column
τ a CUTOFF/far-wing-summation effect or something in the line core?

For several layers (spanning pressure) compute Julia σ(2390.235) at a range of
cutoff radii and compare to LBLRTM's ACTUAL per-layer σ (= τ_ODint / N_CO2).

- If the Julia/LBLRTM ratio → 1 as the cutoff shrinks, the excess lives in the
  2-25 cm⁻¹ FAR WING (distant-line wings / pedestal) → cutoff/line-shape issue.
- If the ratio is flat vs cutoff, the excess is in the near core/normalization.
χ is logically excluded as a cause of the DIFFERENCE (LBLRTM has χ=1 disabled too),
but this sweep shows the functional form regardless.

Run: julia --project -t auto scripts/sigma_cutoff_sweep_2390.jl
"""

using IRSounderLBL, Printf, DelimitedFiles

const NU0 = 2390.235
const G   = wavenumber_grid(NU0, NU0, 1.0)
const CUTS = [8.0, 15.0, 18.0, 20.0, 22.0, 25.0]

prof   = afgl_us_standard_50lev()
layers = layer_properties(prof)
n_lay  = length(layers.p_mid)

# Load CO2 lines out to the widest cutoff (±25) around NU0
all_lines = HITRANLine[]
for iso in 1:3
    fn = iso == 1 ? "co2_645_2760.par" : "co2_645_2760_iso$(iso).par"
    fp = joinpath("data", fn); isfile(fp) || continue
    append!(all_lines, load_hitran_par(fp; ν_min=NU0-25.0, ν_max=NU0+25.0).lines)
end
ll = HITRANLinelist(all_lines)
@printf("CO2 lines within ±25 of %.3f: %d\n", NU0, length(all_lines))

# LBLRTM actual per-layer σ = τ_ODint / N_CO2
T = readdlm("data/lblrtm/lblrtm_per_layer_tau_2390.csv", ',', header=true)
hT = String.(vec(T[2])); τ_lbl = Float64.(T[1][:, findfirst(==("tau_at_nu0"), hT)])
Lm = readdlm("data/lblrtm/lblrtm_layers_43um.csv", ',', header=true)
hL = String.(vec(Lm[2])); N_lbl = Float64.(Lm[1][:, findfirst(==("N_CO2_cm2"), hL)])
σ_lbl = τ_lbl ./ N_lbl

σ_jul(k, cut) = compute_voigt_cross_sections(G, ll, layers.T_cg[CO2][k],
                  layers.p_cg[CO2][k]/1013.25; vmr_self=layers.vmr_cg[CO2][k],
                  cutoff=cut)[1]

@printf("\n%-3s %-7s %-9s", "L", "p_hPa", "σ_LBLRTM")
for c in CUTS; @printf(" r@%-4.0f", c); end
println()
for k in (1, 3, 5, 8, 12, 16, 20)
    @printf("%-3d %-7.1f %-9.3e", k, layers.p_cg[CO2][k], σ_lbl[k])
    for c in CUTS
        @printf(" %5.3f", σ_jul(k, c) / σ_lbl[k])
    end
    println()
end
println("\n(r@C = σ_Julia(cutoff=C) / σ_LBLRTM ; 1.000 = perfect match)")
