"""
Dump Julia per-layer Curtis-Godson properties for the same AFGL US Standard
50-level profile used in the 4.3 µm LBLRTM comparison, so we can diff against
LBLRTM's TAPE6 values (parsed by scripts/parse_lblrtm_tape6_layers.py).

For each of the 49 layers:
  z_from_km, z_to_km, p_eff_hPa, T_eff_K, vmr_CO2, N_CO2_molec_cm2,
  gamma_L_2364, gamma_D_2364

where gamma_L_2364 and gamma_D_2364 are the Lorentz/Doppler HWHM of a
representative strong P-branch CO₂ line near 2364 cm⁻¹, evaluated at the
layer's Curtis-Godson (p_eff, T_eff) — the same widths Julia uses in the
Voigt evaluation.

This dump should be diff'd against data/lblrtm/lblrtm_layers_43um.csv to
see whether the per-line core-temperature comb is driven by per-layer
effective T/p divergence between codes.

Usage:
  julia --project scripts/julia_layer_dump.jl
"""

using IRSounderLBL
using Printf

const OUT = "data/lblrtm/julia_layers_43um.csv"

prof = afgl_us_standard_50lev()
lp   = layer_properties(prof)
n_lay = length(lp.p_mid)

# Reference CO₂ line near 2364 cm⁻¹ — pick a strong P-branch transition from
# the loaded HITRAN linelist (iso 1, in the 2364 cm⁻¹ neighborhood).
println("Locating reference CO₂ line near 2364 cm⁻¹…")
ll = load_hitran_par("data/co2_645_2760.par"; ν_min=2363.0, ν_max=2365.0)
ref = first(sort(ll.lines, by = l -> -l.intensity))  # strongest in window
@printf("  reference line: ν₀=%.4f cm⁻¹  S=%.3e cm⁻¹/(molec·cm⁻²)  γ_air=%.4f  n_air=%.3f\n",
        ref.wavenumber, ref.intensity, ref.air_broad, ref.temp_depend)

z_lev = prof.altitude
co2   = prof.vmr[CO2]
p_cg  = lp.p_cg[CO2]
T_cg  = lp.T_cg[CO2]
vmr_cg= lp.vmr_cg[CO2]
Δp    = lp.Δp

println("Writing per-layer CSV…")
open(OUT, "w") do f
    write(f, "layer,z_from_km,z_to_km,p_eff_hPa,T_eff_K,vmr_CO2,N_CO2_cm2,"
           * "gamma_L_2364,gamma_D_2364\n")
    for k in 1:n_lay
        # column amount uses Curtis-Godson VMR (column-integral) and Δp.
        # Same formula as IRSounderLBL.column_amount (not exported):
        #   N = vmr * Δp_hPa * Nₐ / (g * Mair) * 1e-4   [molec/cm²]
        N_co2 = vmr_cg[k] * Δp[k] * 2.1209e22
        # Evaluate the reference line at the LBL (p_cg, T_cg) — same place
        # the forward model evaluates it. p_cg is hPa, broadening expects atm.
        γ_L, γ_D = pressure_broadened_width(ref, p_cg[k] / 1013.25, T_cg[k];
                                            vmr_self = vmr_cg[k])
        write(f, @sprintf("%d,%.3f,%.3f,%.6e,%.4f,%.6e,%.6e,%.6e,%.6e\n",
                          k, z_lev[k], z_lev[k+1], p_cg[k], T_cg[k],
                          vmr_cg[k], N_co2, γ_L, γ_D))
    end
end
println("  wrote $OUT  ($n_lay layers)")
