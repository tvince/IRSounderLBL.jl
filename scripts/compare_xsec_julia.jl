"""
Export Julia H2O cross-section for single-layer comparison with ARTS.
Matches the layer conditions in compare_xsec_single_layer.py:
  T = 255 K, p = 500 hPa, vmr_H2O = 3e-3

Output: data/julia_xsec_h2o_layer.csv
"""

using RadiativeTransfer

const T_K     = 255.0
const P_HPA   = 500.0
const VMR_H2O = 3.0e-3
const P_ATM   = P_HPA / 1013.25
const NU_MIN  = 1380.0
const NU_MAX  = 1800.0
const DNU     = 0.005
const CUTOFF  = 25.0

println("Loading H2O linelists...")
all_lines = HITRANLine[]
for (fname, required) in [
    ("data/h2o_645_2760.par",      true),
    ("data/h2o_645_2760_iso2.par", true),
    ("data/h2o_645_2760_iso3.par", true),
]
    isfile(fname) || (required && @warn("Missing $fname"); continue)
    ll = load_hitran_par(fname; ν_min=NU_MIN - CUTOFF, ν_max=NU_MAX + CUTOFF)
    append!(all_lines, ll.lines)
end
ll = HITRANLinelist(all_lines)
println("  $(length(ll)) lines")

ν_grid = wavenumber_grid(NU_MIN, NU_MAX, DNU)
println("Computing cross-sections at T=$(T_K) K, p=$(P_HPA) hPa...")
σ = compute_voigt_cross_sections(ν_grid, ll, T_K, P_ATM;
                                  cutoff=CUTOFF, vmr_self=VMR_H2O)

outpath = joinpath("data", "julia_xsec_h2o_layer.csv")
open(outpath, "w") do f
    write(f, "nu_cm1,sigma_cm2\n")
    for (ν, s) in zip(ν_grid.ν, σ)
        write(f, "$(round(ν, digits=4)),$(s)\n")
    end
end
println("Saved → $outpath  ($(length(σ)) points)")
