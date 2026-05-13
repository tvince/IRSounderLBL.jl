"""
Re-fetch CO2 and H2O par files with 25 cm⁻¹ margin beyond the compute grid
(620–2785 instead of 645–2760) so that load_multi's ν_min−CUTOFF window
finds real lines at the spectrum edges.

Overwrites existing data/co2_645_2760*.par and data/h2o_645_2760*.par in place.

Run with:
  HITRAN_API_KEY=<key> julia --project scripts/refetch_wide_margin.jl
"""

using RadiativeTransfer
using Printf

const NU_LO = 620.0   # 645 - 25
const NU_HI = 2785.0  # 2760 + 25

refetch = [
    # (mol_id, iso_id, outfile)
    (2, 1, "data/co2_645_2760.par"),
    (2, 2, "data/co2_645_2760_iso2.par"),
    (2, 3, "data/co2_645_2760_iso3.par"),
    (1, 1, "data/h2o_645_2760.par"),
    (1, 2, "data/h2o_645_2760_iso2.par"),
    (1, 3, "data/h2o_645_2760_iso3.par"),
]

for (mol_id, iso_id, outfile) in refetch
    println("Fetching mol=$mol_id iso=$iso_id → $outfile")
    fetch_hitran_api(mol_id, iso_id, NU_LO, NU_HI; outfile=outfile)
    ll = load_hitran_par(outfile)
    @printf("  %s: %d lines, %.4f – %.4f cm⁻¹\n",
            basename(outfile), length(ll), ll.ν_min, ll.ν_max)
end

println("\nDone. Re-run julia_bt_export.jl to regenerate BT.")
