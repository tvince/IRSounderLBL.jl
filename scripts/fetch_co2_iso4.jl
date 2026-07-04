"""
Fetch CO2 isotopologue 4 (627, HITRAN global iso id 10) over the same wide
620–2785 cm⁻¹ margin used for iso 1–3 (see scripts/refetch_wide_margin.jl), and
write it to data/co2_645_2760_iso4.par so `load_linelist("data/co2_645_2760", 1:4)`
now finds a real iso-4 baseline.

Run with your HITRAN key (kept in your environment, never in the repo):
  HITRAN_API_KEY=<key> julia --project scripts/fetch_co2_iso4.jl
"""

using IRSounderLBL
using Printf

const NU_LO = 620.0   # 645 - 25 cm⁻¹ margin (matches iso 1–3)
const NU_HI = 2785.0  # 2760 + 25

const OUTFILE = "data/co2_645_2760_iso4.par"

println("Fetching CO2 mol=2 iso=4 (627, global id 10) → $OUTFILE")
fetch_hitran_api(2, 4, NU_LO, NU_HI; outfile=OUTFILE)

ll = load_hitran_par(OUTFILE)
@printf("  %s: %d lines, %.4f – %.4f cm⁻¹\n",
        basename(OUTFILE), length(ll), ll.ν_min, ll.ν_max)

# Sanity: every record should carry mol_id=2, iso_id=4.
isos = Set(Int(l.iso_id) for l in ll.lines)
mols = Set(Int(l.mol_id) for l in ll.lines)
@printf("  mol_ids=%s  iso_ids=%s\n", join(sort(collect(mols)), ","), join(sort(collect(isos)), ","))
(mols == Set([2]) && isos == Set([4])) ||
    @warn "unexpected mol/iso content — check the download"

println("\nDone. Now `load_linelist(\"data/co2_645_2760\", 1:4; ...)` loads iso 1–4.")
