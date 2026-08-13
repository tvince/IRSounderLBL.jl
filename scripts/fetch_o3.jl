"""
Fetch O3 (ozone) isotopologues 1–4 over the full IASI spectral range
(620–2785 cm⁻¹ = 645–2760 + 25 cm⁻¹ margin) and write them to
data/o3_645_2760{,_iso2,_iso3,_iso4}.par (same base/naming convention as
data/co2_645_2760*). O3 is currently UNMODELLED in the forward model (CO2 + H2O
only); its ν₂ band (~701 cm⁻¹) overlaps the 15 µm CO2 retrieval window and its
ν₃/ν₁ bands (~1000–1150 cm⁻¹) sit in the atmospheric window. The pre-existing
data/o3_980_1090*.par (ν₃ only) is superseded by this full-range pull.

O3 iso ids (HITRAN global): 1=666(16), 2=668(17), 3=686(18), 4=667(19).

Run with your HITRAN key (kept in your environment, never in the repo):
  HITRAN_API_KEY=<key> julia --project scripts/fetch_o3.jl

Note: O3 is line-dense across this range (ν₂ + ν₃ + ν₁ + hot bands); expect a
multi-hundred-thousand-line download per isotopologue and a few minutes total.
"""

using IRSounderLBL
using Printf

const NU_LO = 620.0   # 645 - 25 cm⁻¹ margin
const NU_HI = 2785.0  # 2760 + 25

const FETCH = [
    (3, 1, "data/o3_645_2760.par"),
    (3, 2, "data/o3_645_2760_iso2.par"),
    (3, 3, "data/o3_645_2760_iso3.par"),
    (3, 4, "data/o3_645_2760_iso4.par"),
]

for (mol_id, iso_id, outfile) in FETCH
    println("Fetching O3 mol=$mol_id iso=$iso_id → $outfile")
    fetch_hitran_api(mol_id, iso_id, NU_LO, NU_HI; outfile=outfile)
    ll = load_hitran_par(outfile)
    isos = sort(collect(Set(Int(l.iso_id) for l in ll.lines)))
    @printf("  %s: %d lines, %.4f–%.4f cm⁻¹, iso_ids=%s\n",
            basename(outfile), length(ll), ll.ν_min, ll.ν_max, join(isos, ","))
end

println("\nDone. load_linelist(\"data/o3_645_2760\", 1:4; ...) now loads O3 over the IASI range.")
