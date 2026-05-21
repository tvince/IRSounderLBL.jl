# Inspect VPWLineMixing's default whitelist for different ν ranges.
# Same code, different behaviour depending on what `load_hitran_relmat` loaded.
using RadiativeTransfer
using Printf

const LM_DIR = "data/Line-mixing_HITRAN2020/data_new"

function show_top(label, ν_min, ν_max; n_top=5)
    println("\n── $label  ($(ν_min)–$(ν_max) cm⁻¹) ──────────────")
    relmat = load_hitran_relmat(LM_DIR, ν_min, ν_max; stot_min=0.0)
    @printf("  bands loaded: %d\n", length(relmat.bands))

    # default behaviour: abundance-weighted top-N
    lm_default = VPWLineMixing(relmat; n_top=n_top)
    println("  default whitelist (top-$n_top abundance-weighted):")
    for name in sort!(collect(lm_default.whitelist))
        b = relmat.bands[findfirst(x -> x.name == name, relmat.bands)]
        @printf("    %-13s  iso=%d  ν∈[%.1f, %.1f]\n",
                name, Int(b.isot), b.ν_min, b.ν_max)
    end

    # per-isotope variant
    lm_iso12 = VPWLineMixing(relmat; n_top=n_top, isotopes=[1, 2])
    println("  isotopes=[1,2] whitelist (top-$n_top per isotope):")
    for name in sort!(collect(lm_iso12.whitelist))
        b = relmat.bands[findfirst(x -> x.name == name, relmat.bands)]
        @printf("    %-13s  iso=%d  ν∈[%.1f, %.1f]\n",
                name, Int(b.isot), b.ν_min, b.ν_max)
    end
end

show_top("15 µm only",   645.0,  800.0)
show_top("Full IASI",    645.0, 2760.0)
show_top("4.3 µm only", 2150.0, 2450.0)

println("\n── Full IASI loaded, ν_window=(645,800) ──────────────")
relmat_full = load_hitran_relmat(LM_DIR, 645.0, 2760.0; stot_min=0.0)
lm = VPWLineMixing(relmat_full; n_top=5, ν_window=(645.0, 800.0))
println("  whitelist with ν_window:")
for name in sort!(collect(lm.whitelist))
    b = relmat_full.bands[findfirst(x -> x.name == name, relmat_full.bands)]
    @printf("    %-13s  iso=%d  ν∈[%.1f, %.1f]\n", name, Int(b.isot), b.ν_min, b.ν_max)
end
