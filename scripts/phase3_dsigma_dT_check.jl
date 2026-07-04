# Phase 3 — validate ∂σ/∂T against central FD of compute_voigt_cross_sections.
using IRSounderLBL
using Printf

# CO2-like synthetic lines (mol 2, iso 1) with realistic E″, n_air, shift.
mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                          Float32(0.07), Float32(0.08), E,
                          Float32(0.75), Float32(-0.003))
ll = HITRANLinelist([mk(700.3, 2.0e-21, 250.0), mk(701.4, 8.0e-22, 600.0),
                     mk(702.6, 1.5e-21, 120.0)])
g = wavenumber_grid(698.0, 705.0, 0.002)   # dense enough to resolve cores

p_atm = 0.5
for T in (220.4, 260.3, 295.6)             # mid-cell (avoid integer-K Q straddle)
    σ, dσ = compute_voigt_cross_sections_dT(g, ll, T, p_atm)
    σ_fwd = compute_voigt_cross_sections(g, ll, T, p_atm)
    h = 0.05                                # stays within the 1 K TIPS cell
    σp = compute_voigt_cross_sections(g, ll, T + h, p_atm)
    σm = compute_voigt_cross_sections(g, ll, T - h, p_atm)
    fd = (σp .- σm) ./ (2h)

    σ_match = maximum(abs.(σ .- σ_fwd))
    nz = findall(>(maximum(σ) * 1e-6), σ)   # compare where σ is non-negligible
    relerr = maximum(abs.(dσ[nz] .- fd[nz])) / (maximum(abs.(fd[nz])) + 1e-300)
    @printf("T=%.1f  σ≡fwd max|Δ|=%.2e | dσ/dT vs FD: max|Δ|=%.3e rel=%.3e\n",
            T, σ_match, maximum(abs.(dσ[nz] .- fd[nz])), relerr)
end
