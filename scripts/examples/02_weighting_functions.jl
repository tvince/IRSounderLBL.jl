# Example 2 — weighting functions: where does each channel get its signal?
#
# Example 1 showed that different channels report different temperatures. This
# one makes that quantitative by computing the Jacobian K = ∂BT/∂T(level): the
# change in each channel's brightness temperature per kelvin of warming at each
# atmospheric level. A channel's *weighting function* is its row of K, and the
# level where that row peaks is the altitude the channel is really measuring.
#
# The Jacobian is computed analytically — differentiating the radiative transfer
# and the line shapes directly, not by finite differences — which is what makes
# the retrieval in example 3 affordable.
#
# Data needed: the default 15 µm line-list set (see 01_forward_spectrum.jl).
# Run:  julia --project=. -t auto scripts/examples/02_weighting_functions.jl
# Time: ~2 min on an M1 Pro with 6 threads.

using IRSounderLBL
using Printf

linelists = try
    default_linelists()
catch err
    println(sprint(showerror, err))
    println("\nRun `data_status()` to see what is present and what is missing.")
    exit(1)
end

prof    = afgl_us_standard_50lev()
nlev    = length(prof.pressure)
sounder = Sounder(ν_min = 645.0, ν_max = 800.0, Δν = 0.25,
                  opd_max = 2.0, fwhm_gauss = 0.5)

# The state vector describes what we differentiate with respect to. Here: the
# temperature at every level, plus the surface temperature. No gases retrieved,
# no emissivity — those blocks are simply switched off.
spec = StateVectorSpec(nlev, GasSpecies[];
                       include_temperature = true,
                       include_tsfc        = true,
                       include_emissivity  = false)

@printf("State vector: %d elements (%d temperature levels + T_sfc)\n", spec.n, nlev)
println("Computing the analytic Jacobian ...")

t0 = time()
jac = analytic_jacobian(prof, linelists, spec;
                        sounder      = sounder,
                        observable   = :bt,
                        internal_dnu = 0.0025)
@printf("  done in %.1f s — K is %d channels × %d state elements\n",
        time() - t0, size(jac.K, 1), size(jac.K, 2))

ν    = jac.ν
K_T  = jac.K[:, spec.temp_range]        # ∂BT/∂T at each level
k_sf = jac.K[:, spec.tsfc_index]        # ∂BT/∂T_sfc

# ── Where each channel peaks ────────────────────────────────────────────────
# The peak of a channel's weighting function is the level it is most sensitive
# to. Reading these down the band traces out the vertical scan: the strongly
# absorbing band centre peaks high, the transparent edges peak at the ground.
println("\nPeak sensitivity by channel")
println("  ν (cm⁻¹)    peak z (km)   ∂BT/∂T_sfc   weighting-function peak")
for ν_t in (648.0, 667.0, 680.0, 700.0, 720.0, 740.0, 770.0, 795.0)
    i     = argmin(abs.(ν .- ν_t))
    lev   = argmax(abs.(view(K_T, i, :)))
    @printf("  %8.2f  %10.1f  %11.3f   %.4f K/K at %.1f hPa\n",
            ν[i], prof.altitude[lev], k_sf[i], K_T[i, lev], prof.pressure[lev])
end

# ── Surface sensitivity splits the band in two ──────────────────────────────
# ∂BT/∂T_sfc is the cleanest summary of opacity: it is ~1 where the atmosphere
# is transparent (the channel sees the ground directly) and ~0 where the band is
# opaque (the surface is completely hidden). Only the transparent channels carry
# information about the surface — which is why retrieving surface temperature
# and emissivity together from a single window is famously ill-posed.
n_surf  = count(>(0.5), k_sf)
n_blind = count(<(0.01), k_sf)
@printf("\nChannels that see the surface (∂BT/∂T_sfc > 0.5): %d of %d\n", n_surf, length(ν))
@printf("Channels blind to the surface (< 0.01):           %d of %d\n", n_blind, length(ν))

# ── Vertical coverage ───────────────────────────────────────────────────────
# Counting how many channels peak in each layer shows what the band can and
# cannot resolve. Gaps here become the vertical resolution limits of any
# retrieval built on these channels — no amount of inversion invents
# information the weighting functions never carried.
println("\nHow many channels peak in each layer")
edges = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 120.0)]
peaks = [prof.altitude[argmax(abs.(view(K_T, i, :)))] for i in 1:length(ν)]
for (lo, hi) in edges
    n = count(z -> lo <= z < hi, peaks)
    @printf("  %5.1f–%5.1f km  %4d  %s\n", lo, hi, n, "*"^cld(n, 4))
end

println("\nNext: 03_retrieval_synthetic.jl — inverting these weighting functions.")
