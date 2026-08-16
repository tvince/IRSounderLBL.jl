# Example 3 — retrieving a temperature profile from a simulated spectrum
#
# A closed-loop (OSSE) retrieval. We invent a "true" atmosphere, simulate the
# spectrum an instrument would measure, add realistic noise, then throw the truth
# away and try to recover it by optimal estimation starting from a climatological
# first guess. Because we know the answer, we can measure exactly how much of it
# came back — and, just as importantly, where it did not.
#
# Everything runs through the same forward model used to make the observation,
# so this isolates the *inversion* from forward-model error. Real retrievals
# (example 4 and the validation branch) face both.
#
# Data needed: the default 15 µm line-list set (see 01_forward_spectrum.jl).
# Run:  julia --project=. -t auto scripts/examples/03_retrieval_synthetic.jl
# Time: ~4 min on an M1 Pro with 6 threads.

using IRSounderLBL
using LinearAlgebra
using Printf
using Random

linelists = try
    default_linelists()
catch err
    println(sprint(showerror, err))
    println("\nRun `data_status()` to see what is present and what is missing.")
    exit(1)
end

sounder = Sounder(ν_min = 645.0, ν_max = 800.0, Δν = 0.25,
                  opd_max = 2.0, fwhm_gauss = 0.5)

# A coarser internal grid than the 0.001 cm⁻¹ default, to keep the example to a
# few minutes. Truth and retrieval use the *same* setting, so the closed loop
# stays consistent; for production work leave `internal_dnu` at its default.
fm = (sounder = sounder, internal_dnu = 0.005)

# ── The "true" atmosphere ───────────────────────────────────────────────────
# US Standard with a warm bump in the lower troposphere and a cold tropopause —
# the kind of structure a climatological prior does not know about.
prior = afgl_us_standard_50lev()
nlev  = length(prior.pressure)

ΔT = [4.0 * exp(-((z - 3.0) / 3.0)^2) - 3.0 * exp(-((z - 12.0) / 2.5)^2)
      for z in prior.altitude]
truth = AtmosphericProfile(prior.pressure, prior.temperature .+ ΔT,
                           prior.altitude, prior.vmr)
T_sfc_true = truth.temperature[1] + 1.5      # a slightly warm skin

@printf("Truth departs from the prior by up to %+.1f K (warm) / %+.1f K (cold)\n",
        maximum(ΔT), minimum(ΔT))

# ── Simulate the measurement ────────────────────────────────────────────────
println("\nSimulating the observed spectrum ...")
t0 = time()
grid, _, y_clean = forward_model(truth, linelists; T_sfc = T_sfc_true, fm...)
ν = grid.ν
@printf("  %d channels in %.1f s\n", length(ν), time() - t0)

# Scene-dependent noise: the instrument's quoted NEΔT is referenced to a 280 K
# blackbody, but in brightness-temperature units the cold, opaque band-centre
# channels are genuinely noisier. `scene_nedt` applies that Planck conversion,
# and `scene_measurement_covariance` adds the off-diagonal correlation that L1C
# apodization introduces between neighbouring channels.
σ  = scene_nedt(ν, y_clean; nedt_280K = 0.25)
Se = scene_measurement_covariance(ν, y_clean; nedt_280K = 0.25,
                                  Δν_sample = sounder.Δν,
                                  fwhm_gauss = sounder.fwhm_gauss)
@printf("  noise σ: %.2f K at the band centre, %.2f K in the window\n",
        σ[argmin(abs.(ν .- 667.0))], σ[argmin(abs.(ν .- 795.0))])

# Draw the noise *from* Se rather than as white noise. Because apodization
# correlates neighbouring channels, an uncorrelated draw would be inconsistent
# with the Se handed to the retrieval, and χ² would come out far above 1 — the
# inversion would rightly complain that the residuals do not look like the noise
# it was told to expect.
Random.seed!(20260815)                        # reproducible draw
y = y_clean .+ cholesky(Se).L * randn(length(y_clean))

# ── Set up the inversion ────────────────────────────────────────────────────
spec = StateVectorSpec(nlev, GasSpecies[];
                       include_temperature = true,
                       include_tsfc        = true,
                       include_emissivity  = false)

# The prior: our best guess before seeing the spectrum, and how confident we are.
# `build_sa` correlates levels in log-pressure, so the retrieval is not free to
# put a spike on one level — a physically smooth profile is assumed.
xa = pack_state(spec, prior; T_sfc = prior.temperature[1])
Sa = build_sa(spec, prior; σ_T = 3.0, L_T = 1.0, σ_tsfc = 3.0)

x_true = pack_state(spec, truth; T_sfc = T_sfc_true)

println("\nRetrieving ...")
t0 = time()
r = optimal_estimation(y, spec, prior, linelists;
                       xa = xa, Sa = Sa, Se = Matrix(Se),
                       max_iter = 8, fm_kwargs = fm)
@printf("  %s in %d iterations, %.1f s\n",
        r.converged ? "converged" : "STOPPED (not converged)", r.n_iter, time() - t0)

# ── How well did it do? ─────────────────────────────────────────────────────
# Two numbers matter. χ²/n near 1 says the spectrum was fitted to within the
# noise — necessary, but not sufficient: a retrieval can fit the radiances
# beautifully and still get the profile wrong. The RMS against truth is the
# honest test, and it is only available because this is a closed loop.
rT       = spec.temp_range
rms_pri  = sqrt(sum(abs2, xa[rT] .- x_true[rT]) / nlev)
rms_post = sqrt(sum(abs2, r.x[rT] .- x_true[rT]) / nlev)

@printf("\nFit      χ²/n = %.2f  (1.0 ⇒ fitted to the noise)\n", r.chi2 / count(r.channel_mask))
@printf("Profile  prior RMS %.3f K → posterior RMS %.3f K  (%.0f%% better)\n",
        rms_pri, rms_post, 100 * (1 - rms_post / rms_pri))
@printf("Surface  truth %.2f K, prior %.2f K, retrieved %.2f K\n",
        T_sfc_true, xa[spec.tsfc_index], r.x[spec.tsfc_index])

# Degrees of freedom for signal: how many independent pieces of information the
# spectrum actually supplied. With 50 temperature levels retrieved from 621
# channels, DOF is far below 50 — the rest of the answer is the prior showing
# through. This is the honest measure of what was measured versus assumed.
@printf("\nInformation content\n")
@printf("  degrees of freedom : %.2f  (of %d retrieved parameters)\n", r.dof, spec.n)
@printf("  Shannon entropy    : %.2f bits\n", r.H)

# The averaging kernel diagonal shows *where* that information landed: near 1
# means the retrieval is reporting the measurement, near 0 means it is repeating
# the prior back to you.
A_diag = [r.A[i, i] for i in rT]
println("\nWhere the information is (averaging-kernel diagonal)")
println("     z (km)   p (hPa)     A_ii   prior err → posterior err")
for z_t in (0.0, 3.0, 6.0, 10.0, 15.0, 25.0, 40.0)
    i = argmin(abs.(prior.altitude .- z_t))
    @printf("  %8.1f  %8.1f   %6.3f   %5.2f K → %5.2f K\n",
            prior.altitude[i], prior.pressure[i], A_diag[i],
            sqrt(Sa[rT[i], rT[i]]), sqrt(r.S_hat[rT[i], rT[i]]))
end

println("\nThe posterior error is smallest in the mid-troposphere, where example 2's")
println("weighting functions were densest, and grows with height as the channels")
println("thin out. Note also that ~9 degrees of freedom are spread across 50")
println("levels: the retrieval returns a smooth profile, not 50 independent")
println("temperatures. A retrieval can only sharpen what the measurement resolves.")
