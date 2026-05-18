"""
Diagnostic plot of the CO2 line-mixing split:
    σ_total = σ_voigt + σ_disp

Shows the Voigt baseline (HITRAN), the dispersive perturbation (S-file LM),
and their sum across 645–800 cm⁻¹.  Mid-tropospheric conditions (T=250 K,
p=0.5 atm) — chosen to make Q(2,4,6) clip ON (typical IASI weighting region).

Output: data/lm_split_diagnostic.png
"""

using RadiativeTransfer
using Plots
using Printf

const NU_MIN = 645.0
const NU_MAX = 800.0
const DNU    = 0.01
const T      = 250.0
const P_ATM  = 0.5

println("Loading data...")
ν = wavenumber_grid(NU_MIN, NU_MAX, DNU)
relmat = load_hitran_relmat("data/Line-mixing_HITRAN2020/data_new", NU_MIN, NU_MAX;
                            stot_min=1e-25)
ll_co2 = HITRANLinelist(load_hitran_par("data/co2_645_2760.par";
                                          ν_min=NU_MIN - 25.0,
                                          ν_max=NU_MAX + 25.0).lines)
@printf "  %d LM bands, %d HITRAN CO2 lines\n" length(relmat.bands) length(ll_co2)

println("Computing cross sections (T=$T K, p=$P_ATM atm)...")
@time σ_voigt = compute_voigt_cross_sections(ν, ll_co2, T, P_ATM)
@time σ_disp  = compute_lm_dispersive_correction(ν, relmat, T, P_ATM)
σ_sum_raw     = σ_voigt .+ σ_disp                  # unclamped (can be negative)
σ_total       = max.(σ_sum_raw, 0.0)               # what the wrapper returns
clamp_mask    = σ_sum_raw .< 0
n_clamped     = count(clamp_mask)
@printf "  clamped %d / %d grid points (%.3f%%)\n" n_clamped ν.n 100*n_clamped/ν.n

# ── Top panel: full cross section (log scale) ────────────────────────────────
σ_floor = 1e-26   # cosmetic floor for log plot
p1 = plot(ν.ν, max.(σ_voigt, σ_floor);
          label="σ_voigt (HITRAN baseline)",
          yscale=:log10, ylabel="σ  (cm²/molec)",
          xlims=(NU_MIN, NU_MAX), ylims=(1e-25, 1e-17),
          lw=1.2, color=:steelblue, legend=:bottomright,
          title=@sprintf("CO2 line-mixing split — T=%g K, p=%g atm", T, P_ATM))
plot!(p1, ν.ν, max.(σ_total, σ_floor);
      label="σ_total = σ_voigt + σ_disp", lw=1.0, color=:firebrick, alpha=0.8)

# Mark clamped points
if n_clamped > 0
    scatter!(p1, ν.ν[clamp_mask], fill(σ_floor * 1.5, n_clamped);
             label="clamped to 0 ($(n_clamped) pts)", ms=2, color=:black, msw=0)
end

# ── Bottom panel: dispersive perturbation (signed, linear) ───────────────────
p2 = plot(ν.ν, σ_disp;
          label="σ_disp (LM perturbation)",
          xlabel="wavenumber (cm⁻¹)", ylabel="σ_disp  (cm²/molec)",
          xlims=(NU_MIN, NU_MAX),
          lw=0.8, color=:darkorange, legend=:topright)
hline!(p2, [0.0]; color=:gray, ls=:dash, label="")

# ── Combine and save ──────────────────────────────────────────────────────────
plt = plot(p1, p2; layout=(2, 1), size=(1100, 700), link=:x,
           left_margin=8Plots.mm, bottom_margin=4Plots.mm)

outpath = joinpath("data", "lm_split_diagnostic.png")
savefig(plt, outpath)
@printf "Saved %s\n" outpath

# ── Numeric summary ───────────────────────────────────────────────────────────
i_peak = argmax(σ_voigt)
i_672  = argmin(abs.(ν.ν .- 672.0))
@printf "\nKey points:\n"
@printf "  Q-branch peak ν=%.2f:  σ_voigt=%.3e  σ_disp/σ_voigt=%+.4f\n" ν.ν[i_peak] σ_voigt[i_peak] σ_disp[i_peak]/σ_voigt[i_peak]
@printf "  Old σ=0 spot ν=%.2f:   σ_voigt=%.3e  σ_disp=%.3e  σ_total=%.3e\n" ν.ν[i_672] σ_voigt[i_672] σ_disp[i_672] σ_total[i_672]
