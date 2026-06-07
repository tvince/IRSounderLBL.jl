"""
Cross-test: feed LBLRTM's per-layer τ at 2390.235 cm⁻¹ (from IOD=1 ODint
output) into Julia's Schwarzschild linear-in-τ RT and see what BT comes
out.  Compare against:
  - Julia native BT @ 2390.235 (forward model): 264.16 K
  - LBLRTM IEMIT=1 BT @ 2390.235 (interpolated): 270.889 K

Three predictions are produced:
  (a) Julia τ + Julia Schwarzschild + Julia profile T levels
       → should match the FWD model 264.16 K (sanity check)
  (b) LBLRTM τ + Julia Schwarzschild + Julia profile T levels
       → tells us whether the 7 K residual lives in Julia's RT vs LBLRTM's
         (if (b) ≈ (a), it's the RT; if (b) ≈ 233 K, it's the τ)

Reads:
  data/lblrtm/instrument_2390_line.csv     -- Julia per-layer τ
  data/lblrtm/lblrtm_per_layer_tau_2390.csv-- LBLRTM per-layer τ (IOD=1)
  AFGL US Std 50-level profile             -- level T

Usage:
  julia --project scripts/bt_from_lblrtm_tau.jl
"""

using IRSounderLBL
using Printf
using DelimitedFiles

const NU0 = 2390.235

prof  = afgl_us_standard_50lev()
T_lev = prof.temperature
n_lay = length(T_lev) - 1
T_sfc = T_lev[1]

function load_taus()
    J = readdlm("data/lblrtm/instrument_2390_line.csv", ',', header=true)
    hJ = String.(vec(J[2])); rJ = J[1]
    τ_jul = Float64.(rJ[:, findfirst(==("tau_layer_jul"), hJ)])

    L = readdlm("data/lblrtm/lblrtm_per_layer_tau_2390.csv", ',', header=true)
    hL = String.(vec(L[2])); rL = L[1]
    τ_lbl = Float64.(rL[:, findfirst(==("tau_at_nu0"), hL)])

    @assert length(τ_jul) == n_lay
    @assert length(τ_lbl) == n_lay
    return τ_jul, τ_lbl
end

τ_jul, τ_lbl = load_taus()

# Schwarzschild linear-in-τ at single wavenumber NU0
function bt_at_nu(τ_layer::Vector{Float64})
    τ_above = zeros(Float64, n_lay + 1)
    for k in n_lay:-1:1
        τ_above[k] = τ_above[k + 1] + τ_layer[k]
    end
    R = 0.0
    for k in 1:n_lay
        T_bot = T_lev[k]; T_top = T_lev[k + 1]
        s     = τ_layer[k]
        ΔT_k  = exp(-τ_above[k+1]) - exp(-τ_above[k])
        ems   = exp(-s)
        # Same _lit_correction as src/Solver/schwarzschild.jl
        f_s   = s < 1e-4 ? s*(0.5 - s*(1.0/3.0 - s*0.125)) : (1.0 - ems)/s - ems
        C_k   = exp(-τ_above[k+1]) * f_s
        R += planck_radiance(NU0, T_top) * ΔT_k +
             (planck_radiance(NU0, T_bot) - planck_radiance(NU0, T_top)) * C_k
    end
    R += planck_radiance(NU0, T_sfc) * exp(-τ_above[1])   # surface
    return brightness_temperature(NU0, R)
end

bt_jul_jul_toon = bt_at_nu(τ_jul)   # Julia τ + Julia Toon RT
bt_lbl_jul_toon = bt_at_nu(τ_lbl)   # LBLRTM τ + Julia Toon RT

# ── CIM cross-test ──────────────────────────────────────────────────────────
# Read LBLRTM TBAR per layer (parsed from TAPE6) and use it as the layer
# mass-weighted T in Julia's CIM Padé source function. This isolates whether
# the residual +3.5 K (Julia CIM @ 50 levels vs LBLRTM) lives in
#   - the CIM transcription   (then BT_cim ≈ Julia CIM result 237 K — bug)
#   - or upstream τ/T_AVE     (then BT_cim ≈ LBLRTM 270.889 K — CIM is faithful).
function load_lblrtm_tbar()
    L = readdlm("data/lblrtm/lblrtm_layers_43um.csv", ',', header=true)
    hL = String.(vec(L[2])); rL = L[1]
    return Float64.(rL[:, findfirst(==("T_eff_K"), hL)])
end
T_avg_lbl = load_lblrtm_tbar()
@assert length(T_avg_lbl) == n_lay

# Same _cim_correction as src/Solver/schwarzschild.jl
@inline function _cim_corr(s::Float64)
    s < 0.06 && return s / 6.0
    ems = exp(-s)
    return 1.0 - 2.0 * (ems / (ems - 1.0) + 1.0 / s)
end

function bt_at_nu_cim(τ_layer::Vector{Float64}, T_avg::Vector{Float64})
    τ_above = zeros(Float64, n_lay + 1)
    for k in n_lay:-1:1
        τ_above[k] = τ_above[k + 1] + τ_layer[k]
    end
    R = 0.0
    for k in 1:n_lay
        T_avg_k = T_avg[k]; T_top = T_lev[k + 1]
        s       = τ_layer[k]
        ΔT_k    = exp(-τ_above[k+1]) - exp(-τ_above[k])   # = 𝒯[k+1]·(1−e^{−s})
        C_k     = ΔT_k * _cim_corr(s)                      # FIXED: include (1−tr) factor
        B_avg   = planck_radiance(NU0, T_avg_k)
        B_top   = planck_radiance(NU0, T_top)
        R += B_avg * ΔT_k + (B_top - B_avg) * C_k
    end
    R += planck_radiance(NU0, T_sfc) * exp(-τ_above[1])   # surface
    return brightness_temperature(NU0, R)
end

bt_lbl_jul_cim = bt_at_nu_cim(τ_lbl, T_avg_lbl)  # LBLRTM τ + LBLRTM TBAR + Julia CIM RT

println("=== BT at 2390.235 cm⁻¹ ===")
@printf("  (a) Julia τ    + Julia Toon RT                    = %.3f K   (FWD: 264.16)\n",
        bt_jul_jul_toon)
@printf("  (b) LBLRTM τ   + Julia Toon RT                    = %.3f K   (Δa: %+.3f)\n",
        bt_lbl_jul_toon, bt_lbl_jul_toon - bt_jul_jul_toon)
@printf("  (c) LBLRTM τ   + LBLRTM TBAR + Julia CIM RT       = %.3f K   (Δb: %+.3f)\n",
        bt_lbl_jul_cim, bt_lbl_jul_cim - bt_lbl_jul_toon)
@printf("  (d) LBLRTM IEMIT=1 measurement (target)           ≈ 270.889 K   (Δc: %+.3f)\n",
        270.889 - bt_lbl_jul_cim)
println("\nInterpretation:")
println("  (c)−(d) magnitude tells us how faithful my CIM transcription is, given")
println("  LBLRTM-side τ AND TBAR (so no upstream contribution).")
println("  - If |(c)−(d)| < 0.5 K  →  CIM is faithful; remaining +3.5 K in Julia FWD")
println("    is upstream (cg_temperature_mass ≠ LBLATM TBAR, or per-layer σ small diff).")
println("  - If (c) ≈ 237 K       →  CIM transcription is missing a term LBLRTM applies.")
