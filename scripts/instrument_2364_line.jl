"""
Per-layer instrumentation of the worst point in the +5..+7 K Julia−LBLRTM
comb: the line core at 2364.105 cm⁻¹ (within the CO₂ ν₃ band-head where
many Q-branch lines overlap).

For each of the 49 layers we compute the TOTAL summed CO₂ cross-section at
exactly ν = 2364.105 cm⁻¹ (including all line wings within ±25 cm⁻¹), times
the layer's CO₂ column. Then we accumulate τ from TOA downward, find the
saturation altitude, and integrate the linear-in-τ source function to predict
the line-core BT at exactly 2364.105 cm⁻¹.

Two runs of the same recipe:
  (a) Julia native:  layer (p_cg, T_cg, vmr_cg, N_CO2) from layer_properties
  (b) Julia formulas + LBLRTM atmosphere:  (p, T, N_CO2) substituted in from
                                            data/lblrtm/lblrtm_layers_43um.csv

Both use the identical Julia cross-section pipeline.  Δ(LBLRTM atmosphere -
Julia atmosphere) isolates how much of the BT comb is explained by layer-
properties divergence vs Julia's internal recipe.

CSV dump: data/lblrtm/instrument_2364_line.csv

Usage:
  julia --project -t auto scripts/instrument_2364_line.jl
"""

using IRSounderLBL
using Printf
using DelimitedFiles

const NU0     = 2364.105        # cm⁻¹
const CUTOFF  = 25.0
const OUT     = "data/lblrtm/instrument_2364_line.csv"
const LBL_CSV = "data/lblrtm/lblrtm_layers_43um.csv"

# Tiny single-point grid at exactly NU0
const NU_GRID = wavenumber_grid(NU0, NU0, 1.0)   # one point at 2364.105

# ── Load profile and layer properties ────────────────────────────────────
prof   = afgl_us_standard_50lev()
layers = layer_properties(prof)
n_lay  = length(layers.p_mid)
z_lev  = prof.altitude
T_lev  = prof.temperature

println("Loading CO2 linelist (iso 1–3) over $(NU0-CUTOFF)–$(NU0+CUTOFF) cm⁻¹…")
function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end
ll_co2 = load_multi(1:3, NU0, NU0, "co2_645_2760")
@printf("  CO2: %d lines in ±25 cm⁻¹ window\n", length(ll_co2))

# ── Load LBLRTM TAPE6 layer values ──────────────────────────────────────
M = readdlm(LBL_CSV, ',', header=true)
hdr = String.(vec(M[2]))
col(name) = M[1][:, findfirst(==(name), hdr)]
lbl_p    = Float64.(col("p_eff_hPa"))
lbl_T    = Float64.(col("T_eff_K"))
lbl_NCO2 = Float64.(col("N_CO2_cm2"))
@assert length(lbl_p) == n_lay "LBLRTM layer count mismatch"

# ── Per-layer summed cross-section at NU0 ───────────────────────────────
"""
Compute σ(NU0) summed over all CO₂ lines within ±25 cm⁻¹, at one (p_hPa, T_K).
"""
function sigma_at_nu0(p_hPa::Float64, T::Float64, vmr_self::Float64)::Float64
    σ = compute_voigt_cross_sections(NU_GRID, ll_co2, T, p_hPa/1013.25;
                                      vmr_self=vmr_self,
                                      cutoff=CUTOFF)
    return σ[1]
end

function instrument(p_eff, T_eff, vmr_eff, N_CO2_layer, label)
    println("\n── $label ────────────────────────────────────")
    σ_arr     = Vector{Float64}(undef, n_lay)
    τ_layer   = Vector{Float64}(undef, n_lay)
    print("  computing per-layer σ(2364.105) and τ … ")
    t0 = time()
    for k in 1:n_lay
        σ_arr[k]   = sigma_at_nu0(p_eff[k], T_eff[k], vmr_eff[k])
        τ_layer[k] = σ_arr[k] * N_CO2_layer[k]
    end
    @printf("done in %.1f s\n", time() - t0)

    # Cumulative τ from TOA. τ_above[k] = optical depth above level k (looking
    # down from TOA to the bottom of layer k - 1 = top of layer k below).
    # Indexing: τ_above is length n_lay+1; τ_above[n_lay+1] = 0 at TOA;
    # τ_above[k] = τ_above[k+1] + τ_layer[k] adds the layer just above level k.
    τ_above = Vector{Float64}(undef, n_lay + 1)
    τ_above[n_lay + 1] = 0.0
    for k in n_lay:-1:1
        τ_above[k] = τ_above[k + 1] + τ_layer[k]
    end

    # Saturation altitude: HIGHEST level (largest k) at which τ_above ≥ 1.
    k_sat = something(findlast(k -> τ_above[k] >= 1.0, 1:n_lay+1), 1)
    @printf("  total column τ at NU0    = %.2e\n", τ_above[1])
    @printf("  saturation level         = %d   z=%.2f km   T=%.2f K   τ_above=%.3f\n",
            k_sat, z_lev[k_sat], T_lev[k_sat], τ_above[k_sat])

    # Schwarzschild linear-in-τ source function at one wavenumber.
    R = 0.0
    for k in 1:n_lay
        T_bot = T_lev[k]; T_top = T_lev[k + 1]
        s     = τ_layer[k]
        ΔT_k  = exp(-τ_above[k+1]) - exp(-τ_above[k])
        ems   = exp(-s)
        C_k   = exp(-τ_above[k+1]) * (s < 1e-4 ? s*(0.5 - s*(1.0/3.0 - s*0.125)) :
                                                (1.0 - ems)/s - ems)
        R += planck_radiance(NU0, T_top) * ΔT_k +
             (planck_radiance(NU0, T_bot) - planck_radiance(NU0, T_top)) * C_k
    end
    # surface (blackbody, ε=1): B(T_sfc) × exp(-τ_above[1])
    R += planck_radiance(NU0, T_lev[1]) * exp(-τ_above[1])
    BT = brightness_temperature(NU0, R)
    @printf("  predicted line-core BT   = %.3f K\n", BT)

    return (σ=σ_arr, τ_layer=τ_layer, τ_above=τ_above, BT=BT,
            k_sat=k_sat)
end

# Native Julia
jul = instrument(
    [layers.p_cg[CO2][k]                                          for k in 1:n_lay],
    [layers.T_cg[CO2][k]                                          for k in 1:n_lay],
    [layers.vmr_cg[CO2][k]                                        for k in 1:n_lay],
    [layers.vmr_cg[CO2][k] * layers.Δp[k] * 2.1209e22             for k in 1:n_lay],
    "Julia native (p_cg, T_cg, N_CO2 from layer_properties)")

# Julia formulas + LBLRTM atmosphere
lbl = instrument(
    Vector{Float64}(lbl_p),
    Vector{Float64}(lbl_T),
    [layers.vmr_cg[CO2][k] for k in 1:n_lay],   # vmr is same
    Vector{Float64}(lbl_NCO2),
    "Julia formulas + LBLRTM TBAR/PBAR/N_CO2 substituted")

# ── CSV dump ─────────────────────────────────────────────────────────────
open(OUT, "w") do f
    write(f, "layer,z_from_km,z_to_km,p_jul,T_jul,N_jul,sigma_jul,tau_layer_jul,tau_above_jul,"
           * "p_lbl,T_lbl,N_lbl,sigma_lbl,tau_layer_lbl,tau_above_lbl\n")
    for k in 1:n_lay
        write(f, @sprintf("%d,%.3f,%.3f,%.6e,%.4f,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%.6e,%.6e,%.6e,%.6e\n",
            k, z_lev[k], z_lev[k+1],
            layers.p_cg[CO2][k], layers.T_cg[CO2][k],
            layers.vmr_cg[CO2][k] * layers.Δp[k] * 2.1209e22,
            jul.σ[k], jul.τ_layer[k], jul.τ_above[k+1],
            lbl_p[k], lbl_T[k], lbl_NCO2[k],
            lbl.σ[k], lbl.τ_layer[k], lbl.τ_above[k+1]))
    end
end
println("\nWrote $OUT  ($n_lay layers)")

println("\n=== Summary: line core BT at 2364.105 cm⁻¹ ===")
@printf("  Julia native              : %.3f K   (saturation z=%.2f km, T=%.2f K)\n",
        jul.BT, z_lev[jul.k_sat], T_lev[jul.k_sat])
@printf("  Julia + LBLRTM atmosphere : %.3f K   (saturation z=%.2f km, T=%.2f K)\n",
        lbl.BT, z_lev[lbl.k_sat], T_lev[lbl.k_sat])
@printf("  Δ (LBL_atm − Jul_atm)     : %+.3f K\n", lbl.BT - jul.BT)
@printf("  Measured Julia BT @ 2364  : 205.69 K\n")
@printf("  Measured LBLRTM BT @ 2364 : ~198 K\n")
println("\nIf Δ ≈ 0, the comb is NOT driven by layer-property differences and the")
println("residual lives in LBLRTM's internal cross-section pipeline (TIPS, broadening,")
println("Voigt normalization, line-coupling beyond what we model, …).")
