"""
Export Julia BT (645–800 cm⁻¹) with CO2 first-order line mixing (VP_Y).

Matches arts_validation_cont_lm.py configuration:
  - H2O MT-CKD continuum ON
  - CO2 VP_Y line mixing (HITRAN 2020 relaxation matrix, stot_min=1e-25)
  - No ILS

Output: data/julia_bt_co2_15um_lm.csv

Run with:
  julia --project -t auto scripts/julia_bt_co2_lm_export.jl
"""

using RadiativeTransfer
using Printf

const NU_MIN  = 645.0
const NU_MAX  = 800.0
const DNU_HI  = 0.005       # high-res internal grid
const DNU_OUT = 0.25        # output grid (matches ARTS)
const CUTOFF  = 25.0
const LM_DIR  = "data/Line-mixing_HITRAN2020/data_new"

t0 = time()

# ── Load relmat ───────────────────────────────────────────────────────────────
println("Loading HITRAN relmat (VP_Y, 645–800 cm⁻¹)...")
@time relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
println("  $(length(relmat.bands)) LM bands loaded")

# ── Load linelists ────────────────────────────────────────────────────────────
println("Loading HITRAN linelists...")
function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(all_lines)
end

ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
ll_h2o = load_multi(1:3, NU_MIN, NU_MAX, "h2o_645_2760")
@printf("  CO2: %d lines   H2O: %d lines\n", length(ll_co2), length(ll_h2o))

# ── Atmosphere ────────────────────────────────────────────────────────────────
prof   = afgl_us_standard_50lev()
layers = layer_properties(prof)
n_layers = length(layers.p_mid)

# ── Grids ─────────────────────────────────────────────────────────────────────
ν_hi  = wavenumber_grid(NU_MIN, NU_MAX, DNU_HI)
ν_out = wavenumber_grid(NU_MIN, NU_MAX, DNU_OUT)
n_hi  = ν_hi.n

# ── Layer optical depths ──────────────────────────────────────────────────────
println("Computing τ (VP_Y CO2 + Voigt H2O + continua)...")
τ = zeros(Float64, n_hi, n_layers)
Nair = 2.1209e22   # molec/(cm²·hPa)

t1 = time()
for k in 1:n_layers
    # CO2: full HITRAN Voigt baseline + S-file LM dispersive perturbation
    vmr_co2 = get(layers.vmr_cg, CO2, nothing)
    if vmr_co2 !== nothing && vmr_co2[k] > 0
        p_atm = layers.p_cg[CO2][k] / 1013.25
        T_k   = layers.T_cg[CO2][k]
        σ_co2 = compute_voigt_lm_cross_sections(ν_hi, ll_co2, relmat, T_k, p_atm;
                                                  cutoff=CUTOFF)
        τ[:, k] .+= σ_co2 .* (vmr_co2[k] * layers.Δp[k] * Nair)
    end

    # H2O standard Voigt
    vmr_h2o = get(layers.vmr_cg, H2O, nothing)
    if vmr_h2o !== nothing && vmr_h2o[k] > 0
        p_atm = layers.p_cg[H2O][k] / 1013.25
        T_k   = layers.T_cg[H2O][k]
        σ_h2o = compute_voigt_cross_sections(ν_hi, ll_h2o, T_k, p_atm;
                                               vmr_self=vmr_h2o[k], cutoff=CUTOFF)
        τ[:, k] .+= σ_h2o .* (vmr_h2o[k] * layers.Δp[k] * Nair)
    end

    # H2O and CO2 continua
    vmr_h2o_mid = get(layers.vmr_mid, H2O, [0.0])[k]
    vmr_co2_mid = get(layers.vmr_mid, CO2, [4.15e-4])[k]
    # Geometric thickness from hydrostatic approximation
    g = 9.80665; Mair = 0.028964; R = 8.314462
    dz_cm = (R * layers.T_mid[k] / (Mair * g)) * (layers.Δp[k] / layers.p_mid[k]) * 100.0
    τ[:, k] .+= h2o_continuum(ν_hi, vmr_h2o_mid, layers.p_mid[k], layers.T_mid[k]) .* dz_cm
    τ[:, k] .+= co2_continuum(ν_hi, vmr_co2_mid, layers.p_mid[k], layers.T_mid[k]) .* dz_cm
end
@printf("  τ cube: %.1f s\n", time() - t1)

# ── Radiative transfer ────────────────────────────────────────────────────────
R_hi = schwarzschild_rte(ν_hi, τ, prof.temperature, prof.temperature[1])
BT_hi = [brightness_temperature(ν_hi.ν[i], R_hi[i]) for i in 1:n_hi]

# Interpolate from high-res to output grid (linear)
BT_out = map(ν_out.ν) do ν
    # find bracketing indices in ν_hi
    i = clamp(searchsortedlast(ν_hi.ν, ν), 1, n_hi - 1)
    t = (ν - ν_hi.ν[i]) / (ν_hi.ν[i+1] - ν_hi.ν[i])
    BT_hi[i] * (1.0 - t) + BT_hi[i+1] * t
end

# ── Save ──────────────────────────────────────────────────────────────────────
outpath = joinpath("data", "julia_bt_co2_15um_lm.csv")
open(outpath, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out.ν, BT_out)
        write(f, "$(round(ν, digits=4)),$(round(bt, digits=6))\n")
    end
end
@printf("Saved %d channels → %s\n", length(BT_out), outpath)
@printf("BT range: %.1f – %.1f K\n", minimum(BT_out), maximum(BT_out))
@printf("Total: %.1f s  (%d threads)\n", time() - t0, Threads.nthreads())
