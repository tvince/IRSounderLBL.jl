"""
Export Julia BT for comparison with ARTS (arts_validation.py).

Toggle APPLY_ILS and NU_MAX at the top of the file as needed.
Output: data/julia_bt_645_800.csv  (nu_cm1, BT_K at 0.25 cm⁻¹ grid)

Run with:
  julia --project -t auto scripts/julia_bt_export.jl
"""

using RadiativeTransfer
using Printf

const NU_MIN    = 645.0
const NU_MAX    = 2760.0
const DNU_OUT   = 0.25
const CUTOFF    = 25.0
const HRF             = 2           # oversampling vs 0.25 cm⁻¹ output; use 50 when APPLY_ILS=true
const APPLY_ILS       = false       # set true to convolve with IASI ILS before resampling
const APPLY_CONTINUUM = false       # set true to add H2O + CO2 MT-CKD continuum
const N_AIR     = 2.1209e22   # molec/(cm²·hPa)

t0 = time()

# ── Load linelists (all IASI species, iso 1–3) ───────────────────────────────
function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || (println("  SKIP missing: $fname"); continue)
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(all_lines)
end

println("Loading HITRAN linelists...")
linelists = Dict{GasSpecies, HITRANLinelist}(
    CO2 => load_multi(1:3,  645.0, 2760.0, "co2_645_2760"),
    H2O => load_multi(1:3,  645.0, 2760.0, "h2o_645_2760"),
    O3  => load_multi(1:3,  980.0, 1090.0, "o3_980_1090"),
    N2O => load_multi(1:3, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => load_multi(1:3, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => load_multi(1:3, 2000.0, 2280.0, "co_2000_2280"),
)
for (sp, ll) in sort(collect(linelists), by=x->string(x[1]))
    @printf("  %-4s: %d lines\n", string(sp), length(ll))
end

# ── Atmosphere ───────────────────────────────────────────────────────────────
prof   = us_standard_atmosphere()
layers = layer_properties(prof)
n_lev  = length(layers.p_mid)

# ── High-res grid ────────────────────────────────────────────────────────────
Δν_hi  = DNU_OUT / HRF
ν_grid = wavenumber_grid(NU_MIN, NU_MAX, Δν_hi)
n_ν    = ν_grid.n

_dz(Δp, p_mid, T) = 8.314462 * T / (0.028964 * 9.80665) * Δp / p_mid * 100.0

# ── Optical depth cube ───────────────────────────────────────────────────────
println("Building τ cube...")
τ = zeros(Float64, n_ν, n_lev)
t1 = time()
for k in 1:n_lev
    Δp_k = layers.Δp[k]
    T_k  = layers.T_mid[k]   # mid-layer T for Planck source function

    # Line-by-line: Curtis-Godson effective VMR, pressure, and temperature
    for (sp, ll) in linelists
        vmr = haskey(layers.vmr_cg, sp) ? layers.vmr_cg[sp][k] : 0.0
        vmr == 0.0 && continue
        p_atm = layers.p_cg[sp][k] / 1013.25
        T_sp  = layers.T_cg[sp][k]
        vmr_s = (sp == H2O) ? vmr : 0.0
        σ = compute_voigt_cross_sections(ν_grid, ll, T_sp, p_atm; cutoff=CUTOFF, vmr_self=vmr_s)
        τ[:, k] .+= σ .* (vmr * Δp_k * N_AIR)
    end

    if APPLY_CONTINUUM
        vmr_h2o = layers.vmr_mid[H2O][k]
        vmr_co2 = layers.vmr_mid[CO2][k]
        dz = _dz(Δp_k, layers.p_mid[k], T_k)
        τ[:, k] .+= h2o_continuum(ν_grid, vmr_h2o, layers.p_mid[k], T_k) .* dz
        τ[:, k] .+= co2_continuum(ν_grid, vmr_co2, layers.p_mid[k], T_k) .* dz
    end
end
@printf("  τ done in %.1f s\n", time() - t1)

# ── RTE → (ILS →) BT ─────────────────────────────────────────────────────────
Tsfc = prof.temperature[1]
R_hi = schwarzschild_rte(ν_grid, τ, prof.temperature, Tsfc)
if APPLY_ILS
    ils_δν, ils_k = ils_kernel(Δν_hi, 2.0, 0.5)   # IASI: opd_max=2 cm, fwhm=0.5 cm⁻¹
    R_hi = apply_ils(ν_grid, R_hi, ils_δν, ils_k)
end
BT_hi = brightness_temperature(ν_grid, R_hi)

# Resample to 0.25 cm⁻¹ output grid (linear interp)
ν_out  = collect(NU_MIN:DNU_OUT:NU_MAX)
BT_out = [let j = searchsortedfirst(ν_grid.ν, ν)
              j == 1 ? BT_hi[1] : j > n_ν ? BT_hi[n_ν] :
              let α = (ν - ν_grid.ν[j-1]) / (ν_grid.ν[j] - ν_grid.ν[j-1])
                  BT_hi[j-1] * (1 - α) + BT_hi[j] * α
              end
          end for ν in ν_out]

@printf("BT range: %.1f – %.1f K  (n=%d)\n", minimum(BT_out), maximum(BT_out), length(BT_out))

# ── Save ─────────────────────────────────────────────────────────────────────
outpath = joinpath("data", "julia_bt_645_800.csv")
open(outpath, "w") do f
    write(f, "nu_cm1,BT_K\n")
    for (ν, bt) in zip(ν_out, BT_out)
        write(f, "$(round(ν, digits=4)),$(round(bt, digits=6))\n")
    end
end
println("Saved → $outpath  ($(length(ν_out)) channels)")
@printf("Total: %.1f s  (%d threads)\n", time() - t0, Threads.nthreads())
