using RadiativeTransfer
using Plots
using Printf
using Statistics

# ── Shared setup ──────────────────────────────────────────────────────────────
const S_MIN  = 1e-23
const HRF    = 2
const CUTOFF = 25.0
const CHUNK  = 100.0
const N_AIR  = 2.1209e22

function hitran_cached_multi(mol_id, iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        ll = load_hitran_par(fpath; ν_min=Float64(ν_min), ν_max=Float64(ν_max))
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

_dz(Δp, p_mid, T) = 8.314462 * T / (0.028964 * 9.80665) * Δp / p_mid * 100.0

println("=== Loading HITRAN line data ===")
ll_raw = Dict{GasSpecies, HITRANLinelist}(
    CO2 => hitran_cached_multi(2, 1:3,  645.0, 2760.0, "co2_645_2760"),
    H2O => hitran_cached_multi(1, 1:3,  645.0, 2760.0, "h2o_645_2760"),
    O3  => hitran_cached_multi(3, 1:3,  980.0, 1090.0, "o3_980_1090"),
    N2O => hitran_cached_multi(4, 1:3, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => hitran_cached_multi(6, 1:3, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => hitran_cached_multi(5, 1:3, 2000.0, 2280.0, "co_2000_2280"),
)

S_THRESH = Dict(sp => S_MIN for sp in keys(ll_raw))
linelists = Dict(sp => filter_linelist(ll, S_THRESH[sp]) for (sp, ll) in ll_raw)

prof     = us_standard_atmosphere()
layers   = layer_properties(prof)
n_layers = length(layers.p_mid)
iasi     = IASIInstrument()

Δν_hi     = iasi.Δν / HRF
ν_hi_full = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)
n_ν       = ν_hi_full.n
ν_iasi    = iasi_grid(iasi)

function lerp_resample(ν_src, R_src, ν_dst)
    n = length(ν_src)
    out = Vector{Float64}(undef, length(ν_dst))
    for (i, ν) in enumerate(ν_dst)
        j = searchsortedfirst(ν_src, ν)
        if j == 1;       out[i] = R_src[1]
        elseif j > n;    out[i] = R_src[n]
        else
            α = (ν - ν_src[j-1]) / (ν_src[j] - ν_src[j-1])
            out[i] = R_src[j-1] * (1-α) + R_src[j] * α
        end
    end
    return out
end

# ── Core: build τ cube and compute BT ────────────────────────────────────────
function run_forward_model(label)
    println("\n=== Building τ cube: $label ===")
    τ = zeros(Float64, n_ν, n_layers)
    t0 = time()
    chunk_starts = iasi.ν_min:CHUNK:(iasi.ν_max - 1e-6)
    n_chunks = length(chunk_starts)

    for (ic, ν_lo) in enumerate(chunk_starts)
        ν_hi_c = min(ν_lo + CHUNK, iasi.ν_max)
        i1 = searchsortedfirst(ν_hi_full.ν, ν_lo)
        i2 = searchsortedlast(ν_hi_full.ν,  ν_hi_c)
        i1 > i2 && continue

        chunk_grid = WavenumberGrid(ν_hi_full.ν[i1:i2], Δν_hi, i2-i1+1)
        chunk_lls  = Dict{GasSpecies, HITRANLinelist}()
        for (sp, ll) in linelists
            sub = [l for l in ll.lines if (ν_lo - CUTOFF) <= l.wavenumber <= (ν_hi_c + CUTOFF)]
            isempty(sub) || (chunk_lls[sp] = HITRANLinelist(sub))
        end

        for k in 1:n_layers
            p_atm = layers.p_mid[k] / 1013.25
            T_k   = layers.T_mid[k]
            Δp_k  = layers.Δp[k]
            dz    = _dz(Δp_k, layers.p_mid[k], T_k)
            for (sp, ll_c) in chunk_lls
                vmr = haskey(layers.vmr_mid, sp) ? layers.vmr_mid[sp][k] : 0.0
                vmr == 0.0 && continue
                σ = compute_voigt_cross_sections(chunk_grid, ll_c, T_k, p_atm; cutoff=CUTOFF)
                τ[i1:i2, k] .+= σ .* (vmr * Δp_k * N_AIR)
            end
            vmr_h2o = layers.vmr_mid[H2O][k]
            vmr_co2 = layers.vmr_mid[CO2][k]
            τ[i1:i2, k] .+= h2o_continuum(chunk_grid, vmr_h2o, layers.p_mid[k], T_k) .* dz
            τ[i1:i2, k] .+= co2_continuum(chunk_grid, vmr_co2, layers.p_mid[k], T_k) .* dz
        end
        elapsed = round(time() - t0; digits=1)
        print("\r  chunk $ic/$n_chunks  ($(round(Int,ν_hi_c)) cm⁻¹)  $(elapsed)s   ")
    end
    @printf("\n  τ done in %.1f s\n", time() - t0)

    Tsfc  = prof.temperature[1]
    R_hi  = schwarzschild_rte(ν_hi_full, τ, layers.T_mid, Tsfc)
    ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss)
    R_apod = apply_ils(ν_hi_full, R_hi, ils_δν, ils_kern)
    R_iasi = lerp_resample(ν_hi_full.ν, R_apod, ν_iasi.ν)
    return brightness_temperature(ν_iasi, R_iasi)
end

# ── Run 1: TIPS-2024 ──────────────────────────────────────────────────────────
BT_tips = run_forward_model("TIPS-2024")

# ── Swap Q_ratio to old power-law ─────────────────────────────────────────────
println("\n=== Swapping Q_ratio to power-law approximation ===")
@eval RadiativeTransfer begin
    const _Q_POWERLAW_EXPONENT = Dict{Tuple{Int,Int}, Float64}(
        (1,1)=>1.5, (1,2)=>1.5, (1,3)=>1.5,
        (2,1)=>1.0, (2,2)=>1.0, (2,3)=>1.0, (2,4)=>1.0,
        (3,1)=>1.5, (4,1)=>1.0,
        (5,1)=>1.0, (5,2)=>1.0,
        (6,1)=>1.5, (6,2)=>1.5,
        (7,1)=>1.0, (9,1)=>1.5, (11,1)=>1.5,
    )
    function Q_ratio(mol_id::Int, iso_id::Int, T::Float64)
        α = get(_Q_POWERLAW_EXPONENT, (mol_id, iso_id), 1.0)
        return (T_REF / T)^α
    end
end

# ── Run 2: power-law ──────────────────────────────────────────────────────────
BT_old = run_forward_model("Power-law")

# ── Difference ───────────────────────────────────────────────────────────────
ΔBT = BT_tips .- BT_old
ν   = ν_iasi.ν

@printf("\nΔBT (TIPS-2024 − power-law):\n")
@printf("  min:  %+.4f K\n", minimum(ΔBT))
@printf("  max:  %+.4f K\n", maximum(ΔBT))
@printf("  RMS:  %.4f K\n",  sqrt(mean(ΔBT.^2)))
@printf("  mean: %+.4f K\n", mean(ΔBT))

# ── Plot ──────────────────────────────────────────────────────────────────────
p_top = plot(ν, BT_tips;
    label="TIPS-2024", color=:midnightblue, lw=0.5,
    ylabel="BT (K)", title="BT Spectrum: TIPS-2024 vs Power-law",
    xlims=(645,2760), ylims=(185,305), legend=:topright,
    grid=true, gridalpha=0.3, framestyle=:box)
plot!(p_top, ν, BT_old;
    label="Power-law (old)", color=:firebrick, lw=0.5, ls=:dash)

p_bot = plot(ν, ΔBT;
    label=nothing, color=:darkgreen, lw=0.5,
    xlabel="Wavenumber (cm⁻¹)", ylabel="ΔBT (K)",
    title=@sprintf("TIPS-2024 − Power-law  [min %+.3f K, max %+.3f K, RMS %.3f K]",
                   minimum(ΔBT), maximum(ΔBT), sqrt(mean(ΔBT.^2))),
    xlims=(645,2760), grid=true, gridalpha=0.3, framestyle=:box)
hline!(p_bot, [0.0]; lc=:black, lw=0.8, ls=:dot, label=nothing)

p = plot(p_top, p_bot;
    layout=(2,1), size=(1400,800), dpi=150,
    left_margin=8Plots.mm, bottom_margin=6Plots.mm)

savefig(p, "tips_comparison.png")
println("\nSaved → tips_comparison.png")
