using IRSounderLBL
using Plots
using Printf
using Statistics

mkpath("data")

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
        if !isfile(fpath)
            println("  Downloading $fname ...")
            fetch_hitran_api(mol_id, iso_id, Float64(ν_min), Float64(ν_max); outfile=fpath)
        end
        ll = load_hitran_par(fpath; ν_min=Float64(ν_min), ν_max=Float64(ν_max))
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), Float64(ν_min), Float64(ν_max))
    return HITRANLinelist(all_lines)
end

println("=== Loading HITRAN line data ===")
ll_raw = Dict{GasSpecies, HITRANLinelist}(
    CO2 => hitran_cached_multi(2, 1:3,  645.0, 2760.0, "co2_645_2760"),
    H2O => hitran_cached_multi(1, 1:3,  645.0, 2760.0, "h2o_645_2760"),
    O3  => hitran_cached_multi(3, 1:3,  980.0, 1090.0, "o3_980_1090"),
    N2O => hitran_cached_multi(4, 1:3, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => hitran_cached_multi(6, 1:3, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => hitran_cached_multi(5, 1:3, 2000.0, 2280.0, "co_2000_2280"),
)

println("=== Applying intensity filter ===")
linelists = Dict{GasSpecies, HITRANLinelist}(
    sp => filter_linelist(ll, S_MIN) for (sp, ll) in ll_raw
)
for (sp, ll) in linelists
    @printf("  %-4s : %d lines\n", string(sp), length(ll))
end

prof     = us_standard_atmosphere()
layers   = layer_properties(prof)
n_layers = length(layers.p_mid)
iasi     = IASIInstrument()

Δν_hi     = iasi.Δν / HRF
ν_hi_full = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)
n_ν       = ν_hi_full.n

_dz(Δp, p_mid, T) = 8.314462 * T / (0.028964 * 9.80665) * Δp / p_mid * 100.0

function lerp_resample(ν_src, R_src, ν_dst)
    n   = length(ν_src)
    out = Vector{Float64}(undef, length(ν_dst))
    for (i, ν) in enumerate(ν_dst)
        j = searchsortedfirst(ν_src, ν)
        if j == 1
            out[i] = R_src[1]
        elseif j > n
            out[i] = R_src[n]
        else
            α      = (ν - ν_src[j-1]) / (ν_src[j] - ν_src[j-1])
            out[i] = R_src[j-1] * (1 - α) + R_src[j] * α
        end
    end
    return out
end

# ── Forward model for one Voigt method ───────────────────────────────────────

function run_forward_model(method::VoigtMethod)
    label = string(method)
    println("\n=== Building τ cube — $label ===")
    τ = zeros(Float64, n_ν, n_layers)
    t0 = time()

    chunk_starts = iasi.ν_min:CHUNK:(iasi.ν_max - 1e-6)
    n_chunks     = length(chunk_starts)

    for (ic, ν_lo) in enumerate(chunk_starts)
        ν_hi_c = min(ν_lo + CHUNK, iasi.ν_max)
        i1 = searchsortedfirst(ν_hi_full.ν, ν_lo)
        i2 = searchsortedlast(ν_hi_full.ν,  ν_hi_c)
        i1 > i2 && continue

        chunk_grid = WavenumberGrid(ν_hi_full.ν[i1:i2], Δν_hi, i2 - i1 + 1)

        chunk_lls = Dict{GasSpecies, HITRANLinelist}()
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
                σ = compute_voigt_cross_sections(chunk_grid, ll_c, T_k, p_atm;
                                                  cutoff=CUTOFF, method=method)
                τ[i1:i2, k] .+= σ .* (vmr * Δp_k * N_AIR)
            end

            vmr_h2o = layers.vmr_mid[H2O][k]
            vmr_co2 = layers.vmr_mid[CO2][k]
            τ[i1:i2, k] .+= h2o_continuum(chunk_grid, vmr_h2o, layers.p_mid[k], T_k) .* dz
            τ[i1:i2, k] .+= co2_cia(chunk_grid, vmr_co2, layers.p_mid[k], T_k) .* dz
        end

        elapsed = round(time() - t0; digits=1)
        print("\r  chunk $ic/$n_chunks  ($(round(Int,ν_hi_c)) cm⁻¹)  $(elapsed)s   ")
    end
    println("\n  Done in $(round(time()-t0; digits=1)) s")

    Tsfc    = prof.temperature[1]
    R_hi    = schwarzschild_rte(ν_hi_full, τ, layers.T_mid, Tsfc)
    ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss)
    R_apod  = apply_ils(ν_hi_full, R_hi, ils_δν, ils_kern)
    ν_iasi  = iasi_grid(iasi)
    R_iasi  = lerp_resample(ν_hi_full.ν, R_apod, ν_iasi.ν)
    BT      = brightness_temperature(ν_iasi, R_iasi)
    @printf("  BT range: %.1f – %.1f K\n", minimum(BT), maximum(BT))
    return BT
end

# ── Run all three methods ─────────────────────────────────────────────────────

BT_fad = run_forward_model(FullFaddeeva)   # reference first
BT_wei = run_forward_model(Weideman)
BT_pv  = run_forward_model(PseudoVoigt)

ν_iasi = iasi_grid(iasi).ν

δBT_wei = BT_wei .- BT_fad
δBT_pv  = BT_pv  .- BT_fad

@printf("\nWeideman    – FullFaddeeva : max |ΔBT| = %.4f mK  rms = %.4f mK\n",
        maximum(abs, δBT_wei)*1e3, sqrt(mean(δBT_wei.^2))*1e3)
@printf("PseudoVoigt – FullFaddeeva : max |ΔBT| = %.4f mK  rms = %.4f mK\n",
        maximum(abs, δBT_pv)*1e3,  sqrt(mean(δBT_pv.^2))*1e3)

# ── Plot ──────────────────────────────────────────────────────────────────────
println("\n=== Generating plot ===")

Tsfc = prof.temperature[1]

features = [
    (667,   218, "CO₂\n15 µm"),
    (900,   293, "window\n8–12 µm"),
    (1042,  270, "O₃\n9.6 µm"),
    (1305,  276, "CH₄/N₂O\n7.7 µm"),
    (1600,  255, "H₂O\n6.3 µm"),
    (2143,  262, "CO\n4.7 µm"),
    (2350,  215, "CO₂\n4.3 µm"),
]

p1 = plot(
    ν_iasi, BT_fad;
    label       = "FullFaddeeva (reference)",
    color       = :black,
    lw          = 0.8,
    ylabel      = "Brightness Temperature (K)",
    title       = "IASI Simulated Spectrum — Voigt Method Comparison",
    xlims       = (645, 2760),
    ylims       = (185, 305),
    xticks      = 645:115:2760,
    yticks      = 190:10:300,
    legend      = :bottomright,
    grid        = true,
    gridalpha   = 0.25,
    framestyle  = :box,
    xformatter  = _ -> "",   # suppress x-axis labels on top panel
    bottom_margin = 0Plots.mm,
)
plot!(p1, ν_iasi, BT_wei;
      label = "Weideman",   color = :dodgerblue,  lw = 0.8, ls = :dash)
plot!(p1, ν_iasi, BT_pv;
      label = "PseudoVoigt", color = :orangered,   lw = 0.8, ls = :dot)

hline!(p1, [Tsfc]; lc=:firebrick, ls=:dash, lw=0.8,
       label="T_sfc = $(round(Int,Tsfc)) K")

for (ν_feat, bt_label, label) in features
    annotate!(p1, ν_feat, bt_label, text(label, :center, 7, :darkgray))
end

# difference panel — y-axis in mK
wei_max  = round(maximum(abs, δBT_wei)*1e3;  digits=1)
pv_max   = round(maximum(abs, δBT_pv)*1e3;   digits=1)
ylim_abs = max(wei_max, pv_max) * 1.2
ylim_abs = max(ylim_abs, 5.0)   # at least ±5 mK visible

p2 = plot(
    ν_iasi, δBT_wei .* 1e3;
    label       = "Weideman − FullFaddeeva",
    color       = :dodgerblue,
    lw          = 0.8,
    xlabel      = "Wavenumber (cm⁻¹)",
    ylabel      = "ΔBT (mK)",
    xlims       = (645, 2760),
    ylims       = (-ylim_abs, ylim_abs),
    xticks      = 645:115:2760,
    legend      = :bottomright,
    grid        = true,
    gridalpha   = 0.25,
    framestyle  = :box,
    top_margin  = 0Plots.mm,
)
plot!(p2, ν_iasi, δBT_pv .* 1e3;
      label = "PseudoVoigt − FullFaddeeva", color = :orangered, lw = 0.8)
hline!(p2, [0.0]; lc=:black, lw=0.6, ls=:dash, label=nothing)

fig = plot(p1, p2;
           layout      = grid(2, 1; heights=[0.72, 0.28]),
           size        = (1400, 700),
           dpi         = 200,
           background_color = :white,
           left_margin  = 10Plots.mm,
           right_margin =  4Plots.mm,
)

savefig(fig, "iasi_voigt_comparison.png")
println("Saved → iasi_voigt_comparison.png")
