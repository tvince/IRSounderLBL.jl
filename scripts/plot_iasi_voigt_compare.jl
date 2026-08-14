using IRSounderLBL
using Plots
using Printf
using Statistics

mkpath("data")

S_MIN = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-23

const S_THRESH = Dict(
    CO2 => S_MIN, H2O => S_MIN, O3 => S_MIN,
    N2O => S_MIN, CH4 => S_MIN, CO => S_MIN,
)

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

println("=== Loading HITRAN line data (iso 1–3) ===")
ll_raw = Dict{GasSpecies, HITRANLinelist}(
    CO2 => hitran_cached_multi(2, 1:3,  645.0, 2760.0, "co2_645_2760"),
    H2O => hitran_cached_multi(1, 1:3,  645.0, 2760.0, "h2o_645_2760"),
    O3  => hitran_cached_multi(3, 1:3,  980.0, 1090.0, "o3_980_1090"),
    N2O => hitran_cached_multi(4, 1:3, 1200.0, 2310.0, "n2o_1200_2310"),
    CH4 => hitran_cached_multi(6, 1:3, 1200.0, 1800.0, "ch4_1200_1800"),
    CO  => hitran_cached_multi(5, 1:3, 2000.0, 2280.0, "co_2000_2280"),
)

println("\n=== Applying intensity filter ===")
linelists = Dict{GasSpecies, HITRANLinelist}()
for (sp, ll) in ll_raw
    filt = filter_linelist(ll, S_THRESH[sp])
    linelists[sp] = filt
    @printf("  %-4s : %6d → %5d lines  (S ≥ %.0e)\n",
            string(sp), length(ll), length(filt), S_THRESH[sp])
end

prof     = us_standard_atmosphere()
layers   = layer_properties(prof)
n_layers = length(layers.p_mid)
iasi     = IASIInstrument()

const HRF    = 2
const CUTOFF = 25.0
const CHUNK  = 100.0
const N_AIR  = 2.1209e22

Δν_hi     = iasi.Δν / HRF
ν_hi_full = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)
n_ν       = ν_hi_full.n

_dz(Δp, p_mid, T) = 8.314462 * T / (0.028964 * 9.80665) * Δp / p_mid * 100.0

# ── Build both τ cubes in a single pass ──────────────────────────────────────
println("\n=== Building τ cubes — Weideman & FullFaddeeva (chunked, HRF=$HRF) ===")
τ_w = zeros(Float64, n_ν, n_layers)   # Weideman
τ_f = zeros(Float64, n_ν, n_layers)   # FullFaddeeva
t0  = time()

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
            col = vmr * Δp_k * N_AIR

            σ_w = compute_voigt_cross_sections(chunk_grid, ll_c, T_k, p_atm;
                                               cutoff=CUTOFF, method=Weideman)
            σ_f = compute_voigt_cross_sections(chunk_grid, ll_c, T_k, p_atm;
                                               cutoff=CUTOFF, method=FullFaddeeva)
            τ_w[i1:i2, k] .+= σ_w .* col
            τ_f[i1:i2, k] .+= σ_f .* col
        end

        # Continua are method-independent — add once to both
        vmr_h2o = layers.vmr_mid[H2O][k]
        vmr_co2 = layers.vmr_mid[CO2][k]
        cont_h2o = h2o_continuum(chunk_grid, vmr_h2o, layers.p_mid[k], T_k) .* dz
        cont_co2 = co2_cia(chunk_grid, vmr_co2, layers.p_mid[k], T_k) .* dz
        τ_w[i1:i2, k] .+= cont_h2o .+ cont_co2
        τ_f[i1:i2, k] .+= cont_h2o .+ cont_co2
    end

    elapsed = round(time() - t0; digits=1)
    print("\r  chunk $ic/$n_chunks  ($(round(Int,ν_hi_c)) cm⁻¹)  $(elapsed)s elapsed   ")
end
println("\n  Done in $(round(time()-t0; digits=1)) s")

# ── RTE → ILS → IASI channels for both methods ───────────────────────────────
Tsfc = prof.temperature[1]

ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss)
ν_iasi = iasi_grid(iasi)

function to_iasi_bt(τ_cube)
    R_hi   = schwarzschild_rte(ν_hi_full, τ_cube, layers.T_mid, Tsfc)
    R_apod = apply_ils(ν_hi_full, R_hi, ils_δν, ils_kern)
    n_src  = length(ν_hi_full.ν)
    R_out  = Vector{Float64}(undef, length(ν_iasi.ν))
    for (i, ν) in enumerate(ν_iasi.ν)
        j = searchsortedfirst(ν_hi_full.ν, ν)
        if j == 1
            R_out[i] = R_apod[1]
        elseif j > n_src
            R_out[i] = R_apod[n_src]
        else
            α = (ν - ν_hi_full.ν[j-1]) / (ν_hi_full.ν[j] - ν_hi_full.ν[j-1])
            R_out[i] = R_apod[j-1] * (1 - α) + R_apod[j] * α
        end
    end
    return brightness_temperature(ν_iasi, R_out)
end

println("\n=== Running RTE + ILS for both methods ===")
BT_w = to_iasi_bt(τ_w)
BT_f = to_iasi_bt(τ_f)
ΔBT  = BT_w .- BT_f

@printf("Weideman   BT range: %.1f – %.1f K\n", minimum(BT_w), maximum(BT_w))
@printf("FullFaddeeva BT range: %.1f – %.1f K\n", minimum(BT_f), maximum(BT_f))
@printf("ΔBT (W–F)  range: %.3f – %.3f K   RMS: %.4f K\n",
        minimum(ΔBT), maximum(ΔBT), sqrt(mean(ΔBT.^2)))

# ── Plot ──────────────────────────────────────────────────────────────────────
println("\n=== Generating plot ===")

ν = ν_iasi.ν

p_top = plot(
    ν, BT_w;
    label       = "Weideman",
    lw          = 0.6,
    color       = :midnightblue,
    xlabel      = "",
    ylabel      = "Brightness Temperature (K)",
    title       = "IASI Simulated BT — Weideman vs FullFaddeeva",
    xlims       = (645, 2760),
    ylims       = (185, 305),
    xticks      = 645:115:2760,
    yticks      = 190:10:300,
    grid        = true,
    gridalpha   = 0.3,
    framestyle  = :box,
    legend      = :topright,
    background_color = :white,
    bottom_margin = 2Plots.mm,
    left_margin  = 10Plots.mm,
)
plot!(p_top, ν, BT_f;
      label = "FullFaddeeva",
      lw    = 0.6,
      color = :crimson,
      alpha = 0.75)
hline!(p_top, [Tsfc]; lc=:gray, ls=:dash, lw=0.8,
       label="T_sfc = $(round(Int,Tsfc)) K")

features = [
    (667,  218, "CO₂\n15 µm"),
    (900,  293, "window\n8–12 µm"),
    (1042, 270, "O₃\n9.6 µm"),
    (1305, 276, "CH₄/N₂O\n7.7 µm"),
    (1600, 255, "H₂O\n6.3 µm"),
    (2143, 262, "CO\n4.7 µm"),
    (2350, 215, "CO₂\n4.3 µm"),
]
for (ν_feat, bt_label, label) in features
    annotate!(p_top, ν_feat, bt_label, text(label, :center, 7, :darkgray))
end

# RMS label
rms_str = @sprintf("RMS ΔBT = %.4f K", sqrt(mean(ΔBT.^2)))
annotate!(p_top, 2700, 296, text(rms_str, :right, 8, :black))

δ_abs = maximum(abs.(ΔBT))
p_bot = plot(
    ν, ΔBT;
    label       = "Weideman − FullFaddeeva",
    lw          = 0.5,
    color       = :darkorange,
    xlabel      = "Wavenumber (cm⁻¹)",
    ylabel      = "ΔBT (K)",
    xlims       = (645, 2760),
    ylims       = (-1.05 * δ_abs, 1.05 * δ_abs),
    xticks      = 645:115:2760,
    grid        = true,
    gridalpha   = 0.3,
    framestyle  = :box,
    legend      = :topright,
    background_color = :white,
    top_margin  = 2Plots.mm,
    left_margin = 10Plots.mm,
)
hline!(p_bot, [0.0]; lc=:black, lw=0.8, ls=:dash, label=nothing)

p_combined = plot(p_top, p_bot;
    layout      = grid(2, 1; heights=[0.72, 0.28]),
    dpi         = 200,
    size        = (1400, 760),
)

outfile = "iasi_voigt_comparison.png"
savefig(p_combined, outfile)
println("Saved → $outfile")
