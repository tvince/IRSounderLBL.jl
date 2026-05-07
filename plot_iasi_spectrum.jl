using RadiativeTransfer
using Plots
using Printf

mkpath("data")

# ── Intensity threshold (cm/molec at 296 K) ───────────────────────────────────
# Pass as first command-line argument to override, e.g.:
#   julia --project plot_iasi_spectrum.jl 1e-23
S_MIN = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1e-23

const S_THRESH = Dict(
    CO2 => S_MIN,
    H2O => S_MIN,
    O3  => S_MIN,
    N2O => S_MIN,
    CH4 => S_MIN,
    CO  => S_MIN,
)

# ── Fetch and cache HITRAN .par files (multiple isotopologues) ────────────────
# iso_id=1 reuses existing files named without suffix; iso 2 & 3 get _iso2/_iso3 suffix.
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
    s_min = S_THRESH[sp]
    filt  = filter_linelist(ll, s_min)
    linelists[sp] = filt
    @printf("  %-4s : %6d → %5d lines  (S ≥ %.0e)\n",
            string(sp), length(ll), length(filt), s_min)
end
println("  Total: $(sum(length(v) for v in values(linelists))) lines")

# ── Atmosphere and grid ───────────────────────────────────────────────────────
prof     = us_standard_atmosphere()
layers   = layer_properties(prof)
n_layers = length(layers.p_mid)
iasi     = IASIInstrument()

const HRF    = 2          # spectral oversampling factor
const CUTOFF = 25.0       # cm⁻¹ line-wing cutoff
const CHUNK  = 100.0      # cm⁻¹ per spectral chunk
const N_AIR  = 2.1209e22  # molec/(cm²·hPa) — hydrostatic column conversion

Δν_hi      = iasi.Δν / HRF
ν_hi_full  = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)
n_ν        = ν_hi_full.n

# Hydrostatic layer thickness in cm
_dz(Δp, p_mid, T) = 8.314462 * T / (0.028964 * 9.80665) * Δp / p_mid * 100.0

# ── Chunked optical depth build ───────────────────────────────────────────────
println("\n=== Building τ cube (chunked, HRF=$HRF) ===")
τ  = zeros(Float64, n_ν, n_layers)
t0 = time()

chunk_starts = iasi.ν_min:CHUNK:(iasi.ν_max - 1e-6)
n_chunks     = length(chunk_starts)

for (ic, ν_lo) in enumerate(chunk_starts)
    ν_hi_c = min(ν_lo + CHUNK, iasi.ν_max)

    # Indices in the full high-res grid
    i1 = searchsortedfirst(ν_hi_full.ν, ν_lo)
    i2 = searchsortedlast(ν_hi_full.ν,  ν_hi_c)
    i1 > i2 && continue

    chunk_grid = WavenumberGrid(ν_hi_full.ν[i1:i2], Δν_hi, i2 - i1 + 1)

    # Subset each species linelist to lines within this chunk ± cutoff
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

        # Line absorption
        for (sp, ll_c) in chunk_lls
            vmr = haskey(layers.vmr_mid, sp) ? layers.vmr_mid[sp][k] : 0.0
            vmr == 0.0 && continue
            σ = compute_voigt_cross_sections(chunk_grid, ll_c, T_k, p_atm; cutoff=CUTOFF)
            τ[i1:i2, k] .+= σ .* (vmr * Δp_k * N_AIR)
        end

        # Continua
        vmr_h2o = layers.vmr_mid[H2O][k]
        vmr_co2 = layers.vmr_mid[CO2][k]
        τ[i1:i2, k] .+= h2o_continuum(chunk_grid, vmr_h2o, layers.p_mid[k], T_k) .* dz
        τ[i1:i2, k] .+= co2_continuum(chunk_grid, vmr_co2, layers.p_mid[k], T_k) .* dz
    end

    elapsed = round(time() - t0; digits=1)
    print("\r  chunk $ic/$n_chunks  ($(round(Int,ν_hi_c)) cm⁻¹)  $(elapsed)s elapsed   ")
end
println("\n  Done in $(round(time()-t0;digits=1)) s")

# ── Radiative transfer → ILS → IASI channels ─────────────────────────────────
Tsfc   = prof.temperature[1]
R_hi   = schwarzschild_rte(ν_hi_full, τ, layers.T_mid, Tsfc)

ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss)
R_apod = apply_ils(ν_hi_full, R_hi, ils_δν, ils_kern)

# Linear resample onto IASI 0.25 cm⁻¹ grid
ν_iasi = iasi_grid(iasi)
function lerp_resample(ν_src, R_src, ν_dst)
    n = length(ν_src)
    out = Vector{Float64}(undef, length(ν_dst))
    for (i, ν) in enumerate(ν_dst)
        j = searchsortedfirst(ν_src, ν)
        if j == 1
            out[i] = R_src[1]
        elseif j > n
            out[i] = R_src[n]
        else
            α = (ν - ν_src[j-1]) / (ν_src[j] - ν_src[j-1])
            out[i] = R_src[j-1] * (1 - α) + R_src[j] * α
        end
    end
    return out
end

R_iasi  = lerp_resample(ν_hi_full.ν, R_apod, ν_iasi.ν)
BT_iasi = brightness_temperature(ν_iasi, R_iasi)

@printf("\nBT range: %.1f – %.1f K\n", minimum(BT_iasi), maximum(BT_iasi))

# ── Plot ──────────────────────────────────────────────────────────────────────
println("\n=== Generating plot ===")

p = plot(
    ν_iasi.ν, BT_iasi;
    xlabel      = "Wavenumber (cm⁻¹)",
    ylabel      = "Brightness Temperature (K)",
    title       = "IASI Simulated Spectrum — US Standard Atmosphere (1976)",
    label       = nothing,
    lw          = 0.5,
    color       = :midnightblue,
    dpi         = 200,
    size        = (1400, 520),
    xlims       = (645, 2760),
    ylims       = (185, 305),
    xticks      = 645:115:2760,
    yticks      = 190:10:300,
    grid        = true,
    gridalpha   = 0.3,
    framestyle  = :box,
    background_color = :white,
    left_margin = 8Plots.mm,
    bottom_margin = 6Plots.mm,
)

# Surface-temperature reference line
hline!(p, [Tsfc]; lc=:firebrick, ls=:dash, lw=1.0,
       label="T_sfc = $(round(Int,Tsfc)) K")

# Annotate major spectral features
features = [
    (667,   218, "CO₂\n15 µm"),
    (900,   293, "window\n8–12 µm"),
    (1042,  270, "O₃\n9.6 µm"),
    (1305,  276, "CH₄/N₂O\n7.7 µm"),
    (1600,  255, "H₂O\n6.3 µm"),
    (2143,  262, "CO\n4.7 µm"),
    (2350,  215, "CO₂\n4.3 µm"),
]

for (ν_feat, bt_label, label) in features
    annotate!(p, ν_feat, bt_label,
              text(label, :center, 7, :darkgray))
end

savefig(p, "iasi_bt_spectrum.png")
println("Saved → iasi_bt_spectrum.png")
