"""
IASI instrument model.

IASI (Infrared Atmospheric Sounding Interferometer) on MetOp-A/B/C:
- Spectral range:  645–2760 cm⁻¹
- Spectral sampling: 0.25 cm⁻¹ (after apodization)
- Number of channels: 8461
- Native OPD: 2.0 cm (unapodized Δν = 0.25 cm⁻¹)
- Apodization: Norton-Beer Medium
- Pixel size at nadir: 12 km diameter (circular footprint)
- Swath: ±48.33° (≈ 2200 km)
"""

"""
    IASIInstrument

IASI instrument specification.

# Fields
- `ν_min`:       minimum wavenumber (cm⁻¹) = 645.0
- `ν_max`:       maximum wavenumber (cm⁻¹) = 2760.0
- `Δν`:          spectral sampling (cm⁻¹) = 0.25
- `n_channels`:  number of spectral channels = 8461
- `opd_max`:     maximum OPD (cm) = 2.0; defines the sinc ILS width
- `fwhm_gauss`:  FWHM of the Gaussian broadening convolved with the sinc (cm⁻¹) = 0.5
"""
struct IASIInstrument
    ν_min::Float64
    ν_max::Float64
    Δν::Float64
    n_channels::Int
    opd_max::Float64
    fwhm_gauss::Float64
end

"""
    IASIInstrument(; fwhm_gauss=0.5)

Construct the nominal IASI instrument (MetOp-A/B/C specification).
The ILS is modelled as sinc(2π opd_max δν) convolved with a Gaussian of
width `fwhm_gauss` cm⁻¹.
"""
function IASIInstrument(; fwhm_gauss::Float64 = 0.5)
    ν_min = 645.0
    ν_max = 2760.0
    Δν    = 0.25
    n_ch  = round(Int, (ν_max - ν_min) / Δν) + 1   # = 8461
    return IASIInstrument(ν_min, ν_max, Δν, n_ch, 2.0, fwhm_gauss)
end

"""
    iasi_grid(iasi) -> WavenumberGrid

Return the IASI spectral grid as a `WavenumberGrid`.
"""
function iasi_grid(iasi::IASIInstrument)
    return wavenumber_grid(iasi.ν_min, iasi.ν_max, iasi.Δν)
end

"""
    iasi_forward_model(prof, linelists;
                       iasi=IASIInstrument(),
                       geom=nadir_geometry(),
                       T_sfc=nothing,
                       ε_sfc=1.0,
                       high_res_factor=4,
                       cutoff=25.0,
                       backend=CPU()) -> (ν_iasi, R_iasi, BT_iasi)

Full IASI forward model: atmosphere → level optical depths → RTE → ILS → IASI L1C.

# Arguments
- `prof`:           `AtmosphericProfile` for the column
- `linelists`:      `Dict{GasSpecies, HITRANLinelist}` for all species
- `iasi`:           `IASIInstrument` specification
- `geom`:           `ViewingGeometry`
- `T_sfc`:          surface skin temperature (K); defaults to profile[1]
- `ε_sfc`:          surface emissivity
- `high_res_factor`:internal spectral over-sampling relative to IASI Δν
- `cutoff`:         Voigt wing cutoff (cm⁻¹)
- `backend`:        compute backend

# Returns
`(ν_iasi, R_iasi, BT_iasi)` — IASI channel wavenumbers, spectral radiance
[mW/m²/sr/cm⁻¹], and brightness temperature [K].
"""
function iasi_forward_model(prof::AtmosphericProfile,
                             linelists::Dict{GasSpecies, HITRANLinelist};
                             iasi::IASIInstrument     = IASIInstrument(),
                             geom::ViewingGeometry    = nadir_geometry(),
                             T_sfc::Union{Float64, Nothing} = nothing,
                             ε_sfc::Float64           = 1.0,
                             high_res_factor::Int     = 4,
                             cutoff::Float64          = 25.0,
                             backend                  = CPU())

    # ── 1. High-resolution internal grid ────────────────────────────────
    Δν_hi = iasi.Δν / high_res_factor
    ν_grid_hi = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)

    # ── 2. Compute layer properties ──────────────────────────────────────
    layers = layer_properties(prof)
    n_layers = length(layers.p_mid)
    n_ν_hi   = ν_grid_hi.n

    # ── 3. Build optical depth cube ──────────────────────────────────────
    τ_layers = zeros(Float64, n_ν_hi, n_layers)
    for k in 1:n_layers
        vmr_k = Dict{GasSpecies, Float64}(
            sp => layers.vmr_mid[sp][k] for sp in keys(layers.vmr_mid)
        )
        vmr_h2o = get(vmr_k, H2O, 0.0)
        vmr_co2 = get(vmr_k, CO2, 4.15e-4)

        # Line-by-line contribution per species
        for (sp, ll) in linelists
            vmr = get(vmr_k, sp, 0.0)
            vmr == 0.0 && continue
            σ_sp = compute_voigt_cross_sections(ν_grid_hi, ll,
                                                 layers.T_mid[k],
                                                 layers.p_mid[k] / 1013.25;
                                                 cutoff=cutoff,
                                                 backend=backend)
            Nair_per_vmr = 2.1209e22
            N_col = vmr * layers.Δp[k] * Nair_per_vmr
            τ_layers[:, k] .+= σ_sp .* N_col
        end

        # Add continua
        dz_cm = _dp_to_dz_cm(layers.Δp[k], layers.p_mid[k], layers.T_mid[k])
        τ_layers[:, k] .+= h2o_continuum(ν_grid_hi, vmr_h2o, layers.p_mid[k], layers.T_mid[k]) .* dz_cm
        τ_layers[:, k] .+= co2_continuum(ν_grid_hi, vmr_co2, layers.p_mid[k], layers.T_mid[k]) .* dz_cm
    end

    # ── 4. Radiative transfer ────────────────────────────────────────────
    Tsfc = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    R_hi = schwarzschild_rte(ν_grid_hi, τ_layers, layers.T_mid, Tsfc;
                             μ=geom.μ, ε_sfc=ε_sfc)

    # ── 5. Apply IASI ILS (Norton-Beer apodization) ──────────────────────
    ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss)
    R_apod = apply_ils(ν_grid_hi, R_hi, ils_δν, ils_kern)

    # ── 6. Resample to IASI channels ────────────────────────────────────
    ν_iasi = wavenumber_grid(iasi.ν_min, iasi.ν_max, iasi.Δν)
    R_iasi = _resample_to_iasi(ν_grid_hi, R_apod, ν_iasi)

    # ── 7. Brightness temperatures ───────────────────────────────────────
    BT_iasi = brightness_temperature(ν_iasi, R_iasi)

    return ν_iasi, R_iasi, BT_iasi
end

"""
    _resample_to_iasi(ν_hi, R_hi, ν_iasi) -> Vector{Float64}

Linearly interpolate the high-resolution apodized spectrum onto the
IASI channel wavenumber grid.
"""
function _resample_to_iasi(ν_hi::WavenumberGrid,
                            R_hi::AbstractVector{<:Real},
                            ν_iasi::WavenumberGrid)::Vector{Float64}
    itp = linear_interpolation(ν_hi.ν, R_hi; extrapolation_bc=Flat())
    return itp.(ν_iasi.ν)
end

Base.show(io::IO, iasi::IASIInstrument) =
    print(io, "IASIInstrument($(iasi.ν_min)–$(iasi.ν_max) cm⁻¹, " *
              "Δν=$(iasi.Δν) cm⁻¹, $(iasi.n_channels) channels, " *
              "sinc⊗Gaussian FWHM=$(iasi.fwhm_gauss) cm⁻¹)")
