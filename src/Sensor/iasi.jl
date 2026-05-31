"""
IASI instrument model.

IASI (Infrared Atmospheric Sounding Interferometer) on MetOp-A/B/C:
- Spectral range:  645–2760 cm⁻¹
- Spectral sampling: 0.25 cm⁻¹ (after apodization)
- Number of channels: 8461
- Native OPD: 2.0 cm (unapodized Δν = 0.25 cm⁻¹)
- Apodization: Gaussian (IASI L1C convention; Norton-Beer available as option)
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
                       apply_continuum=true,
                       with_ils=true,
                       apodization=:gaussian,
                       line_mixing=nothing,
                       T_method=:logp_at_pcg,
                       source_function=:cim,
                       backend=CPU()) -> (ν_iasi, R_iasi, BT_iasi)

Full IASI forward model: atmosphere → level optical depths → RTE → ILS → IASI L1C.

# Arguments
- `prof`:             `AtmosphericProfile` for the column
- `linelists`:        `Dict{GasSpecies, HITRANLinelist}` for all species
- `iasi`:             `IASIInstrument` specification
- `geom`:             `ViewingGeometry`
- `T_sfc`:            surface skin temperature (K); defaults to profile[1]
- `ε_sfc`:            surface emissivity
- `high_res_factor`:  internal spectral over-sampling relative to IASI Δν
- `cutoff`:           Voigt wing cutoff (cm⁻¹)
- `apply_continuum`:  master switch for continua (default true)
- `continua`:         which continua to include when `apply_continuum`; any of
                      `:h2o`, `:co2` (MT-CKD CO₂), `:co2_cia`, `:n2`, `:o2`
                      (default all). E.g. `(:co2,)` isolates the MT-CKD CO₂ continuum.
- `with_ils`:         convolve with IASI ILS before resampling (default true)
- `apodization`:      ILS apodization style: `:gaussian` (default, matches IASI L1C),
                      `:norton_beer_weak`/`_medium`/`_strong` for alternative tapers
- `line_mixing`:      CO2 line-mixing model, e.g. `VPYLineMixing(relmat)`; `nothing` (default) disables LM
- `T_method`:         per-layer effective-T rule for the LBL cross-section:
                      `:logp_at_pcg` (default, per-species CG temperature) or
                      `:mass_weighted` (single mass-weighted layer T, = LBLRTM TBAR).
                      See `layer_properties`.
- `source_function`:  layer Planck source: `:cim` (default, Clough–Iacono–Moncet
                      Padé anchored at the CG layer-mean T — ~3× more accurate at
                      native 5-km layering) or `:toon` (legacy linear-in-τ at level
                      boundary T's). See `schwarzschild_rte`.
- `backend`:          compute backend

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
                             apply_continuum::Bool    = true,
                             continua                 = (:h2o, :co2, :co2_cia, :n2, :o2),
                             with_ils::Bool           = true,
                             apodization::Symbol      = :gaussian,
                             line_mixing::Union{Nothing, AbstractLineMixing} = nothing,
                             T_method::Symbol         = :logp_at_pcg,
                             source_function::Symbol  = :cim,
                             backend                  = CPU())

    # ── 1. High-resolution internal grid ────────────────────────────────
    Δν_hi = iasi.Δν / high_res_factor
    ν_grid_hi = wavenumber_grid(iasi.ν_min, iasi.ν_max, Δν_hi)

    # ── 2. Compute layer properties ──────────────────────────────────────
    layers = layer_properties(prof; T_method=T_method)
    n_layers = length(layers.p_mid)
    n_ν_hi   = ν_grid_hi.n

    # ── 3. Build optical depth cube ──────────────────────────────────────
    τ_layers  = zeros(Float64, n_ν_hi, n_layers)
    Nair_per_vmr = 2.1209e22   # molec/(cm²·hPa)

    # LBL: Curtis-Godson effective VMR, pressure, and temperature per layer
    for k in 1:n_layers
        for (sp, ll) in linelists
            vmr = haskey(layers.vmr_cg, sp) ? layers.vmr_cg[sp][k] : 0.0
            vmr == 0.0 && continue
            p_cg_atm = layers.p_cg[sp][k] / 1013.25
            T_cg_sp  = layers.T_cg[sp][k]
            vmr_self = (sp == H2O) ? vmr : 0.0
            σ_sp = _species_cross_section(line_mixing, sp, ν_grid_hi, ll,
                                           T_cg_sp, p_cg_atm;
                                           vmr_self=vmr_self,
                                           cutoff=cutoff, backend=backend)
            τ_layers[:, k] .+= σ_sp .* (vmr * layers.Δp[k] * Nair_per_vmr)
        end
    end

    # Continua: mid-layer conditions (smooth — trapezoidal correction negligible)
    if apply_continuum
        for k in 1:n_layers
            vmr_h2o = haskey(layers.vmr_mid, H2O) ? layers.vmr_mid[H2O][k] : 0.0
            vmr_co2 = haskey(layers.vmr_mid, CO2) ? layers.vmr_mid[CO2][k] : 4.15e-4
            dz_cm   = _dp_to_dz_cm(layers.Δp[k], layers.p_mid[k], layers.T_mid[k])
            vmr_dry = 1.0 - vmr_h2o
            (:h2o     in continua) && (τ_layers[:, k] .+= h2o_continuum(ν_grid_hi, vmr_h2o, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:co2     in continua) && (τ_layers[:, k] .+= co2_continuum(ν_grid_hi, vmr_co2, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:co2_cia in continua) && (τ_layers[:, k] .+= co2_cia(ν_grid_hi, vmr_co2, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:n2      in continua) && (τ_layers[:, k] .+= n2_cia(ν_grid_hi, 0.78084 * vmr_dry, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
            (:o2      in continua) && (τ_layers[:, k] .+= o2_cia(ν_grid_hi, 0.20946 * vmr_dry, layers.p_mid[k], layers.T_mid[k]) .* dz_cm)
        end
    end

    # ── 4. Radiative transfer ────────────────────────────────────────────
    # The default :cim source function needs the per-layer mass-weighted T
    # (T_AVE, the same TBAR LBLRTM prints); it is ~3× more accurate at native
    # 5-km layering than :toon (see scripts/convergence_sweep_43um.jl). For the
    # legacy :toon LIT, T_ave is unused; build it only for :cim.
    T_ave_for_src = if source_function == :cim
        p_lev = prof.pressure
        T_lev = prof.temperature
        [cg_temperature_mass(T_lev[k], T_lev[k+1], p_lev[k], p_lev[k+1])
         for k in 1:n_layers]
    else
        nothing
    end
    Tsfc = isnothing(T_sfc) ? prof.temperature[1] : T_sfc
    R_hi = schwarzschild_rte(ν_grid_hi, τ_layers, prof.temperature, Tsfc;
                             μ=geom.μ, ε_sfc=ε_sfc,
                             source_function=source_function,
                             T_ave=T_ave_for_src)

    # ── 5. Apply IASI ILS (Norton-Beer apodization) ──────────────────────
    if with_ils
        ils_δν, ils_kern = ils_kernel(Δν_hi, iasi.opd_max, iasi.fwhm_gauss;
                                      apodization=apodization)
        R_apod = apply_ils(ν_grid_hi, R_hi, ils_δν, ils_kern)
    else
        R_apod = R_hi
    end

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
