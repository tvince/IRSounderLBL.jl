"""
Generic hyperspectral IR sounder model.

A `Sounder` describes a nadir-viewing Fourier-transform infrared sounder by its
spectral grid and instrument line shape (ILS = sinc of the maximum optical path
difference, convolved with a Gaussian apodization). The same forward model serves
IASI, IASI-NG, CrIS, and MTG-IRS; each is just a `Sounder` with different
`ν_min`/`ν_max`/`Δν`/`opd_max`/`fwhm_gauss`. Named constructors ([`IASIInstrument`],
[`IASINGInstrument`], [`CrISInstrument`], [`MTGIRSInstrument`]) fill in the
published specification for each mission.
"""

"""
    Sounder

Hyperspectral IR sounder specification (a single contiguous spectral band).

# Fields
- `ν_min`:       minimum wavenumber (cm⁻¹)
- `ν_max`:       maximum wavenumber (cm⁻¹)
- `Δν`:          spectral sampling (cm⁻¹)
- `n_channels`:  number of spectral channels on `[ν_min, ν_max]` at `Δν`
- `opd_max`:     maximum optical path difference (cm); sets the sinc ILS width
                 (`Δν = 1/(2·opd_max)` at full spectral resolution)
- `fwhm_gauss`:  FWHM of the Gaussian apodization convolved with the sinc (cm⁻¹);
                 `0` ⇒ pure (unapodized) sinc ILS

Multi-band sounders (CrIS, MTG-IRS) have gaps between detector bands; a single
`Sounder` spans the full range and simulates the gap region too — drop those
channels with [`exclude_channels`](@ref)/`channel_mask`, or construct a single
band with the `band` keyword of the named constructor.
"""
struct Sounder
    ν_min::Float64
    ν_max::Float64
    Δν::Float64
    n_channels::Int
    opd_max::Float64
    fwhm_gauss::Float64
end

"""
    Sounder(; ν_min, ν_max, Δν, opd_max=1/(2Δν), fwhm_gauss=0.0, n_channels=derived)

Generic keyword constructor. `n_channels` defaults to `round((ν_max−ν_min)/Δν)+1`
and `opd_max` to the full-spectral-resolution value `1/(2·Δν)`. Use this for a
sounder without a named preset; otherwise prefer the mission constructors.
"""
function Sounder(; ν_min::Real, ν_max::Real, Δν::Real,
                 opd_max::Real = 1.0 / (2.0 * Δν),
                 fwhm_gauss::Real = 0.0,
                 n_channels::Union{Integer, Nothing} = nothing)
    n_ch = n_channels === nothing ? round(Int, (ν_max - ν_min) / Δν) + 1 : Int(n_channels)
    return Sounder(Float64(ν_min), Float64(ν_max), Float64(Δν), n_ch,
                   Float64(opd_max), Float64(fwhm_gauss))
end

"""
    IASIInstrument(; fwhm_gauss=0.5) -> Sounder

IASI (Infrared Atmospheric Sounding Interferometer) on MetOp-A/B/C, L1C spec:
645–2760 cm⁻¹, Δν = 0.25 cm⁻¹, 8461 channels, OPD 2.0 cm, Gaussian apodization
FWHM 0.5 cm⁻¹ (the L1C self-apodization).
"""
function IASIInstrument(; fwhm_gauss::Real = 0.5)
    return Sounder(; ν_min = 645.0, ν_max = 2760.0, Δν = 0.25,
                   opd_max = 2.0, fwhm_gauss = fwhm_gauss)   # n_channels → 8461
end

# Back-compat: the old positional `IASIInstrument(ν_min, ν_max, Δν, n_channels,
# opd_max, fwhm_gauss)` constructor, now building a generic `Sounder`. Lets the
# existing scripts/tests that construct sub-window instruments keep working
# unchanged; new code should use `Sounder(; …)` or a mission constructor.
IASIInstrument(ν_min::Real, ν_max::Real, Δν::Real, n_channels::Integer,
               opd_max::Real, fwhm_gauss::Real) =
    Sounder(Float64(ν_min), Float64(ν_max), Float64(Δν), Int(n_channels),
            Float64(opd_max), Float64(fwhm_gauss))

"""
    IASINGInstrument(; fwhm_gauss=0.25) -> Sounder

IASI-NG on MetOp-SG-A: same 645–2760 cm⁻¹ range at twice IASI's resolution —
Δν = 0.125 cm⁻¹ (16921 channels), OPD 4.0 cm. Apodization FWHM defaults to
0.25 cm⁻¹ (half IASI's, matching the finer sampling).
"""
function IASINGInstrument(; fwhm_gauss::Real = 0.25)
    return Sounder(; ν_min = 645.0, ν_max = 2760.0, Δν = 0.125,
                   opd_max = 4.0, fwhm_gauss = fwhm_gauss)
end

"""
    CrISInstrument(; band=:full, fwhm_gauss=0.0) -> Sounder

CrIS (Cross-track Infrared Sounder) on Suomi-NPP / JPSS, Full Spectral Resolution:
Δν = 0.625 cm⁻¹, OPD 0.8 cm, three detector bands
- `:lwir` 650–1095 cm⁻¹, `:mwir` 1210–1750 cm⁻¹, `:swir` 2155–2550 cm⁻¹.
`:full` (default) spans 650–2550 cm⁻¹ as one grid — the two inter-band gaps
(1095–1210, 1750–2155 cm⁻¹) are simulated and should be dropped with
[`exclude_channels`](@ref). CrIS SDR is unapodized (pure sinc), so `fwhm_gauss`
defaults to 0; pass the mission Hamming spec if you apodize.
"""
function CrISInstrument(; band::Symbol = :full, fwhm_gauss::Real = 0.0)
    lo, hi = band === :full  ? (650.0, 2550.0) :
             band === :lwir  ? (650.0, 1095.0) :
             band === :mwir  ? (1210.0, 1750.0) :
             band === :swir  ? (2155.0, 2550.0) :
             error("CrIS band must be :full, :lwir, :mwir, or :swir; got :$band")
    return Sounder(; ν_min = lo, ν_max = hi, Δν = 0.625, opd_max = 0.8,
                   fwhm_gauss = fwhm_gauss)
end

"""
    MTGIRSInstrument(; band=:full, fwhm_gauss=0.0) -> Sounder

MTG-IRS (Meteosat Third Generation InfraRed Sounder) on MTG-S, two bands at
Δν = 0.625 cm⁻¹, OPD 0.8 cm:
- `:lwir` 700–1210 cm⁻¹, `:mwir` 1600–2175 cm⁻¹.
`:full` (default) spans 700–2175 cm⁻¹ as one grid; drop the 1210–1600 cm⁻¹ gap
with [`exclude_channels`](@ref).
"""
function MTGIRSInstrument(; band::Symbol = :full, fwhm_gauss::Real = 0.0)
    lo, hi = band === :full ? (700.0, 2175.0) :
             band === :lwir ? (700.0, 1210.0) :
             band === :mwir ? (1600.0, 2175.0) :
             error("MTG-IRS band must be :full, :lwir, or :mwir; got :$band")
    return Sounder(; ν_min = lo, ν_max = hi, Δν = 0.625, opd_max = 0.8,
                   fwhm_gauss = fwhm_gauss)
end

"""
    sounder_grid(s::Sounder) -> WavenumberGrid

Return the sounder's channel grid as a `WavenumberGrid`.
"""
function sounder_grid(s::Sounder)
    return wavenumber_grid(s.ν_min, s.ν_max, s.Δν)
end

# Back-compat alias (deprecated name; `sounder_grid` is the generic spelling).
const iasi_grid = sounder_grid

"""
    _internal_grid(s, Δν_hi, with_ils) -> WavenumberGrid

High-resolution internal monochromatic grid at spacing `Δν_hi`. When `with_ils`,
the grid is padded by the ILS kernel half-width (`ILS_HALFWIDTH_CM`) on each side:
`apply_ils`/`ILSConvolver` zero-pad beyond the grid, so without this margin every
channel within the kernel half-width of `ν_min`/`ν_max` would roll off (the radiance
collapses at the window edges). After convolution the spectrum is resampled back to
`[ν_min, ν_max]`, discarding the rolled-off skirt. The pad is an integer number of
`Δν_hi` steps, so the un-padded grid — and hence the channel grid — stays an
aligned subset. Used identically by `forward_model` and `analytic_jacobian` so
their spectra agree to round-off.
"""
function _internal_grid(s::Sounder, Δν_hi::Float64, with_ils::Bool)
    if with_ils
        pad = ceil(Int, ILS_HALFWIDTH_CM / Δν_hi) * Δν_hi
        return wavenumber_grid(s.ν_min - pad, s.ν_max + pad, Δν_hi)
    end
    return wavenumber_grid(s.ν_min, s.ν_max, Δν_hi)
end

"""
    forward_model(prof, linelists;
                  sounder=IASIInstrument(),
                  geom=nadir_geometry(),
                  T_sfc=nothing,
                  ε_sfc=1.0,
                  internal_dnu=0.001,
                  high_res_factor=nothing,
                  cutoff=25.0,
                  apply_continuum=true,
                  with_ils=true,
                  apodization=:gaussian,
                  line_mixing=nothing,
                  T_method=:logp_at_pcg,
                  source_function=:cim,
                  backend=CPU()) -> (ν, R, BT)

Full sounder forward model: atmosphere → level optical depths → RTE → ILS →
channel grid. Works for any [`Sounder`](@ref) (IASI, IASI-NG, CrIS, MTG-IRS).

# Arguments
- `prof`:             `AtmosphericProfile` for the column
- `linelists`:        `Dict{GasSpecies, HITRANLinelist}` for all species
- `sounder`:          [`Sounder`](@ref) specification (default IASI L1C)
- `geom`:             `ViewingGeometry`
- `T_sfc`:            surface skin temperature (K); defaults to profile[1]
- `ε_sfc`:            surface emissivity
- `internal_dnu`:     target ABSOLUTE internal monochromatic grid spacing (cm⁻¹,
                      default 0.001). Sized to resolve the Doppler-limited line
                      cores; converges the ILS-convolved BT to ≤6 mK RMS / ≤20 mK
                      worst-point vs a 0.0005 reference, across IASI and IASI-NG in
                      both the 15 µm and 4.3 µm bands (4.3 µm is ~0.05 mK; 15 µm,
                      with narrower cores, is the limiting ~5 mK case). See
                      scripts/validation/grid_convergence_iasing.jl. Auto-adapts to any sensor
                      Δν. Set `high_res_factor` to override.
- `high_res_factor`:  legacy override — internal over-sampling as a divisor of
                      the sensor Δν (`Δν_int = sounder.Δν / high_res_factor`). When
                      `nothing` (default), the grid is derived from `internal_dnu`.
                      An explicit value takes precedence over `internal_dnu`.
- `cutoff`:           Voigt wing cutoff (cm⁻¹)
- `dptmn`:            absolute per-layer optical-depth floor for strength-based
                      line rejection (default 1e-6). Per layer, a line is dropped
                      when its peak OD contribution `peak_σ·vmr·Δp·N_air` falls
                      below `dptmn` — LBLRTM's DPTMIN criterion. BT-lossless at
                      1e-6 (≤0.6 mK over the IASI ILS, ~5× fewer lines; see
                      scripts/validation/validate_line_rejection_bt.jl). Set 0.0 to disable.
                      Skipped on the line-mixing CO₂ path (its band coefficients
                      assume the full line set).
- `apply_continuum`:  master switch for continua (default true)
- `continua`:         which continua to include when `apply_continuum`; any of
                      `:h2o`, `:co2` (MT-CKD CO₂), `:co2_cia`, `:n2`, `:o2`
                      (default all). E.g. `(:co2,)` isolates the MT-CKD CO₂ continuum.
- `with_ils`:         convolve with the ILS before resampling (default true)
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
`(ν, R, BT)` — channel wavenumbers, spectral radiance [mW/m²/sr/cm⁻¹], and
brightness temperature [K].
"""
function forward_model(prof::AtmosphericProfile,
                       linelists::Dict{GasSpecies, HITRANLinelist};
                       sounder::Sounder        = IASIInstrument(),
                       iasi::Union{Sounder, Nothing} = nothing,  # deprecated alias
                       geom::ViewingGeometry   = nadir_geometry(),
                       T_sfc::Union{Float64, Nothing} = nothing,
                       ε_sfc::Float64          = 1.0,
                       internal_dnu::Union{Float64, Nothing} = 0.001,
                       high_res_factor::Union{Int, Nothing}   = nothing,
                       cutoff::Float64         = 25.0,
                       lm_cutoff::Float64      = cutoff,
                       dptmn::Float64          = 1e-6,
                       apply_continuum::Bool   = true,
                       continua                = (:h2o, :co2, :co2_cia, :n2, :o2),
                       with_ils::Bool          = true,
                       apodization::Symbol     = :gaussian,
                       line_mixing::Union{Nothing, AbstractLineMixing} = nothing,
                       T_method::Symbol        = :logp_at_pcg,
                       source_function::Symbol = :cim,
                       backend                 = CPU())

    inst = iasi === nothing ? sounder : iasi    # back-compat: accept `iasi=` too

    # ── 1. High-resolution internal grid ────────────────────────────────
    # The internal monochromatic grid must resolve the narrowest line cores
    # (Doppler floor ~0.004 cm⁻¹ at 4.3 µm, ~0.001 at 15 µm). `internal_dnu`
    # targets that ABSOLUTE spacing; `high_res_factor`, if given, overrides it
    # (legacy relative divisor of the sensor Δν). See scripts/validation/grid_convergence_iasing.jl:
    # internal 0.001 cm⁻¹ converges the ILS-convolved BT to <1e-4 K, while the old
    # high_res_factor=4 (=0.0625 cm⁻¹ for IASI) was ≈1 K off in dense bands.
    hrf = if high_res_factor !== nothing
        high_res_factor
    elseif internal_dnu !== nothing
        max(1, ceil(Int, inst.Δν / internal_dnu))
    else
        4
    end
    Δν_hi = inst.Δν / hrf
    ν_grid_hi = _internal_grid(inst, Δν_hi, with_ils)

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
            coef     = vmr * layers.Δp[k] * Nair_per_vmr

            # Strength-based line rejection: drop lines whose peak OD contribution
            # (peak_σ·coef) is below `dptmn`. Skipped on the line-mixing CO₂ path,
            # whose band coefficients assume the full line set (see _reject_weak_lines).
            lm_co2 = sp == CO2 && (line_mixing isa VPYLineMixing ||
                                   line_mixing isa VPWLineMixing)
            ll_use = (dptmn > 0.0 && !lm_co2) ?
                _reject_weak_lines(ll, T_cg_sp, p_cg_atm, coef;
                                   vmr_self=vmr_self, dptmn=dptmn) : ll

            σ_sp = _species_cross_section(line_mixing, sp, ν_grid_hi, ll_use,
                                           T_cg_sp, p_cg_atm;
                                           vmr_self=vmr_self,
                                           cutoff=cutoff, lm_cutoff=lm_cutoff, backend=backend)
            τ_layers[:, k] .+= σ_sp .* coef
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
    # 5-km layering than :toon (see scripts/validation/convergence_sweep_43um.jl). For the
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

    # ── 5. Apply the ILS ─────────────────────────────────────────────────
    if with_ils
        ils_δν, ils_kern = ils_kernel(Δν_hi, inst.opd_max, inst.fwhm_gauss;
                                      apodization=apodization)
        R_apod = apply_ils(ν_grid_hi, R_hi, ils_δν, ils_kern)
    else
        R_apod = R_hi
    end

    # ── 6. Resample to the sounder channels ──────────────────────────────
    ν_out = wavenumber_grid(inst.ν_min, inst.ν_max, inst.Δν)
    R_out = _resample_to_channels(ν_grid_hi, R_apod, ν_out)

    # ── 7. Brightness temperatures ───────────────────────────────────────
    BT_out = brightness_temperature(ν_out, R_out)

    return ν_out, R_out, BT_out
end

# Back-compat alias (deprecated name; `forward_model` is the generic spelling).
# Same function object, so `iasi_forward_model(...; iasi=inst)` keeps working.
const iasi_forward_model = forward_model

"""
    _resample_to_channels(ν_hi, R_hi, ν_out) -> Vector{Float64}

Linearly interpolate the high-resolution apodized spectrum onto the sounder
channel wavenumber grid.
"""
function _resample_to_channels(ν_hi::WavenumberGrid,
                               R_hi::AbstractVector{<:Real},
                               ν_out::WavenumberGrid)::Vector{Float64}
    itp = linear_interpolation(ν_hi.ν, R_hi; extrapolation_bc=Flat())
    return itp.(ν_out.ν)
end

# Back-compat alias (deprecated name).
const _resample_to_iasi = _resample_to_channels

Base.show(io::IO, s::Sounder) =
    print(io, "Sounder($(s.ν_min)–$(s.ν_max) cm⁻¹, " *
              "Δν=$(s.Δν) cm⁻¹, $(s.n_channels) channels, " *
              "sinc(OPD=$(s.opd_max))⊗Gaussian(FWHM=$(s.fwhm_gauss)) cm⁻¹)")
