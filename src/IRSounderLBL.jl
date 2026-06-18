module IRSounderLBL

# Utils
include("Utils/wavenumber_grid.jl")
include("Utils/interpolation.jl")

# Atmosphere
include("Atmosphere/species.jl")
include("Atmosphere/profiles.jl")
include("Atmosphere/layers.jl")
include("Atmosphere/standard_atm.jl")

# HITRAN
include("HITRAN/linelist.jl")
include("HITRAN/fetch.jl")
include("HITRAN/cache.jl")
include("HITRAN/partition.jl")
include("HITRAN/broadening.jl")

# Cross Sections
include("CrossSections/voigt.jl")
include("CrossSections/continuum.jl")
include("CrossSections/optical_depth.jl")
include("CrossSections/line_mixing.jl")
include("CrossSections/cross_section_jacobian.jl")

# Solver
include("Solver/planck.jl")
include("Solver/transmittance.jl")
include("Solver/schwarzschild.jl")
include("Solver/schwarzschild_jacobian.jl")

# Sensor
include("Sensor/geometry.jl")
include("Sensor/ils.jl")
include("Sensor/ils_fft.jl")
include("Sensor/iasi.jl")
include("Sensor/iasi_l1c.jl")

# Parallel
include("Parallel/strategy.jl")

# Estimation (Jacobians / retrieval)
include("Estimation/state_vector.jl")
include("Estimation/jacobian_fd.jl")
include("Estimation/vmr_jacobian.jl")
include("Estimation/sa_builder.jl")
include("Estimation/covariance.jl")
include("Estimation/optimal_estimation.jl")

# Utils exports
export WavenumberGrid, wavenumber_grid

# Atmosphere exports
export GasSpecies, HITRAN_MOLECULE_ID, SPECIES_NAME
export H2O, CO2, O3, N2O, CO, CH4, O2, SO2, NH3
export AtmosphericProfile, n_levels, species
export pressure_layers, layer_properties
export us_standard_atmosphere, tropical_atmosphere, subarctic_atmosphere,
       afgl_us_standard_50lev

# HITRAN exports
export HITRANLine, HITRANLinelist, filter_linelist
export load_hitran_par, fetch_hitran_api
export load_linelist, linelist_cache_dir, clear_linelist_cache, set_linelist_cache_dir!
export T_REF, partition_function, Q_ratio
export pressure_broadened_width, temperature_scaled_intensity, pressure_shift

# Cross section exports
export VoigtMethod, Weideman, PseudoVoigt, FullFaddeeva
export weideman_voigt, faddeeva_voigt, pseudo_voigt_profile
export voigt_profile, compute_voigt_cross_sections, compute_voigt_cross_sections_dT
export compute_voigt_cross_sections_grad
export h2o_continuum, co2_continuum, co2_cia, n2_cia, o2_cia
export layer_optical_depth
export RelmatLine, RelmatBand, HITRANRelmatData
export load_hitran_relmat, compute_voigt_lm_cross_sections,
       compute_lm_dispersive_correction
export AbstractLineMixing, VPYLineMixing, VPWLineMixing
export BandModes, band_modes, compute_vpw_band_xsec,
       compute_vpw_band_perturbation, compute_voigt_vpw_cross_sections,
       default_vpw_whitelist

# Solver exports
export planck_radiance, brightness_temperature
export level_transmittances
export schwarzschild_rte, schwarzschild_rte_jacobian

# Sensor exports
export ViewingGeometry, airmass_factor
export nadir_geometry, iasi_scan_angles
export scan_angle_to_local_zenith, iasi_zenith_angles
export IASIInstrument, iasi_grid
export ils_kernel, apply_ils, apply_ils_fft, ILSConvolver, ils_apply!
export NORTON_BEER_COEFFS, norton_beer_apodization
export iasi_forward_model
export read_iasi_l1c, IASIL1CGranule, nfov, measurement, cloud_fraction, solar_reflection_angle

# Parallel exports
export ComputeBackend, detect_backend

# Estimation exports
export StateVectorSpec, pack_state, unpack_state, state_labels
export Jacobian, finite_difference_jacobian, default_fd_steps, column
export analytic_jacobian
export optimal_estimation, RetrievalResult
export apodized_measurement_covariance
export build_sa
export dB_dT

end # module IRSounderLBL
