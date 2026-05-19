module RadiativeTransfer

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
include("HITRAN/partition.jl")
include("HITRAN/broadening.jl")

# Cross Sections
include("CrossSections/voigt.jl")
include("CrossSections/continuum.jl")
include("CrossSections/optical_depth.jl")
include("CrossSections/line_mixing.jl")

# Solver
include("Solver/planck.jl")
include("Solver/transmittance.jl")
include("Solver/schwarzschild.jl")

# Sensor
include("Sensor/geometry.jl")
include("Sensor/ils.jl")
include("Sensor/iasi.jl")

# Parallel
include("Parallel/strategy.jl")

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
export T_REF, partition_function, Q_ratio
export pressure_broadened_width, temperature_scaled_intensity, pressure_shift

# Cross section exports
export VoigtMethod, Weideman, PseudoVoigt, FullFaddeeva
export weideman_voigt, faddeeva_voigt, pseudo_voigt_profile
export voigt_profile, compute_voigt_cross_sections
export h2o_continuum, co2_continuum
export layer_optical_depth
export RelmatLine, RelmatBand, HITRANRelmatData
export load_hitran_relmat, compute_voigt_lm_cross_sections,
       compute_lm_dispersive_correction
export AbstractLineMixing, VPYLineMixing

# Solver exports
export planck_radiance, brightness_temperature
export level_transmittances
export schwarzschild_rte

# Sensor exports
export ViewingGeometry, airmass_factor
export nadir_geometry, iasi_scan_angles
export IASIInstrument, iasi_grid
export ils_kernel, apply_ils
export NORTON_BEER_COEFFS, norton_beer_apodization
export iasi_forward_model

# Parallel exports
export ComputeBackend, detect_backend

end # module RadiativeTransfer
