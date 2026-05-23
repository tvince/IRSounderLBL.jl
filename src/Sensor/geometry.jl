"""
Satellite viewing geometry for a nadir-scanning radiometer.

IASI scans ±48.33° across-track from nadir in 15 steps per half-scan,
giving 30 pixels across the swath (each 50 km × 50 km at nadir).
"""

"""
    ViewingGeometry

Encapsulates satellite viewing geometry.

# Fields
- `zenith_angle`:   viewing zenith angle θ (degrees)  in [0, 90)
- `azimuth_angle`:  viewing azimuth angle φ (degrees) in [0, 360)
- `μ`:              cos(θ), the airmass factor for plane-parallel RT
"""
struct ViewingGeometry
    zenith_angle::Float64   # degrees
    azimuth_angle::Float64  # degrees
    μ::Float64              # cos(zenith_angle)
end

"""
    ViewingGeometry(zenith_angle; azimuth_angle=0.0)

Construct a `ViewingGeometry` from the viewing zenith angle (degrees).
"""
function ViewingGeometry(zenith_angle::Float64; azimuth_angle::Float64=0.0)
    0.0 <= zenith_angle < 90.0 ||
        error("Viewing zenith angle must be in [0, 90); got $(zenith_angle)°")
    μ = cosd(zenith_angle)
    return ViewingGeometry(zenith_angle, azimuth_angle, μ)
end

"""
    nadir_geometry() -> ViewingGeometry

Return a purely nadir (θ=0°) viewing geometry.
"""
nadir_geometry() = ViewingGeometry(0.0)

"""
    airmass_factor(geom) -> Float64

Return the plane-parallel airmass factor 1/cos(θ) = 1/μ.
"""
airmass_factor(geom::ViewingGeometry) = 1.0 / geom.μ

"""
    iasi_scan_angles() -> Vector{Float64}

Return the 30 nominal IASI across-track scan angles (degrees) for a full
scan line.  IASI measures 4 pixels per step, 15 steps per half-scan,
covering ±48.33°.

These are angles **at the satellite**, off-nadir.  For the local zenith
angle at the surface (the quantity that goes into plane-parallel RT),
pass them through `scan_angle_to_local_zenith`.
"""
function iasi_scan_angles()
    # 15 step positions each side: steps are at ±1.67°, ±5.00°, ..., ±48.33°
    n_steps = 15
    max_angle = 48.33
    half = range(max_angle / n_steps / 2, max_angle;
                 length=n_steps) |> collect
    return vcat(-reverse(half), half)
end

# Mean Earth radius (km) — IUGG value.
const _R_EARTH_KM = 6371.0
# MetOp-A/B/C nominal altitude (km) — sun-synchronous polar orbit.
const _METOP_ALT_KM = 817.0

"""
    scan_angle_to_local_zenith(scan_angle; h_sat_km=817.0, R_earth_km=6371.0) -> Float64

Convert a satellite off-nadir scan angle (degrees) into the local viewing
zenith angle at the surface (degrees), accounting for Earth curvature.

For a satellite at altitude `h_sat_km` above a spherical Earth of radius
`R_earth_km`, the law of sines on the triangle (Earth center, satellite,
surface intersection) gives:

    sin(θ_local) = (R_earth + h_sat) / R_earth × sin(scan_angle)

For IASI on MetOp at scan_angle = 48.33°, this yields θ_local ≈ 57.5°.
The plane-parallel airmass 1/cos(θ_local) is what should drive RT — using
the raw scan angle would under-estimate the slant path by ~30 % at the
swath edge.
"""
function scan_angle_to_local_zenith(scan_angle::Real;
                                    h_sat_km::Real    = _METOP_ALT_KM,
                                    R_earth_km::Real  = _R_EARTH_KM)::Float64
    s = (R_earth_km + h_sat_km) / R_earth_km * sind(abs(scan_angle))
    # |scan_angle| must be < arcsin(R_earth / (R_earth + h)) — the limb angle
    # at the satellite, beyond which the LOS misses the planet entirely.
    s <= 1.0 || error("scan_angle $(scan_angle)° exceeds limb angle from h=$(h_sat_km) km")
    return asind(s)
end

"""
    iasi_zenith_angles(; h_sat_km=817.0) -> Vector{Float64}

Return the 30 local viewing zenith angles (degrees) corresponding to the
nominal IASI scan positions, ready to pass into `ViewingGeometry` for
plane-parallel RT.  Symmetric across nadir; sign of `iasi_scan_angles()`
is preserved so cross-track sense is retained.
"""
function iasi_zenith_angles(; h_sat_km::Real = _METOP_ALT_KM)
    return [sign(s) * scan_angle_to_local_zenith(s; h_sat_km=h_sat_km)
            for s in iasi_scan_angles()]
end

Base.show(io::IO, g::ViewingGeometry) =
    print(io, "ViewingGeometry(θ=$(round(g.zenith_angle,digits=2))°, μ=$(round(g.μ,digits=4)))")
