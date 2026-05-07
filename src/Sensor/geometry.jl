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
"""
function iasi_scan_angles()
    # 15 step positions each side: steps are at ±1.67°, ±5.00°, ..., ±48.33°
    n_steps = 15
    max_angle = 48.33
    half = range(max_angle / n_steps / 2, max_angle;
                 length=n_steps) |> collect
    return vcat(-reverse(half), half)
end

Base.show(io::IO, g::ViewingGeometry) =
    print(io, "ViewingGeometry(θ=$(round(g.zenith_angle,digits=2))°, μ=$(round(g.μ,digits=4)))")
