"""
    WavenumberGrid

Uniform wavenumber grid for spectral calculations.

# Fields
- `ν`: wavenumber array (cm⁻¹)
- `Δν`: grid spacing (cm⁻¹)
- `n`: number of grid points
"""
struct WavenumberGrid
    ν::Vector{Float64}   # wavenumber array (cm⁻¹)
    Δν::Float64          # grid spacing (cm⁻¹)
    n::Int               # number of grid points
end

"""
    wavenumber_grid(ν_min, ν_max, Δν)

Construct a uniform `WavenumberGrid` from `ν_min` to `ν_max` with spacing `Δν`.
All values in cm⁻¹.
"""
function wavenumber_grid(ν_min::Float64, ν_max::Float64, Δν::Float64)
    ν = collect(ν_min:Δν:ν_max)
    return WavenumberGrid(ν, Δν, length(ν))
end

"""
    wavenumber_grid(ν_min, ν_max; n)

Construct a uniform `WavenumberGrid` with exactly `n` points.
"""
function wavenumber_grid(ν_min::Float64, ν_max::Float64; n::Int)
    Δν = (ν_max - ν_min) / (n - 1)
    ν = LinRange(ν_min, ν_max, n) |> collect
    return WavenumberGrid(ν, Δν, n)
end

Base.length(g::WavenumberGrid) = g.n
Base.show(io::IO, g::WavenumberGrid) =
    print(io, "WavenumberGrid($(g.ν[1])–$(g.ν[end]) cm⁻¹, Δν=$(g.Δν) cm⁻¹, n=$(g.n))")
