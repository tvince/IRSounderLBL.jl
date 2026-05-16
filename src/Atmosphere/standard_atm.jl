"""
Standard atmosphere profiles: US Standard (1976), Tropical, and Subarctic Summer.

The 43-level profiles use pressure levels from the original AFGL tabulation.
The 50-level profile (afgl_us_standard_50lev) extends to 120 km by reading
data/afgl_us_standard_50lev.csv, which was generated from the ARTS XML data
repository (Anderson et al. 1986, AFGL-TR-86-0110).
"""

# AFGL 43-level standard pressure grid (hPa), surface to TOA
const AFGL_PRESSURE = Float64[
    1013.25, 904.78, 805.12, 714.87, 633.59, 560.74, 495.74, 437.82,
     386.00, 339.68, 298.50, 261.99, 229.72, 201.16, 175.90, 153.56,
     133.74, 116.09, 100.59,  86.99,  75.07,  64.75,  55.77,  47.97,
      41.22,  35.39,  30.38,  26.10,  22.47,  19.40,  16.75,  14.51,
      12.57,  10.88,   9.33,   7.71,   6.13,   4.65,   3.33,   2.10,
       1.30,   0.82,   0.52
]

# ── US Standard Atmosphere 1976 ──────────────────────────────────────────────

const US_STD_TEMPERATURE = Float64[
    288.15, 281.65, 275.15, 268.66, 262.17, 255.68, 249.19, 242.70,
    236.21, 229.73, 223.25, 216.78, 216.65, 216.65, 216.65, 216.65,
    216.65, 216.65, 216.65, 216.65, 216.65, 217.58, 218.57, 219.57,
    220.56, 221.55, 222.54, 223.53, 224.52, 225.52, 226.51, 227.50,
    228.49, 233.74, 239.28, 245.18, 253.57, 260.00, 260.00, 258.10,
    249.40, 240.00, 231.00
]

# VMR of major gases for US Standard (mol/mol)
const US_STD_H2O = Float64[
    7.745e-3, 6.071e-3, 4.631e-3, 3.182e-3, 2.158e-3, 1.397e-3, 9.254e-4,
    5.720e-4, 3.667e-4, 1.583e-4, 6.996e-5, 3.613e-5, 1.906e-5, 1.085e-5,
    5.927e-6, 5.000e-6, 3.950e-6, 3.850e-6, 3.825e-6, 3.850e-6, 3.900e-6,
    3.975e-6, 4.065e-6, 4.200e-6, 4.300e-6, 4.425e-6, 4.575e-6, 4.725e-6,
    4.825e-6, 4.900e-6, 4.950e-6, 5.025e-6, 5.150e-6, 5.225e-6, 5.250e-6,
    5.225e-6, 5.100e-6, 4.750e-6, 4.200e-6, 3.500e-6, 2.825e-6, 2.050e-6,
    1.330e-6
]

const US_STD_CO2 = fill(4.15e-4, length(AFGL_PRESSURE))   # 415 ppm (2023 value)

const US_STD_O3 = Float64[
    3.017e-8, 3.535e-8, 3.987e-8, 4.495e-8, 5.208e-8, 6.071e-8, 7.071e-8,
    8.591e-8, 1.059e-7, 1.507e-7, 2.103e-7, 2.999e-7, 4.450e-7, 6.400e-7,
    8.800e-7, 1.188e-6, 1.517e-6, 1.804e-6, 2.020e-6, 2.178e-6, 2.221e-6,
    2.178e-6, 2.058e-6, 1.849e-6, 1.566e-6, 1.255e-6, 9.770e-7, 7.340e-7,
    5.160e-7, 3.480e-7, 2.450e-7, 1.716e-7, 1.153e-7, 7.320e-8, 4.440e-8,
    2.450e-8, 1.241e-8, 5.250e-9, 1.978e-9, 6.750e-10, 2.225e-10, 8.500e-11,
    3.500e-11
]

const US_STD_CH4 = Float64[
    1.700e-6, 1.700e-6, 1.700e-6, 1.700e-6, 1.700e-6, 1.700e-6, 1.700e-6,
    1.699e-6, 1.697e-6, 1.693e-6, 1.685e-6, 1.675e-6, 1.665e-6, 1.653e-6,
    1.640e-6, 1.626e-6, 1.610e-6, 1.590e-6, 1.567e-6, 1.540e-6, 1.510e-6,
    1.478e-6, 1.444e-6, 1.408e-6, 1.370e-6, 1.330e-6, 1.288e-6, 1.244e-6,
    1.198e-6, 1.150e-6, 1.100e-6, 1.048e-6, 9.938e-7, 9.375e-7, 8.788e-7,
    8.175e-7, 7.538e-7, 6.875e-7, 6.188e-7, 5.475e-7, 4.738e-7, 3.975e-7,
    3.188e-7
]

const US_STD_N2O = Float64[
    3.200e-7, 3.200e-7, 3.200e-7, 3.200e-7, 3.200e-7, 3.200e-7, 3.200e-7,
    3.200e-7, 3.200e-7, 3.195e-7, 3.183e-7, 3.164e-7, 3.140e-7, 3.109e-7,
    3.072e-7, 3.027e-7, 2.977e-7, 2.919e-7, 2.854e-7, 2.782e-7, 2.703e-7,
    2.617e-7, 2.524e-7, 2.424e-7, 2.318e-7, 2.206e-7, 2.088e-7, 1.965e-7,
    1.837e-7, 1.705e-7, 1.571e-7, 1.438e-7, 1.307e-7, 1.178e-7, 1.054e-7,
    9.338e-8, 8.222e-8, 7.109e-8, 6.052e-8, 5.101e-8, 4.254e-8, 3.487e-8,
    2.773e-8
]

const US_STD_CO = Float64[
    1.500e-7, 1.450e-7, 1.399e-7, 1.349e-7, 1.312e-7, 1.303e-7, 1.288e-7,
    1.247e-7, 1.185e-7, 1.094e-7, 9.962e-8, 8.964e-8, 7.814e-8, 6.374e-8,
    5.025e-8, 3.941e-8, 3.069e-8, 2.489e-8, 1.966e-8, 1.549e-8, 1.331e-8,
    1.232e-8, 1.232e-8, 1.307e-8, 1.400e-8, 1.498e-8, 1.598e-8, 1.710e-8,
    1.850e-8, 2.009e-8, 2.187e-8, 2.400e-8, 2.618e-8, 2.849e-8, 3.122e-8,
    3.444e-8, 3.799e-8, 4.173e-8, 4.584e-8, 5.142e-8, 5.859e-8, 6.674e-8,
    7.624e-8
]

# AFGL 43-level altitude (km), approximate geometric altitudes
const AFGL_ALTITUDE = Float64[
     0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,
     8.00,  9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00,
    16.00, 17.00, 18.00, 19.00, 20.00, 21.00, 22.00, 23.00,
    24.00, 25.00, 26.00, 27.00, 28.00, 29.00, 30.00, 31.00,
    32.00, 33.00, 34.00, 35.00, 37.00, 40.00, 45.00, 50.00,
    55.00, 60.00, 65.00
]

"""
    us_standard_atmosphere()

Return the 1976 US Standard Atmosphere as an `AtmosphericProfile` on the
43 AFGL pressure levels.
"""
function us_standard_atmosphere()
    vmr = Dict{GasSpecies, Vector{Float64}}(
        H2O => copy(US_STD_H2O),
        CO2 => copy(US_STD_CO2),
        O3  => copy(US_STD_O3),
        CH4 => copy(US_STD_CH4),
        N2O => copy(US_STD_N2O),
        CO  => copy(US_STD_CO),
    )
    return AtmosphericProfile(
        copy(AFGL_PRESSURE),
        copy(US_STD_TEMPERATURE),
        copy(AFGL_ALTITUDE),
        vmr
    )
end

# ── Tropical Atmosphere ──────────────────────────────────────────────────────

const TROPICAL_TEMPERATURE = Float64[
    299.70, 293.70, 287.70, 283.70, 277.00, 270.30, 263.60, 254.00,
    242.00, 230.00, 219.00, 208.00, 197.00, 194.80, 194.80, 194.80,
    194.80, 194.80, 194.80, 198.80, 202.70, 206.70, 210.70, 214.60,
    217.00, 219.20, 221.40, 223.60, 225.80, 227.00, 230.00, 233.74,
    239.28, 245.18, 251.20, 257.10, 263.00, 263.00, 263.00, 258.10,
    249.40, 240.00, 231.00
]

# Tropical H2O: significantly wetter lower troposphere
const TROPICAL_H2O = Float64[
    2.593e-2, 1.949e-2, 1.534e-2, 8.600e-3, 4.441e-3, 3.346e-3, 2.101e-3,
    1.289e-3, 7.637e-4, 4.098e-4, 1.912e-4, 7.306e-5, 2.905e-5, 1.401e-5,
    7.487e-6, 5.000e-6, 3.950e-6, 3.850e-6, 3.825e-6, 3.850e-6, 3.900e-6,
    3.975e-6, 4.065e-6, 4.200e-6, 4.300e-6, 4.425e-6, 4.575e-6, 4.725e-6,
    4.825e-6, 4.900e-6, 4.950e-6, 5.025e-6, 5.150e-6, 5.225e-6, 5.250e-6,
    5.225e-6, 5.100e-6, 4.750e-6, 4.200e-6, 3.500e-6, 2.825e-6, 2.050e-6,
    1.330e-6
]

"""
    tropical_atmosphere()

Return a tropical `AtmosphericProfile` (AFGL model atmosphere 2).
"""
function tropical_atmosphere()
    vmr = Dict{GasSpecies, Vector{Float64}}(
        H2O => copy(TROPICAL_H2O),
        CO2 => copy(US_STD_CO2),
        O3  => copy(US_STD_O3),
        CH4 => copy(US_STD_CH4),
        N2O => copy(US_STD_N2O),
        CO  => copy(US_STD_CO),
    )
    return AtmosphericProfile(
        copy(AFGL_PRESSURE),
        copy(TROPICAL_TEMPERATURE),
        copy(AFGL_ALTITUDE),
        vmr
    )
end

# ── Subarctic Summer Atmosphere ──────────────────────────────────────────────

const SUBARCTIC_TEMPERATURE = Float64[
    287.00, 282.00, 276.00, 271.00, 266.00, 260.00, 253.00, 246.00,
    239.00, 232.00, 225.00, 225.00, 225.00, 225.00, 225.00, 225.00,
    225.00, 225.00, 225.00, 225.00, 226.00, 228.00, 230.00, 232.00,
    234.00, 236.00, 238.00, 240.00, 242.00, 244.00, 246.00, 248.00,
    250.00, 252.00, 254.00, 256.00, 258.00, 260.00, 262.00, 260.00,
    250.00, 240.00, 230.00
]

const SUBARCTIC_H2O = Float64[
    5.422e-3, 4.952e-3, 4.172e-3, 2.980e-3, 1.818e-3, 1.020e-3, 5.470e-4,
    3.200e-4, 1.700e-4, 7.700e-5, 3.700e-5, 1.800e-5, 9.500e-6, 6.000e-6,
    4.500e-6, 3.800e-6, 3.600e-6, 3.600e-6, 3.700e-6, 3.800e-6, 3.900e-6,
    4.000e-6, 4.100e-6, 4.200e-6, 4.300e-6, 4.425e-6, 4.575e-6, 4.725e-6,
    4.825e-6, 4.900e-6, 4.950e-6, 5.025e-6, 5.150e-6, 5.225e-6, 5.250e-6,
    5.225e-6, 5.100e-6, 4.750e-6, 4.200e-6, 3.500e-6, 2.825e-6, 2.050e-6,
    1.330e-6
]

"""
    subarctic_atmosphere()

Return a subarctic summer `AtmosphericProfile` (AFGL model atmosphere 5).
"""
function subarctic_atmosphere()
    vmr = Dict{GasSpecies, Vector{Float64}}(
        H2O => copy(SUBARCTIC_H2O),
        CO2 => copy(US_STD_CO2),
        O3  => copy(US_STD_O3),
        CH4 => copy(US_STD_CH4),
        N2O => copy(US_STD_N2O),
        CO  => copy(US_STD_CO),
    )
    return AtmosphericProfile(
        copy(AFGL_PRESSURE),
        copy(SUBARCTIC_TEMPERATURE),
        copy(AFGL_ALTITUDE),
        vmr
    )
end

# ── AFGL US Standard 50-level (0–120 km) ────────────────────────────────────

"""
    afgl_us_standard_50lev(; data_dir="data")

Return the AFGL US Standard Atmosphere on 50 pressure levels (0–120 km) read
from `data/afgl_us_standard_50lev.csv`. Extends the standard 43-level profile
into the mesosphere and lower thermosphere, which is required for correct
radiative transfer in optically thick bands such as CO₂ 4.3 µm.

Data source: ARTS XML data repository (Anderson et al. 1986, AFGL-TR-86-0110).
Generate the CSV with scripts/build_afgl_50lev.py.
"""
function afgl_us_standard_50lev(; data_dir::String = "data")
    path = joinpath(data_dir, "afgl_us_standard_50lev.csv")
    isfile(path) || error("Profile CSV not found: $path\n" *
                          "Run scripts/build_afgl_50lev.py to generate it.")

    p   = Float64[]
    T   = Float64[]
    z   = Float64[]
    h2o = Float64[]; co2 = Float64[]; o3 = Float64[]
    n2o = Float64[]; ch4 = Float64[]; co = Float64[]

    open(path) do f
        header = readline(f)   # skip header
        for line in eachline(f)
            cols = split(line, ',')
            push!(p,   parse(Float64, cols[1]))
            push!(T,   parse(Float64, cols[2]))
            push!(z,   parse(Float64, cols[3]))
            push!(h2o, parse(Float64, cols[4]))
            push!(co2, parse(Float64, cols[5]))
            push!(o3,  parse(Float64, cols[6]))
            push!(n2o, parse(Float64, cols[7]))
            push!(ch4, parse(Float64, cols[8]))
            push!(co,  parse(Float64, cols[9]))
        end
    end

    vmr = Dict{GasSpecies, Vector{Float64}}(
        H2O => h2o, CO2 => co2, O3 => o3,
        N2O => n2o, CH4 => ch4, CO => co,
    )
    return AtmosphericProfile(p, T, z, vmr)
end
