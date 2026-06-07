"""
Band-head subdivision test (2386-2398 cm⁻¹). Hypothesis: the -3.9 K Julia-LBLRTM
band-head residual is finite-layer-thickness error in the coarse 5-km upper
atmosphere (especially the 95-120 km thermosphere, where T rises +60 K/layer and
the existing 50-95 km subdivision does NOT reach), analogous to the ARTS
model-top artifact.

Runs the CIM default recipe on the band head with:
  (a) native AFGL 50-level profile
  (b) a full-upper-atmosphere subdivided profile (1-km steps 50-120 km)
Both written to CSV; diff vs LBLRTM done separately in Python. If (b) warms the
band head toward LBLRTM, the residual is upper-layer discretization, not far-wing.

Run: julia --project -t auto scripts/julia_bt_43um_bandhead_subdiv.jl
"""

using IRSounderLBL
using Printf

const NU_MIN, NU_MAX = 2386.0, 2398.0
const DNU, CUTOFF = 0.005, 25.0
const OUTDIR = "data/lblrtm"
const SUBDIV_CSV = "data/afgl_us_standard_subdiv_full.csv"

function load_profile_csv(path::String)
    p = Float64[]; T = Float64[]; z = Float64[]
    h2o = Float64[]; co2 = Float64[]; o3 = Float64[]
    n2o = Float64[]; ch4 = Float64[]; co = Float64[]
    open(path) do f
        readline(f)
        for line in eachline(f)
            c = split(line, ',')
            push!(p, parse(Float64, c[1])); push!(T, parse(Float64, c[2]))
            push!(z, parse(Float64, c[3])); push!(h2o, parse(Float64, c[4]))
            push!(co2, parse(Float64, c[5])); push!(o3, parse(Float64, c[6]))
            push!(n2o, parse(Float64, c[7])); push!(ch4, parse(Float64, c[8]))
            push!(co, parse(Float64, c[9]))
        end
    end
    vmr = Dict{GasSpecies, Vector{Float64}}(H2O=>h2o, CO2=>co2, O3=>o3,
                                            N2O=>n2o, CH4=>ch4, CO=>co)
    return AtmosphericProfile(p, T, z, vmr)
end

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("band head grid: %.1f–%.1f cm⁻¹, %d points\n", NU_MIN, NU_MAX, n_ch)

println("Loading CO2 linelist (iso 1–3)…")
all_lines = HITRANLine[]
for iso in 1:3
    fn = iso == 1 ? "co2_645_2760.par" : "co2_645_2760_iso$(iso).par"
    fp = joinpath("data", fn); isfile(fp) || continue
    append!(all_lines, load_hitran_par(fp; ν_min=NU_MIN-CUTOFF, ν_max=NU_MAX+CUTOFF).lines)
end
ll = Dict{GasSpecies, HITRANLinelist}(CO2 => HITRANLinelist(all_lines))
@printf("  CO2: %d lines\n", length(all_lines))

function run_and_write(prof, tag, outfile)
    @printf("\n[%s] %d levels, z_top=%.0f km …\n", tag, length(prof.altitude),
            prof.altitude[end])
    t = time()
    ν, _, BT = iasi_forward_model(prof, ll; iasi=iasi, high_res_factor=1,
        cutoff=CUTOFF, apply_continuum=false, continua=(), with_ils=false,
        line_mixing=nothing)   # source_function=:cim is the default
    @printf("  %.1f s; band-head mean BT = %.3f K (min %.3f, max %.3f)\n",
            time()-t, sum(BT)/length(BT), minimum(BT), maximum(BT))
    open(outfile, "w") do f
        write(f, "nu_cm1,BT_K\n")
        for (x, b) in zip(ν.ν, BT); write(f, "$(round(x,digits=5)),$(round(b,digits=6))\n"); end
    end
    println("  wrote ", outfile)
end

run_and_write(afgl_us_standard_50lev(), "native 50-lev",
              joinpath(OUTDIR, "julia_bt_43um_bandhead_native.csv"))
run_and_write(load_profile_csv(SUBDIV_CSV), "subdiv 50-120km 1km",
              joinpath(OUTDIR, "julia_bt_43um_bandhead_subdiv.csv"))
println("\nDone.")
