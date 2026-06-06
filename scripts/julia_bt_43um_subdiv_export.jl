"""
Phase-7 layer-subdivision test: rerun the 2355-2375 cm⁻¹ window using the
86-level subdivided AFGL profile (1 km steps from 50 to 95 km) for both:
  (1) default Julia recipe   (Toon LIT + T at p_cg via log-p interp)
  (2) CG-consistent recipe   (CIM Padé + mass-weighted T_AVE)

Both compared against an LBLRTM run that uses the same 86-level profile.

If the comb collapses for BOTH variants with the finer atmosphere, the
remaining ~3.5 K with CIM at 50 levels was indeed the finite-layer-thickness
error of the linear-B-in-τ assumption.

Run with:
  julia --project -t auto scripts/julia_bt_43um_subdiv_export.jl
"""

using IRSounderLBL
using Printf

const NU_MIN  = 2355.0
const NU_MAX  = 2375.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const PROFILE_CSV = "data/afgl_us_standard_subdiv.csv"

# ── Load subdivided AFGL profile (86 levels) ────────────────────────────────
function load_subdiv_profile(path::String)
    p = Float64[]; T = Float64[]; z = Float64[]
    h2o = Float64[]; co2 = Float64[]; o3 = Float64[]
    n2o = Float64[]; ch4 = Float64[]; co = Float64[]
    open(path) do f
        readline(f)
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
    vmr = Dict{GasSpecies, Vector{Float64}}(H2O=>h2o, CO2=>co2, O3=>o3,
                                             N2O=>n2o, CH4=>ch4, CO=>co)
    return AtmosphericProfile(p, T, z, vmr)
end

prof = load_subdiv_profile(PROFILE_CSV)
@printf("loaded subdivided profile: %d levels, z = %.1f … %.1f km\n",
        length(prof.altitude), prof.altitude[1], prof.altitude[end])

n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
@printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n", NU_MIN, NU_MAX, DNU, n_ch)

function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

println("Loading CO2 linelist (iso 1–3)…")
ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
@printf("  CO2: %d lines\n", length(ll_co2))

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)

function run_and_write(outfile, T_method, source_function)
    @printf("\nForward model: cont-OFF, T_method=:%s, source_function=:%s …\n",
            T_method, source_function)
    t = time()
    ν_out, _, BT = iasi_forward_model(prof, linelists;
                                       iasi            = iasi,
                                       high_res_factor = 1,
                                       cutoff          = CUTOFF,
                                       apply_continuum = false,
                                       continua        = (),
                                       with_ils        = false,
                                       line_mixing     = nothing,
                                       T_method        = T_method,
                                       source_function = source_function)
    @printf("  %.1f s; BT %.3f–%.3f K (mean %.3f)\n",
            time() - t, minimum(BT), maximum(BT), sum(BT)/length(BT))
    open(outfile, "w") do f
        write(f, "nu_cm1,BT_K\n")
        for (ν, bt) in zip(ν_out.ν, BT)
            write(f, "$(round(ν, digits=5)),$(round(bt, digits=6))\n")
        end
    end
    println("  wrote ", outfile)
end

run_and_write(joinpath(OUTDIR, "julia_bt_43um_subdiv_toon.csv"),
              :logp_at_pcg, :toon)
run_and_write(joinpath(OUTDIR, "julia_bt_43um_subdiv_cim.csv"),
              :mass_weighted, :cim)
println("\nDone.")
