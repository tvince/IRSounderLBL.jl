"""
Layer-thickness convergence sweep for the 4.3 µm CO₂ ν₃ band (2355-2375 cm⁻¹,
cont-OFF, CO₂ iso 1-3 only).

GOAL — physical accuracy, NOT LBLRTM-matching.  This sweep measures the RT
solver's own finite-layer-thickness error by refining the atmosphere until the
TOA brightness temperature stops changing.  The converged spectrum is the
self-defined physics truth; whichever single-layer recipe sits closest to it at
the native 50-level resolution is the most physically accurate default.

METHOD
  1. Refine the AFGL 50-level profile so that every layer above
     Z_REFINE_FLOOR with Δz > dz_max is split into equal-in-z sub-layers.
     T is interpolated LINEARLY IN z (LBLATM convention); p via log-p linear in
     z (hydrostatic for thin sub-layers).  CO₂ VMR is constant so its
     interpolation is irrelevant here.  The interpolation rule is held FIXED
     across all resolutions, so the limit is well-defined and only the solver
     discretization changes.
  2. Sweep dz_max ∈ DZ_SWEEP (coarse → fine).
  3. Run all four recipe corners of the 2×2:
        opacity T_method ∈ {:logp_at_pcg, :mass_weighted}
        source_function  ∈ {:toon, :cim}
  4. Reference = finest run per recipe.  Report RMS and max |BT(dz) − BT_fine|.
  5. Two correctness checks:
        (a) Common limit — all four recipes MUST converge to the same BT_fine
            (they are discretizations of one integral).  Cross-recipe spread at
            the finest resolution is reported; a large spread ⇒ RT bug.
        (b) Convergence order — successive |BT(dz) − BT(dz/2)| should shrink
            ~4× per halving for an O(Δz²) linear-B source.

OPTIONAL sensitivity check (RUN_PCHIP): rebuild the finest profile with a
shape-preserving (Steffen monotone) T interpolation instead of linear-in-z, and
report the gap between the two converged limits.  That gap is the
profile-representation uncertainty from the 5-km AFGL sampling, separate from
the RT discretization error.

Run with:
  julia --project -t auto scripts/convergence_sweep_43um.jl
"""

using IRSounderLBL
using Printf
using Interpolations

# ── Configuration ───────────────────────────────────────────────────────────
const NU_MIN  = 2355.0
const NU_MAX  = 2375.0
const DNU     = 0.005
const CUTOFF  = 25.0
const OUTDIR  = "data/lblrtm"
const SUMMARY = joinpath(OUTDIR, "convergence_sweep_43um_summary.csv")

# Refine only where the action is (saturation altitude ≈ 97.5 km; comb error in
# the 50-95 km mesosphere).  Lower layers are already finely spaced and not the
# error source, so leaving them coarse keeps the layer count — and the
# cross-section cost — bounded.
const Z_REFINE_FLOOR = parse(Float64, get(ENV, "Z_REFINE_FLOOR", "40.0"))   # km
const DZ_SWEEP = [5.0, 2.5, 1.25, 0.625, 0.3125]   # km, coarse → fine
const RUN_PCHIP = true                # finest-resolution interpolation sensitivity

# The 2×2 recipe corners.  (label, T_method, source_function)
const RECIPES = [
    ("toon_logp",  :logp_at_pcg,  :toon),   # current default
    ("cim_logp",   :logp_at_pcg,  :cim),    # hybrid: per-species opacity T, CG source
    ("toon_mass",  :mass_weighted, :toon),  # mass-T opacity, level source
    ("cim_mass",   :mass_weighted, :cim),   # Phase-7 winner (fully CG-consistent)
]

# ── Profile refinement ──────────────────────────────────────────────────────
"""
Insert levels so every layer above `z_floor` with Δz > `dz_max` is split into
`ceil(Δz/dz_max)` equal-in-z sub-layers.  Returns a new AtmosphericProfile.

`T_interp` selects the temperature representation between original levels:
  :linear_z  — T linear in z (LBLATM convention; the primary sweep)
  :steffen   — Steffen (2002) shape-preserving monotone cubic in z (sensitivity)
Pressure is always log-p linear in z.  Each species VMR is linear in z (CO₂ is
constant here, so this is a no-op for the 4.3 µm test).
"""
function refine_profile(prof::AtmosphericProfile, dz_max::Float64;
                        z_floor::Float64 = Z_REFINE_FLOOR,
                        T_interp::Symbol = :linear_z)
    z0 = prof.altitude            # surface-first, strictly increasing
    p0 = prof.pressure
    T0 = prof.temperature
    n0 = length(z0)

    # T interpolant over the ORIGINAL knots (in z)
    T_itp = if T_interp == :steffen
        interpolate(z0, T0, SteffenMonotonicInterpolation())
    else
        linear_interpolation(z0, T0; extrapolation_bc=Flat())
    end
    logp_itp = linear_interpolation(z0, log.(p0); extrapolation_bc=Flat())
    vmr_itp = Dict(sp => linear_interpolation(z0, v; extrapolation_bc=Flat())
                   for (sp, v) in prof.vmr)

    # Build the refined z grid
    z_new = Float64[z0[1]]
    for k in 1:(n0 - 1)
        z_lo, z_hi = z0[k], z0[k + 1]
        Δz = z_hi - z_lo
        nsub = (z_lo ≥ z_floor && Δz > dz_max) ? ceil(Int, Δz / dz_max) : 1
        for j in 1:nsub
            push!(z_new, z_lo + (Δz * j) / nsub)   # ends exactly at z_hi
        end
    end

    T_new = [T_itp(z) for z in z_new]
    p_new = [exp(logp_itp(z)) for z in z_new]
    vmr_new = Dict{GasSpecies, Vector{Float64}}(
        sp => [itp(z) for z in z_new] for (sp, itp) in vmr_itp)

    return AtmosphericProfile(p_new, T_new, z_new, vmr_new)
end

# ── Linelist (load once; range fixed) ───────────────────────────────────────
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

# ── One forward-model evaluation ────────────────────────────────────────────
function run_bt(prof, linelists, iasi, T_method, source_function)
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
    return ν_out.ν, BT
end

rms(d)  = sqrt(sum(abs2, d) / length(d))
wmax(d) = d[argmax(abs.(d))]

# ── Main ────────────────────────────────────────────────────────────────────
function main()
    n_ch = round(Int, (NU_MAX - NU_MIN) / DNU) + 1
    iasi = IASIInstrument(NU_MIN, NU_MAX, DNU, n_ch, 2.0, 0.5)
    @printf("grid: %.3f–%.3f cm⁻¹, Δν=%.4f, %d points\n",
            NU_MIN, NU_MAX, DNU, n_ch)

    println("Loading CO₂ linelist (iso 1–3)…")
    ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
    @printf("  CO₂: %d lines\n", length(ll_co2))
    linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2)

    base = afgl_us_standard_50lev()

    # Pre-refine all resolutions (linear-in-z T) and report layer counts
    profs = Dict{Float64, AtmosphericProfile}()
    println("\nRefined profiles (linear-in-z T, refine z ≥ $(Z_REFINE_FLOOR) km):")
    for dz in DZ_SWEEP
        pr = refine_profile(base, dz)
        profs[dz] = pr
        @printf("  dz_max %6.4f km → %3d levels (%3d layers)\n",
                dz, length(pr.altitude), length(pr.altitude) - 1)
    end

    # results[recipe_label][dz] = (ν, BT)
    results = Dict{String, Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}}()
    νref = Float64[]

    for (label, T_method, source_function) in RECIPES
        results[label] = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
        @printf("\n── recipe %-10s (T=%-13s src=%s) ──\n",
                label, T_method, source_function)
        for dz in DZ_SWEEP
            t = time()
            ν, BT = run_bt(profs[dz], linelists, iasi, T_method, source_function)
            isempty(νref) && (νref = ν)
            results[label][dz] = (ν, BT)
            @printf("   dz %6.4f: %5.1f s   BT mean %.3f  min %.3f  max %.3f\n",
                    dz, time() - t, sum(BT)/length(BT), minimum(BT), maximum(BT))
        end
    end

    dz_fine = minimum(DZ_SWEEP)

    # ── Self-convergence per recipe (reference = finest of that recipe) ──────
    println("\n========================================================")
    println("SELF-CONVERGENCE  |BT(dz) − BT(finest)|  per recipe")
    println("(finest dz_max = $(dz_fine) km is the reference)")
    println("========================================================")
    open(SUMMARY, "w") do io
        write(io, "recipe,dz_max_km,n_layers,rms_vs_finest_K,max_vs_finest_K\n")
        for (label, _, _) in RECIPES
            BT_fine = results[label][dz_fine][2]
            @printf("\n%-10s  %8s %8s %8s\n", label, "dz_max", "RMS", "max")
            for dz in DZ_SWEEP
                d = results[label][dz][2] .- BT_fine
                nlay = length(profs[dz].altitude) - 1
                @printf("            %8.4f %8.4f %+8.4f\n", dz, rms(d), wmax(d))
                write(io, @sprintf("%s,%.4f,%d,%.5f,%.5f\n",
                                   label, dz, nlay, rms(d), wmax(d)))
            end
        end
    end
    @printf("\nwrote %s\n", SUMMARY)

    # ── Convergence order: |BT(dz) − BT(dz/2)| ratio (expect ~4×) ───────────
    println("\n========================================================")
    println("CONVERGENCE ORDER  successive max|BT(dz) − BT(dz/2)|")
    println("(ratio → ~4 confirms O(Δz²); 'truth' = Richardson extrap.)")
    println("========================================================")
    for (label, _, _) in RECIPES
        @printf("\n%-10s\n", label)
        prev = nothing
        for i in 1:(length(DZ_SWEEP) - 1)
            dz_c, dz_f = DZ_SWEEP[i], DZ_SWEEP[i + 1]   # coarse, fine (=dz_c/2)
            d = results[label][dz_c][2] .- results[label][dz_f][2]
            m = maximum(abs.(d))
            ratio = isnothing(prev) ? NaN : prev / m
            @printf("   %6.4f→%6.4f km:  max Δ %8.4f K   ratio %s\n",
                    dz_c, dz_f, m, isnan(ratio) ? "  —" : @sprintf("%.2f", ratio))
            prev = m
        end
    end

    # ── Which native-resolution recipe is closest to the converged truth? ───
    # Richardson-extrapolate each recipe's two finest runs to estimate BT_truth,
    # then measure the NATIVE (5 km) recipe error against it.
    println("\n========================================================")
    println("NATIVE-RESOLUTION ACCURACY vs Richardson-extrapolated truth")
    println("(per recipe: BT_truth from two finest; error of dz_max=5 km run)")
    println("========================================================")
    dz_c, dz_f = DZ_SWEEP[end - 1], DZ_SWEEP[end]   # two finest
    @printf("\n%-10s  %10s %10s\n", "recipe", "native_RMS", "native_max")
    for (label, _, _) in RECIPES
        BT_c = results[label][dz_c][2]
        BT_f = results[label][dz_f][2]
        BT_truth = BT_f .+ (BT_f .- BT_c) ./ 3.0     # Richardson, p=2 ⇒ /(2²−1)
        d_native = results[label][5.0][2] .- BT_truth
        @printf("%-10s  %10.4f %+10.4f\n", label, rms(d_native), wmax(d_native))
    end

    # ── Common-limit check at finest resolution ─────────────────────────────
    println("\n========================================================")
    println("COMMON-LIMIT CHECK at finest dz_max=$(dz_fine) km")
    println("(all recipes must agree; large spread ⇒ RT bug, not discretization)")
    println("========================================================")
    ref_label = "cim_mass"
    BT_ref = results[ref_label][dz_fine][2]
    @printf("\nreference = %s\n%-10s  %10s %10s\n", ref_label, "recipe", "RMS", "max")
    for (label, _, _) in RECIPES
        d = results[label][dz_fine][2] .- BT_ref
        @printf("%-10s  %10.4f %+10.4f\n", label, rms(d), wmax(d))
    end

    # ── Optional: interpolation-representation sensitivity ───────────────────
    if RUN_PCHIP
        println("\n========================================================")
        println("PROFILE-REPRESENTATION SENSITIVITY at dz_max=$(dz_fine) km")
        println("linear-in-z T  vs  Steffen monotone-cubic T")
        println("(gap = AFGL 5-km sampling uncertainty, separate from RT error)")
        println("========================================================")
        prof_steffen = refine_profile(base, dz_fine; T_interp=:steffen)
        for (label, T_method, source_function) in RECIPES
            _, BT_lin = results[label][dz_fine]
            _, BT_stf = run_bt(prof_steffen, linelists, iasi,
                               T_method, source_function)
            d = BT_stf .- BT_lin
            @printf("%-10s  RMS %.4f K   max %+.4f K\n", label, rms(d), wmax(d))
        end
    end

    println("\nDone.")
end

main()
