# Diff S-file (HITRAN 2020 LM) line parameters vs HITRAN .par for CO2 around
# the 665 cm⁻¹ Q-branch.  Tests whether ARTS's "replace CO2 with S-file lines"
# choice explains the 33 K BT gap at 665.00.
#
# What we compare per line:
#   ν₀       — line center (cm⁻¹)
#   S_T0     — line intensity at T₀ (296 K)
#   γ_air    — air-broadened HWHM (cm⁻¹/atm) at T₀
#   n_air    — temperature exponent
#   E_lower  — lower-state energy (cm⁻¹)
#
# Matching: by ν₀ to within 0.005 cm⁻¹ and same isotopologue.

using RadiativeTransfer
using Printf

const NU_MIN = 663.0
const NU_MAX = 667.0
const CUTOFF = 25.0
const LM_DIR = "data/Line-mixing_HITRAN2020/data_new"

# ── Pull HITRAN .par lines for CO2 iso-1 in window ─────────────────────────────
function load_hitran_co2(ν_min, ν_max)
    all_lines = HITRANLine[]
    for iso_id in 1:3
        fname = iso_id == 1 ? "co2_645_2760.par" : "co2_645_2760_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min, ν_max=ν_max)
        append!(all_lines, ll.lines)
    end
    all_lines
end

# Approximate S(T₀) from S-file fields the same way Julia LM does internally:
#   S(T₀) = Dipo² × PopuT0 × ν × (1 − exp(−Ct·ν/T₀))
# Ct = h·c/k_B in cm·K (per ARTS / HITRAN convention).
const _CT  = 1.4387769
const _T0  = 296.0
sfile_S_T0(rl) = rl.DipoT^2 * rl.PopuT0 * rl.ν * (1.0 - exp(-_CT * rl.ν / _T0))

# Match S-file line to HITRAN line by ν₀ within tol; require same iso.
function find_match(s_ν, s_iso, hit_lines; tol=0.005)
    best = nothing
    best_dν = tol
    for h in hit_lines
        Int(h.iso_id) == s_iso || continue
        dν = abs(Float64(h.wavenumber) - s_ν)
        if dν < best_dν
            best_dν = dν
            best = h
        end
    end
    best, best_dν
end

println("Loading relmat for $(NU_MIN)–$(NU_MAX) cm⁻¹ …")
relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
@printf("  %d bands\n", length(relmat.bands))

println("\nLoading HITRAN .par CO2 lines (iso 1–3) for $(NU_MIN-1)–$(NU_MAX+1) cm⁻¹ …")
hit_lines = load_hitran_co2(NU_MIN - 1, NU_MAX + 1)
@printf("  %d HITRAN lines\n", length(hit_lines))
hit_iso1 = filter(h -> Int(h.iso_id) == 1, hit_lines)
@printf("  %d iso-1 HITRAN lines\n", length(hit_iso1))

# Pull all S-file lines in the window (across all bands, iso-1 only for now)
s_lines = NamedTuple[]
for band in relmat.bands
    Int(band.isot) == 1 || continue
    for rl in band.lines
        if NU_MIN ≤ rl.ν ≤ NU_MAX
            push!(s_lines, (band_name=band.name, line=rl))
        end
    end
end
println("\n", length(s_lines), " S-file iso-1 lines in ", NU_MIN, "-", NU_MAX, " cm⁻¹")

# Match each S-file line to a HITRAN line
mismatches = Ref(0)
matched   = Ref(0)
big_diffs = NamedTuple[]
println("\nSample S-file vs HITRAN comparison (iso-1, 663-667 cm⁻¹):")
@printf("%-13s  %9s  %9s  %8s  %8s  %s\n",
        "band", "ν_sfile", "ν_hitran", "S_sfile", "S_hitran", "Δν (×1e-4)")
for (i, e) in enumerate(s_lines)
    rl = e.line
    h, dν = find_match(rl.ν, 1, hit_iso1; tol=0.005)
    if h === nothing
        mismatches[] += 1
        continue
    end
    matched[] += 1
    S_s = sfile_S_T0(rl)
    S_h = Float64(h.intensity)
    Δν_e4 = dν * 1e4
    if i ≤ 25 || Δν_e4 > 1.0
        @printf("%-13s  %9.5f  %9.5f  %.3e  %.3e  %+7.2f\n",
                e.band_name, rl.ν, h.wavenumber, S_s, S_h, Δν_e4)
    end
    if Δν_e4 > 1.0
        push!(big_diffs, (band=e.band_name, ν_s=rl.ν, ν_h=Float64(h.wavenumber), Δν_e4=Δν_e4))
    end
end

@printf("\nTotals: matched %d, unmatched %d  (tol=5e-3 cm⁻¹)\n", matched[], mismatches[])

println("\nS-file lines without a HITRAN match within 5e-3 cm⁻¹:")
nshow = 0
for e in s_lines
    rl = e.line
    h, _ = find_match(rl.ν, 1, hit_iso1; tol=0.005)
    if h === nothing
        @printf("  ν=%.5f  band=%-13s  Dipo=%.3e\n", rl.ν, e.band_name, rl.DipoT)
        nshow += 1
        nshow >= 10 && break
    end
end

println("\nHITRAN lines in 664.99-665.05 cm⁻¹ (around the spike):")
for h in hit_iso1
    if 664.99 <= Float64(h.wavenumber) <= 665.05
        @printf("  ν=%.5f  S=%.3e  γ_air=%.5f  E_low=%.3f\n",
                Float64(h.wavenumber), Float64(h.intensity),
                Float64(h.air_broad), Float64(h.lower_energy))
    end
end

println("\nS-file iso-1 lines in 664.99-665.05 cm⁻¹:")
for e in s_lines
    rl = e.line
    if 664.99 <= rl.ν <= 665.05
        S_s = sfile_S_T0(rl)
        @printf("  ν=%.5f  S=%.3e  γ_air=%.5f  band=%s\n",
                rl.ν, S_s, Float64(rl.gV_air), e.band_name)
    end
end
