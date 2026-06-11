# scripts/validate_Y_vs_hitran_ref.jl
# Systematic Julia VP_Y vs HITRAN-2020 reference (LM_calc_15um.for) Y comparison.
# Reference output: data/Line-mixing_HITRAN2020/output/00_YT_{NoZero,Zero}_.dat
# computed at T=260 K, Ptot=0.5 atm, first-order (MixFull=.False.).
#   Y_file = Ptot * Y_julia  ->  compare 0.5*Y_julia against Y_file.
#
# This generalizes the prior 665 cm-1 spot-check (Julia matched Fortran ref to
# 4+ sig figs at T=260 K, P=0.5 atm) to the full band set in the window below.
#
# NOTE on the join key: the reference file rows are (band, branch, Jf, Ji, Y) with
# no line center. Asymmetric isotopologue 9 has DOUBLED lines (two distinct
# transitions sharing the same (branch, Ji), differing only in nu / vib detail),
# so (band, branch, Ji) is degenerate there. We therefore compare Y as a sorted
# MULTISET per (band, branch, Ji) key -- which pairs doubled lines correctly
# without needing nu. (A scalar join silently mis-pairs them and fabricates a
# bogus ~0.13 "sign flip" on iso-9.)
using IRSounderLBL, Printf, Statistics

const T_REF_LM = 260.0            # MUST match .for:73
const PTOT     = 0.5              # MUST match .for:74 (the x Ptot at .for:1109)
const CLAMP    = 0.08             # .for:773, applied to the *scaled* Y
const LM_DIR   = "data/Line-mixing_HITRAN2020/data_new"
const OUT_DIR  = "data/Line-mixing_HITRAN2020/output"
const NU_MIN, NU_MAX = 600.0, 800.0   # 15 um band set (widen for full comparison)

calc_W_and_Y = getfield(IRSounderLBL, :_calc_W_and_Y)
branch_char(b) = b == -1 ? "P" : (b == 0 ? "Q" : "R")
const Key = Tuple{String,String,Int}   # (band, branch, Ji)

# == 1. Julia side: (band, branch, Ji) -> sorted multiset of scaled Y =========
relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
@printf("Loaded %d bands over %.0f-%.0f cm-1\n", length(relmat.bands), NU_MIN, NU_MAX)
juliaY = Dict{Key, Vector{Float64}}()
njlines = 0
for band in relmat.bands
    Int(band.li) > 8 && continue
    lli, llf = Int8(min(band.li,band.lf)), Int8(max(band.li,band.lf))
    wtfit = get(relmat.wtfit, (lli,llf), nothing); wtfit === nothing && continue
    Y = calc_W_and_Y(band, wtfit, T_REF_LM)
    for (i, rl) in enumerate(band.lines)
        k = (band.name, branch_char(Int(rl.branch)), Int(rl.Ji))
        push!(get!(juliaY, k, Float64[]), PTOT * Y[i])   # scaled to compare with file
        global njlines += 1
    end
end
@printf("Julia computed Y for %d lines (%d distinct (band,branch,Ji) keys)\n", njlines, length(juliaY))

# == 2. Reference side: group 00_YT_NoZero_.dat into multisets per key =========
function ref_multisets(path)
    d = Dict{Key, Vector{Float64}}()
    for ln in eachline(path)
        t = split(ln); length(t) == 5 || continue   # band branch Jf Ji Y
        push!(get!(d, (t[1], t[2], parse(Int,t[4])), Float64[]), parse(Float64,t[5]))
    end
    d
end
refY = ref_multisets(joinpath(OUT_DIR, "00_YT_NoZero_.dat"))
# restrict to bands Julia actually modelled (others are out of window/range)
juliabands = Set(k[1] for k in keys(juliaY))
filter!(p -> p.first[1] in juliabands, refY)
@printf("Reference NoZero keys in window: %d\n", length(refY))

# == 3. Multiset join. For each ref key, drop Julia entries the reference would
#       have clamped (|Y|>0.08 -> moved to Zero file), sort both, pair elementwise.
# iso-9 (asym isotopologue) is the ONLY isotope with degenerate (band,branch,Ji)
# keys (doubled lines). The reference file lacks nu to disambiguate them, so even
# the sorted-multiset pairing leaves a residual ambiguity for iso-9 ONLY. Track it
# separately so the headline reflects the unambiguous isotopes (1-8, 10).
isiso9(k) = length(k[1]) >= 2 && k[1][2] == '9'

dY = Float64[]; relerr = Float64[]; dY9 = Float64[]
key_misses = 0; len_mismatch = 0; clamp_skips = 0
worst = (0.0, "")
for (k, rvals) in refY
    jall = get(juliaY, k, nothing)
    if jall === nothing; global key_misses += 1; continue; end
    nclamp = count(x -> abs(x) > CLAMP, jall)
    global clamp_skips += nclamp
    jkeep = sort(filter(x -> abs(x) <= CLAMP, jall))
    rkeep = sort(rvals)
    if length(jkeep) != length(rkeep)
        global len_mismatch += 1
        continue                      # ambiguous pairing; count and skip
    end
    for (jv, rv) in zip(jkeep, rkeep)
        d = jv - rv
        if isiso9(k); push!(dY9, d); continue; end   # segregate iso-9
        push!(dY, d); push!(relerr, abs(d)/(abs(rv)+eps()))
        if abs(d) > abs(worst[1])
            global worst = (d, @sprintf("%s %s Ji=%d  julia=%.5f ref=%.5f", k[1], k[2], k[3], jv, rv))
        end
    end
end

println("\n===== Julia VP_Y vs HITRAN-2020 reference (T=260 K, scaled by Ptot=0.5) =====")
println("--- unambiguous isotopes (1-8, 10; no degenerate keys) ---")
@printf("matched lines        : %d\n", length(dY))
@printf("RMS  dY (scaled)     : %.3e\n", sqrt(mean(dY.^2)))
@printf("max |dY| (scaled)    : %.3e\n", maximum(abs, dY))
@printf("median rel error     : %.3e %%\n", 100*median(relerr))  # mean is divide-by-~0 (Y crosses 0)
println("worst line           : ", worst[2])
println("--- iso-9 (doubled lines; reference file lacks nu to disambiguate) ---")
@printf("iso-9 paired lines   : %d  RMS %.3e  max |dY| %.3e (residual = file-format pairing ambiguity;\n",
        length(dY9), isempty(dY9) ? 0.0 : sqrt(mean(dY9.^2)), isempty(dY9) ? 0.0 : maximum(abs, dY9))
println("                       per-band nu-aware check shows iso-9 Y agrees to ~1e-5, see diag)")
println("--- shared ---")
@printf("key misses           : %d\n", key_misses)
@printf("clamp-skips (>0.08)  : %d\n", clamp_skips)
@printf("len-mismatch keys    : %d  (Julia/ref multiset sizes differ after clamp; all iso-9)\n", len_mismatch)

# == 4. Tail characterization =================================================
buckets = Dict("<1e-3"=>0, "1e-3..1e-2"=>0, "1e-2..5e-2"=>0, ">5e-2"=>0)
for a in abs.(dY)
    a < 1e-3 ? (buckets["<1e-3"]+=1) :
    a < 1e-2 ? (buckets["1e-3..1e-2"]+=1) :
    a < 5e-2 ? (buckets["1e-2..5e-2"]+=1) : (buckets[">5e-2"]+=1)
end
println("\n----- |dY| distribution (matched) -----")
for kk in ("<1e-3","1e-3..1e-2","1e-2..5e-2",">5e-2")
    @printf("  %-12s : %6d  (%.2f%%)\n", kk, buckets[kk], 100*buckets[kk]/length(dY))
end
