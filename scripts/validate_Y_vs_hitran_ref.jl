# scripts/validate_Y_vs_hitran_ref.jl
# Systematic Julia VP_Y vs HITRAN-2020 reference (LM_calc_15um.for) Y comparison.
# Reference output: data/Line-mixing_HITRAN2020/output/00_YT_{NoZero,Zero}_.dat
# computed at T=260 K, Ptot=0.5 atm, first-order (MixFull=.False.).
#   Y_file = Ptot * Y_julia  ->  compare 0.5*Y_julia against Y_file.
#
# This generalizes the prior 665 cm-1 spot-check (Julia matched Fortran ref to
# 4+ sig figs at T=260 K, P=0.5 atm) to the full band set in the window below.
using IRSounderLBL, Printf, Statistics

const T_REF_LM = 260.0            # MUST match .for:73
const PTOT     = 0.5              # MUST match .for:74 (the x Ptot at .for:1109)
const CLAMP    = 0.08             # .for:773, applied to the *scaled* Y
const LM_DIR   = "data/Line-mixing_HITRAN2020/data_new"
const OUT_DIR  = "data/Line-mixing_HITRAN2020/output"
const NU_MIN, NU_MAX = 600.0, 800.0   # 15 um band set (widen for full comparison)

calc_W_and_Y = getfield(IRSounderLBL, :_calc_W_and_Y)
branch_char(b) = b == -1 ? "P" : (b == 0 ? "Q" : "R")

# == 1. Julia side: (name, branch, Ji) -> Y_per_atm at T_REF_LM ===============
relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
@printf("Loaded %d bands over %.0f-%.0f cm-1\n", length(relmat.bands), NU_MIN, NU_MAX)
juliaY = Dict{Tuple{String,String,Int}, Float64}()
for band in relmat.bands
    Int(band.li) > 8 && continue
    lli, llf = Int8(min(band.li,band.lf)), Int8(max(band.li,band.lf))
    wtfit = get(relmat.wtfit, (lli,llf), nothing); wtfit === nothing && continue
    Y = calc_W_and_Y(band, wtfit, T_REF_LM)
    for (i, rl) in enumerate(band.lines)
        juliaY[(band.name, branch_char(Int(rl.branch)), Int(rl.Ji))] = Y[i]
    end
end
@printf("Julia computed Y for %d lines\n", length(juliaY))

# == 2. Reference side: parse 00_YT_NoZero_.dat (band branch Jf Ji Y) =========
function parse_nozero(path)
    rows = Tuple{String,String,Int,Int,Float64}[]
    for ln in eachline(path)
        t = split(ln); length(t) == 5 || continue   # band br Jf Ji Y
        push!(rows, (t[1], t[2], parse(Int,t[3]), parse(Int,t[4]), parse(Float64,t[5])))
    end
    rows
end
function parse_zero(path)
    rows = Tuple{String,String,Int,Int}[]
    for ln in eachline(path)
        t = split(ln); length(t) == 4 || continue   # band br Jf Ji (no Y)
        push!(rows, (t[1], t[2], parse(Int,t[3]), parse(Int,t[4])))
    end
    rows
end
refrows = parse_nozero(joinpath(OUT_DIR, "00_YT_NoZero_.dat"))
zerorows = parse_zero(joinpath(OUT_DIR, "00_YT_Zero_.dat"))

# Restrict reference rows to bands present in the Julia window (others are out of range)
juliabands = Set(k[1] for k in keys(juliaY))
refwin  = filter(r -> r[1] in juliabands, refrows)
zerowin = filter(r -> r[1] in juliabands, zerorows)
@printf("Reference NoZero lines in window: %d  (Zero: %d)\n", length(refwin), length(zerowin))

# == 3. Join + stats. Compare 0.5*Y_julia (scaled) against Y_file. ===========
dY = Float64[]; relerr = Float64[]; misses = 0; clamp_skips = 0
worst = (0.0, "")
for (band, br, Jf, Ji, yfile) in refwin
    yj = get(juliaY, (band, br, Ji), nothing)
    if yj === nothing; global misses += 1; continue; end
    yj_scaled = PTOT * yj
    if abs(yj_scaled) > CLAMP; global clamp_skips += 1; continue; end  # ref would've zeroed it
    d = yj_scaled - yfile
    push!(dY, d)
    push!(relerr, abs(d) / (abs(yfile) + eps()))
    if abs(d) > abs(worst[1])
        global worst = (d, @sprintf("%s %s Ji=%d  julia=%.5f ref=%.5f", band, br, Ji, yj_scaled, yfile))
    end
end

println("\n===== Julia VP_Y vs HITRAN-2020 reference (T=260 K, scaled by Ptot=0.5) =====")
@printf("matched lines        : %d\n", length(dY))
@printf("RMS  dY (scaled)     : %.3e\n", sqrt(mean(dY.^2)))
@printf("max |dY| (scaled)    : %.3e\n", maximum(abs, dY))
@printf("median rel error     : %.3e %%\n", 100*median(relerr))  # mean is a divide-by-~0 artifact (Y crosses 0)
@printf("misses (no Julia ln) : %d\n", misses)
@printf("clamp-skips (>0.08)  : %d\n", clamp_skips)
println("worst line           : ", worst[2])

# == 4. Zero-file cross-check: lines the reference zeroed where Julia is moderate ==
# A genuine mismatch is a Zero-file line where Julia gives |0.5*Y| in (eps, CLAMP).
zero_mismatch = 0
for (band, br, Jf, Ji) in zerowin
    yj = get(juliaY, (band, br, Ji), nothing); yj === nothing && continue
    s = PTOT * abs(yj)
    if 1e-4 < s <= CLAMP
        global zero_mismatch += 1
        zero_mismatch <= 10 && @printf("  ZERO-MISMATCH %s %s Ji=%d  julia(scaled)=%.5f (ref=0)\n", band, br, Ji, PTOT*yj)
    end
end
@printf("Zero-file mismatches (Julia moderate, ref=0): %d\n", zero_mismatch)

# == 5. Tail characterization: how concentrated is the disagreement? ==========
buckets = Dict("<1e-3"=>0, "1e-3..1e-2"=>0, "1e-2..5e-2"=>0, ">5e-2"=>0)
tail = Tuple{Float64,String}[]
for (band, br, Jf, Ji, yfile) in refwin
    yj = get(juliaY, (band, br, Ji), nothing); yj === nothing && continue
    s = PTOT * yj; abs(s) > CLAMP && continue
    a = abs(s - yfile)
    a < 1e-3      ? (buckets["<1e-3"]      += 1) :
    a < 1e-2      ? (buckets["1e-3..1e-2"] += 1) :
    a < 5e-2      ? (buckets["1e-2..5e-2"] += 1) :
                    (buckets[">5e-2"]      += 1)
    a > 1e-2 && push!(tail, (s - yfile, @sprintf("%-14s %s Ji=%-3d julia=%+.5f ref=%+.5f", band, br, Ji, s, yfile)))
end
println("\n----- |dY| distribution (matched, unclamped) -----")
for k in ("<1e-3","1e-3..1e-2","1e-2..5e-2",">5e-2")
    @printf("  %-12s : %6d  (%.2f%%)\n", k, buckets[k], 100*buckets[k]/length(dY))
end
sort!(tail, by=x->-abs(x[1]))
@printf("\nlines with |dY|>1e-2: %d   (top 15 by |dY|)\n", length(tail))
for (d, s) in tail[1:min(15,end)]; @printf("  dY=%+.4f  %s\n", d, s); end
