# In-hand spectroscopy check for the 710-730 O3 residual: read the HITRAN per-line
# uncertainty codes + reference IDs straight from the .par records we already have
# (data/o3_645_2760.par, iso-1). No fetch needed. Answers "how well does HITRAN itself
# claim to know the 715.5-715.7 R-branch lines?" and "do they trace to a different
# measurement than the 722-723 lines?" — the two clusters that pull O3 opposite ways.
#
# HITRAN .par (160-char) field map (1-indexed):
#   1-2 mol · 3 iso · 4-15 ν · 16-25 S · 46-55 E'' · 128-133 Ierr(ν,S,γair,γself,n,δ)
#   · 134-145 Iref(6×I2)
# Error-code meaning differs by parameter:
#   ν (position, ABSOLUTE cm⁻¹): 0 ≥1 · 1 ≥0.1 · 2 ≥0.01 · 3 ≥0.001 · 4 ≥1e-4 · 5 ≥1e-5 · 6 <1e-5
#   S / widths (RELATIVE): 0 unreported · 1 default · 2 estimate · 3 ≥20% · 4 10-20%
#                          · 5 5-10% · 6 2-5% · 7 1-2% · 8 <1%
#
#   julia --project=. scripts/o3_hitran_uncertainty.jl

using Printf

const PAR = "data/o3_645_2760.par"                       # iso-1 O3
const POSMAP = Dict(0=>"≥1 cm⁻¹", 1=>"≥0.1", 2=>"≥0.01", 3=>"≥0.001",
                    4=>"1e-4–1e-3", 5=>"1e-5–1e-4", 6=>"<1e-5")
const RELMAP = Dict(0=>"unreported", 1=>"default", 2=>"estimate", 3=>"≥20%",
                    4=>"10–20%", 5=>"5–10%", 6=>"2–5%", 7=>"1–2%", 8=>"<1%")

digit(s, i) = (c = s[i]; c == ' ' ? 0 : parse(Int, c))

struct Rec; ν::Float64; S::Float64; E::Float64; cν::Int; cS::Int; refS::Int; end

recs = Rec[]
for line in eachline(PAR)
    length(line) >= 145 || continue
    ν = parse(Float64, strip(line[4:15]))
    710.0 <= ν <= 730.0 || continue
    S = parse(Float64, strip(line[16:25]))
    E = parse(Float64, strip(line[46:55]))
    cν = digit(line, 128); cS = digit(line, 129)
    refS = parse(Int, strip(line[136:137]))
    push!(recs, Rec(ν, S, E, cν, cS, refS))
end
sort!(recs, by=r->r.ν)
@printf("Read %d iso-1 O3 lines in 710–730 from %s\n", length(recs), PAR)

function census(name, lo, hi)
    w = [r for r in recs if lo <= r.ν <= hi]
    @printf("\n── %s  [%.2f, %.2f]  (%d lines) ─────────────────────\n", name, lo, hi, length(w))
    # strongest lines: their position & intensity uncertainty and S-reference
    strong = sort(w, by=r->-r.S)[1:min(8, length(w))]
    @printf("  %10s %10s %7s  %-11s %-9s %5s\n", "ν", "S(296)", "E''", "S unc", "ν unc", "refS")
    for r in strong
        @printf("  %10.4f %.3e %7.1f  %-11s %-9s %5d\n",
                r.ν, r.S, r.E, RELMAP[r.cS], POSMAP[r.cν], r.refS)
    end
    # S-intensity-weighted uncertainty-code distribution (who carries the absorption)
    Wtot = sum(r.S for r in w; init=0.0)
    @printf("  S-weighted intensity-uncertainty mix:\n")
    for code in sort(unique(r.cS for r in w))
        frac = sum(r.S for r in w if r.cS == code; init=0.0) / Wtot
        @printf("     code %d (%-10s): %5.1f%% of ΣS\n", code, RELMAP[code], 100*frac)
    end
    refs = sort(unique(r.refS for r in w))
    @printf("  S references present: %s\n", join(refs, ", "))
    return w
end

c715 = census("715 cluster (wants MORE O3)", 714.5, 717.0)
c722 = census("722–723 cluster (want LESS O3)", 721.5, 724.0)

# Do the two clusters trace to different intensity measurements?
r715 = Set(r.refS for r in c715); r722 = Set(r.refS for r in c722)
@printf("\nS-reference overlap: 715∩722 = %s · only-715 = %s · only-722 = %s\n",
        join(sort(collect(intersect(r715, r722))), ","),
        join(sort(collect(setdiff(r715, r722))), ","),
        join(sort(collect(setdiff(r722, r715))), ","))
