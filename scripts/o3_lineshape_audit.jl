# Lineshape audit for the 710-730 O3 dipole. The reachability test (o3_reachability.jl)
# closed amount/shape; the only candidate left is per-line LINESHAPE. We already have the
# broadening parameters + their HITRAN uncertainty/reference codes on disk — never examined.
# Read γ_air, γ_self, n_air, δ_air and their Ierr codes/Iref for the 715 cluster (wants MORE
# O3) vs the 722-723 cluster (wants LESS). Questions:
#   • do the two clusters have systematically DIFFERENT γ_air or n_air? (a width bias that
#     acts oppositely on the two R-branch J-groups is the one thing that can make the dipole)
#   • how well does HITRAN claim to know these WIDTHS? (S was 1-2%; if γ_air is code 3-4 =
#     10-20%+, there's room for a width error the intensity check couldn't see)
#   • do the widths trace to the same measurement (Iref) as the intensities, or a different one?
#
# HITRAN 160-char .par (1-indexed):
#   4-15 ν · 16-25 S · 36-40 γair(F5.4) · 41-45 γself(F5.3) · 46-55 E'' · 56-59 nair(F4.2)
#   · 60-67 δair(F8.6) · 113-127 lower-local-quanta · 128-133 Ierr(ν,S,γair,γself,n,δ)
#   · 134-145 Iref(6×I2: ν,S,γair,γself,n,δ)
# Width/n codes (RELATIVE): 0 unreported·1 default·2 estimate·3 ≥20%·4 10-20%·5 5-10%·6 2-5%·7 1-2%·8 <1%
#
#   julia --project=. scripts/o3_lineshape_audit.jl

using Printf

const PAR = "data/o3_645_2760.par"
const RELMAP = Dict(0=>"unrep", 1=>"deflt", 2=>"est", 3=>"≥20%", 4=>"10-20%",
                    5=>"5-10%", 6=>"2-5%", 7=>"1-2%", 8=>"<1%")
digit(s, i) = (c = s[i]; c == ' ' ? 0 : parse(Int, c))
ref2(s, a, b) = (t = strip(s[a:b]); isempty(t) ? -1 : parse(Int, t))

struct L
    ν::Float64; S::Float64; γa::Float64; γs::Float64; E::Float64; na::Float64; δ::Float64
    cγa::Int; cn::Int; cδ::Int; rγa::Int; rn::Int; q::String
end

recs = L[]
for line in eachline(PAR)
    length(line) >= 145 || continue
    ν = parse(Float64, strip(line[4:15]));  710.0 <= ν <= 730.0 || continue
    push!(recs, L(ν,
        parse(Float64, strip(line[16:25])),          # S
        parse(Float64, strip(line[36:40])),          # γ_air
        parse(Float64, strip(line[41:45])),          # γ_self
        parse(Float64, strip(line[46:55])),          # E''
        parse(Float64, strip(line[56:59])),          # n_air
        parse(Float64, strip(line[60:67])),          # δ_air
        digit(line,130), digit(line,132), digit(line,133),   # Ierr γair, n, δ
        ref2(line,138,139), ref2(line,142,143),              # Iref γair, n
        strip(line[113:127])))                        # lower local quanta
end
sort!(recs, by=r->r.ν)
@printf("Read %d iso-1 O3 lines in 710-730\n", length(recs))

function cluster(name, lo, hi)
    w = [r for r in recs if lo <= r.ν <= hi]
    Wtot = sum(r.S for r in w; init=0.0)
    smean(f) = sum(r.S*f(r) for r in w)/Wtot    # S-weighted mean
    @printf("\n══ %s  [%.2f,%.2f]  %d lines ══\n", name, lo, hi, length(w))
    @printf("  S-weighted:  γ_air=%.4f  γ_self=%.4f  n_air=%.3f  δ_air=%+.5f  ⟨E''⟩=%.1f\n",
            smean(r->r.γa), smean(r->r.γs), smean(r->r.na), smean(r->r.δ), smean(r->r.E))
    # width-uncertainty mix (S-weighted)
    @printf("  γ_air uncertainty mix (S-wtd): ")
    for code in sort(unique(r.cγa for r in w))
        frac = sum(r.S for r in w if r.cγa==code; init=0.0)/Wtot
        @printf("%s=%.0f%% ", RELMAP[code], 100*frac)
    end
    println()
    @printf("  n_air uncertainty mix (S-wtd): ")
    for code in sort(unique(r.cn for r in w))
        frac = sum(r.S for r in w if r.cn==code; init=0.0)/Wtot
        @printf("%s=%.0f%% ", RELMAP[code], 100*frac)
    end
    println()
    @printf("  γ_air refs: %s   n_air refs: %s\n",
            join(sort(unique(r.rγa for r in w)), ","), join(sort(unique(r.rn for r in w)), ","))
    @printf("  %10s %9s %7s %6s %6s  %-6s %-6s  %s\n",
            "ν","S","γ_air","n_air","δ_air","γ_unc","n_unc","lower-quanta")
    for r in sort(w, by=r->-r.S)[1:min(8,length(w))]
        @printf("  %10.4f %.2e %6.4f %6.3f %+6.4f  %-6s %-6s  %s\n",
                r.ν, r.S, r.γa, r.na, r.δ, RELMAP[r.cγa], RELMAP[r.cn], r.q)
    end
    return w
end

c715 = cluster("715 cluster (wants MORE O3)", 714.5, 717.0)
c722 = cluster("722-723 cluster (wants LESS O3)", 721.5, 724.0)

Wt(w) = sum(r.S for r in w)
gwt(w) = sum(r.S*r.γa for r in w)/Wt(w)
nwt(w) = sum(r.S*r.na for r in w)/Wt(w)
@printf("\n── cluster contrast (S-weighted) ──\n")
@printf("  γ_air: 715=%.4f  722=%.4f  Δ=%+.4f (%.1f%%)\n",
        gwt(c715), gwt(c722), gwt(c715)-gwt(c722), 100*(gwt(c715)-gwt(c722))/gwt(c722))
@printf("  n_air: 715=%.3f  722=%.3f  Δ=%+.3f\n", nwt(c715), nwt(c722), nwt(c715)-nwt(c722))
