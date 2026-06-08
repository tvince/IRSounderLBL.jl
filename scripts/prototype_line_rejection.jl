#!/usr/bin/env julia
#=
PROTOTYPE: strength-based line rejection (Option B, LBLRTM DPTMIN-style).

Per layer (T,p), drop lines whose PEAK cross-section contribution (Snorm·H(0,y),
i.e. value at line centre) is below ε × the strongest line's peak. Fewer lines →
smaller ±cutoff windows → faster gather. Stays in the fast gather/KA pattern.

Risk: the summed far wings of many dropped weak lines form a pedestal, so rejection
can bias σ low. This sweeps ε and reports σ error vs the full line set + speedup, at
a high-altitude (narrow), surface (broad), and mid layer.

Run:  julia --project -t auto scripts/prototype_line_rejection.jl
=#
using IRSounderLBL
const M = IRSounderLBL
using Printf

const ISPI = M._INV_SQRT_PI; const SQLN2 = M._SQRT_LN2

# per-line peak σ contribution at (T,p): Snorm·H(0,y) = Snorm·erfcx(y)
function line_peaks(ll, T, p; vmr_self=0.0)
    n = length(ll.lines); pk = Vector{Float64}(undef, n)
    for (j,line) in enumerate(ll.lines)
        S     = M.temperature_scaled_intensity(line, T)
        gl,gd = M.pressure_broadened_width(line, p, T; vmr_self=vmr_self)
        gd    = max(gd, 1e-10); f = SQLN2/gd; y = gl*f
        pk[j] = (S*f*ISPI) * M.faddeeva_voigt(0.0, y)
    end
    pk
end

ν1,ν2,cutoff = 645.0,800.0,25.0
grid = M.wavenumber_grid(ν1,ν2,0.001)
ll   = M.load_linelist(joinpath("data","co2_645_2760"),1:3; ν_min=ν1-cutoff, ν_max=ν2+cutoff)
prof = M.afgl_us_standard_50lev(); layers = M.layer_properties(prof)
@printf("region %.0f-%.0f  grid=%d  lines=%d  threads=%d\n",
        ν1,ν2,grid.n,length(ll.lines),Threads.nthreads())

xsec(sub, T, p) = M.compute_voigt_cross_sections(grid, sub, T, p; cutoff=cutoff, method=M.FullFaddeeva)

εs = (1e-5, 1e-4, 1e-3, 1e-2)

for (lbl,k) in (("high-alt(narrow)",45), ("mid",20), ("surface(broad)",1))
    T = layers.T_cg[M.CO2][k]; p = layers.p_cg[M.CO2][k]/1013.25
    σref = xsec(ll, T, p); peak = maximum(σref)
    pk = line_peaks(ll, T, p); pkmax = maximum(pk)
    @printf("\nlayer %2d  %-16s  T=%.1f p=%.4f   (%d lines)\n", k, lbl, T, p, length(ll.lines))
    @printf("   ε      kept     %%kept   max|Δσ|/pk   bandΣ rel-err\n")
    for ε in εs
        keep = pk .>= ε*pkmax
        sub  = M.HITRANLinelist(ll.lines[keep])
        σf   = xsec(sub, T, p)
        dabs = maximum(abs.(σf .- σref))/peak
        brel = abs(sum(σf) - sum(σref))/sum(σref)   # band-integrated (pedestal/continuum) bias
        @printf("  %.0e  %6d   %5.1f%%   %.2e     %.2e\n",
                ε, count(keep), 100count(keep)/length(keep), dabs, brel)
    end
end

# timing at mid layer for a couple thresholds
k=20; T=layers.T_cg[M.CO2][k]; p=layers.p_cg[M.CO2][k]/1013.25
pk=line_peaks(ll,T,p); pkmax=maximum(pk)
subs = Dict(ε => M.HITRANLinelist(ll.lines[pk .>= ε*pkmax]) for ε in (0.0, 1e-4, 1e-3, 1e-2))
for (_,s) in subs; xsec(s,T,p); end  # compile
bench(s,n=7)=minimum(@elapsed(xsec(s,T,p)) for _ in 1:n)
@printf("\ntiming (mid layer, min of 7):\n")
t0 = bench(subs[0.0])
@printf("  full      %6d lines  %.4f s  (1.00×)\n", length(subs[0.0].lines), t0)
for ε in (1e-4,1e-3,1e-2)
    t = bench(subs[ε])
    @printf("  ε=%.0e  %6d lines  %.4f s  (%.2f× faster)\n",
            ε, length(subs[ε].lines), t, t0/t)
end
