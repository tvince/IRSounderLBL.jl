#!/usr/bin/env julia
#=
PROTOTYPE: per-line grid decimation for the Voigt cross-section.

Each line is evaluated on a FINE grid (Δν_fine, e.g. 0.001) only within ±d_fine of
its centre, and on a COARSE grid (Δν_coarse, e.g. 0.01) in the far wing out to the
cutoff. Two accumulators (fine + coarse) are filled by a scatter loop with per-chunk
partial accumulators (no write races), then the coarse accumulator is linearly
interpolated up to the fine grid and added — once. This is the simplified, two-zone
version of LBLRTM's F1/F2/F3 decimation.

Compares against:
  • production  = compute_voigt_cross_sections (KA kernel, full fine grid)  [accuracy ref + speed]
  • plain-fine  = same math, plain threaded loop, full fine grid            [isolates decimation factor]
  • decimated   = the prototype

Run:  julia --project -t auto scripts/prototype_decimation.jl
=#
using IRSounderLBL
const M = IRSounderLBL
using Printf

const ISPI  = M._INV_SQRT_PI
const SQLN2 = M._SQRT_LN2

# core H with Option-A far-wing Lorentzian (matches production FullFaddeeva)
@inline function Hval(x, y, x_far)
    abs(x) > x_far ? y * ISPI / (x*x + y*y) : M.faddeeva_voigt(x, y)
end

# precompute per-line params for one (T, p)
function lineparams(ll, T, p; vmr_self=0.0)
    n = length(ll.lines)
    ν0=Vector{Float64}(undef,n); f=similar(ν0); y=similar(ν0); Sn=similar(ν0)
    for (j,line) in enumerate(ll.lines)
        ν0[j] = M.pressure_shift(line, p; vmr_self=vmr_self)
        S     = M.temperature_scaled_intensity(line, T)
        gl,gd = M.pressure_broadened_width(line, p, T; vmr_self=vmr_self)
        gd    = max(gd, 1e-10)
        fj    = SQLN2/gd
        f[j]=fj; y[j]=gl*fj; Sn[j]=S*fj*ISPI
    end
    ν0,f,y,Sn
end

# ---- reference: plain threaded full-fine (same math as production) ----
function plain_fine(grid, ν0,f,y,Sn; cutoff=25.0, x_far=M._X_FAR)
    ν1=grid.ν[1]; Δ=grid.Δν; nf=grid.n; n_L=length(ν0)
    σ=zeros(nf); rc=1.0/cutoff
    Hc=[Hval(cutoff*f[j], y[j], x_far) for j in 1:n_L]
    Threads.@threads for i in 1:nf
        νi=ν1+(i-1)*Δ
        lo=M._lower_bound(ν0, νi-cutoff, n_L); hi=M._upper_bound(ν0, νi+cutoff, n_L)
        acc=0.0
        for j in lo:hi
            Δν=νi-ν0[j]; x=Δν*f[j]; zr=Δν*rc
            acc += Sn[j]*(Hval(x,y[j],x_far) - (2.0-zr*zr)*Hc[j])
        end
        σ[i]=max(acc,0.0)
    end
    σ
end

# ---- prototype: per-line decimation, two accumulators ----
function decimated(grid, ν0,f,y,Sn; cutoff=25.0, x_far=M._X_FAR,
                   d_fine=1.0, dν_coarse=0.01)
    ν1=grid.ν[1]; Δ=grid.Δν; nf=grid.n; n_L=length(ν0)
    νcend=grid.ν[end]
    nc = floor(Int,(νcend-ν1)/dν_coarse)+1
    rc=1.0/cutoff
    Hc=[Hval(cutoff*f[j], y[j], x_far) for j in 1:n_L]
    nch=max(1,Threads.nthreads())
    bnds=[round(Int,(c-1)*n_L/nch)+1 : round(Int,c*n_L/nch) for c in 1:nch]
    fineP=[zeros(nf) for _ in 1:nch]; coarseP=[zeros(nc) for _ in 1:nch]
    Threads.@threads for c in 1:nch
        fine=fineP[c]; coarse=coarseP[c]
        for j in bnds[c]
            ν0j=ν0[j]; fj=f[j]; yj=y[j]; Sn_j=Sn[j]; Hcj=Hc[j]
            # fine core: output indices within ±d_fine
            ilo=max(1, ceil(Int,(ν0j-d_fine-ν1)/Δ)+1)
            ihi=min(nf, floor(Int,(ν0j+d_fine-ν1)/Δ)+1)
            @inbounds for i in ilo:ihi
                Δν=(ν1+(i-1)*Δ)-ν0j; x=Δν*fj; zr=Δν*rc
                fine[i]+=Sn_j*(Hval(x,yj,x_far)-(2.0-zr*zr)*Hcj)
            end
            # coarse wing: coarse indices in (d_fine, cutoff]
            clo=max(1, ceil(Int,(ν0j-cutoff-ν1)/dν_coarse)+1)
            chi=min(nc, floor(Int,(ν0j+cutoff-ν1)/dν_coarse)+1)
            @inbounds for cc in clo:chi
                Δν=(ν1+(cc-1)*dν_coarse)-ν0j
                abs(Δν)<=d_fine && continue
                x=Δν*fj; zr=Δν*rc
                coarse[cc]+=Sn_j*(Hval(x,yj,x_far)-(2.0-zr*zr)*Hcj)
            end
        end
    end
    fine=reduce(+,fineP); coarse=reduce(+,coarseP)
    σ=fine
    @inbounds for i in 1:nf
        νi=ν1+(i-1)*Δ; t=(νi-ν1)/dν_coarse; c0=floor(Int,t)+1
        if 1<=c0<nc
            fr=t-(c0-1); σ[i]+=coarse[c0]*(1-fr)+coarse[c0+1]*fr
        elseif c0==nc
            σ[i]+=coarse[nc]
        end
        σ[i]=max(σ[i],0.0)
    end
    σ
end

# ─────────────────────── driver ───────────────────────
ν1,ν2,cutoff = 710.0,720.0,25.0
grid = M.wavenumber_grid(ν1,ν2,0.001)
ll   = M.load_linelist(joinpath("data","co2_645_2760"),1:3; ν_min=ν1-cutoff, ν_max=ν2+cutoff)
prof = M.afgl_us_standard_50lev(); layers=M.layer_properties(prof)
@printf("region %.0f-%.0f  fine grid=%d pts  lines=%d  threads=%d\n",
        ν1,ν2,grid.n,length(ll.lines),Threads.nthreads())

# accuracy at a high-altitude layer (narrow Doppler cores = worst case) + surface
for (lbl,k) in (("high-alt (narrow cores)", 45), ("surface (broad)", 1))
    T=layers.T_cg[M.CO2][k]; p=layers.p_cg[M.CO2][k]/1013.25
    ν0,f,y,Sn = lineparams(ll,T,p)
    σref = M.compute_voigt_cross_sections(grid, ll, T, p; cutoff=cutoff, method=M.FullFaddeeva)
    σdec = decimated(grid, ν0,f,y,Sn; cutoff=cutoff)
    peak=maximum(σref)
    dabs=maximum(abs.(σdec.-σref))
    mask=σref.>1e-4*peak
    drel=maximum(abs.((σdec[mask].-σref[mask])./σref[mask]))
    @printf("  layer %2d %-24s T=%.1f p=%.4f  max|Δσ|/peak=%.2e  max rel(σ>1e-4pk)=%.2e\n",
            k,lbl,T,p,dabs/peak,drel)
end

# timing at a mid layer
k=20; T=layers.T_cg[M.CO2][k]; p=layers.p_cg[M.CO2][k]/1013.25
ν0,f,y,Sn = lineparams(ll,T,p)
# compile
M.compute_voigt_cross_sections(grid,ll,T,p;cutoff=cutoff,method=M.FullFaddeeva)
plain_fine(grid,ν0,f,y,Sn;cutoff=cutoff); decimated(grid,ν0,f,y,Sn;cutoff=cutoff)
bench(f,n=7)= minimum(@elapsed(f()) for _ in 1:n)
t_prod = bench(()->M.compute_voigt_cross_sections(grid,ll,T,p;cutoff=cutoff,method=M.FullFaddeeva))
t_pf   = bench(()->plain_fine(grid,ν0,f,y,Sn;cutoff=cutoff))
t_dec  = bench(()->decimated(grid,ν0,f,y,Sn;cutoff=cutoff))
@printf("\nper-layer cross-section time (min of 7):\n")
@printf("  production (KA, full fine) : %.4f s\n", t_prod)
@printf("  plain-threaded full fine   : %.4f s\n", t_pf)
@printf("  decimated (prototype)      : %.4f s\n", t_dec)
@printf("  decimation factor (plain/dec) = %.2fx ;  decimated vs production = %.2fx\n",
        t_pf/t_dec, t_prod/t_dec)
# analytic eval-count
nfine=2*1.0/0.001; ncoarse=2*(25.0-1.0)/0.01
@printf("  eval-count/line: full-fine=%.0f  decimated=%.0f  (%.1fx fewer)\n",
        2*cutoff/0.001, nfine+ncoarse, (2*cutoff/0.001)/(nfine+ncoarse))
