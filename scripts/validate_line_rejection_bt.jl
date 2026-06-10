#!/usr/bin/env julia
#=
BT validation of local-adaptive line rejection (Option B), LBLRTM DPTMIN/DPTFAC-style.

Per layer, keep line j iff   peak_j > a·σ_max + b·σ_local(ν0_j)
  peak_j        = Snorm_j·H(0,y_j)         (line's centre cross-section)
  σ_local(ν0_j) = full σ at the line centre (the local opacity already present)
  a (absolute floor, ×band-max σ) kills ultra-weak lines in transparent regions;
  b (relative)   drops lines negligible vs the local opacity in busy regions.
Mirrors LBLRTM oprop.f90:  SPEAK ≤ DPTMN + DPTFC·R4(JJ).

Builds full vs rejected τ cubes, runs the SAME CIM RT + real IASI ILS, and compares
channel brightness temperature. Target: max|ΔBT| < 0.01 K.

Run:  julia --project -t auto scripts/validate_line_rejection_bt.jl
=#
using IRSounderLBL
const M = IRSounderLBL
using Printf, Statistics

const ISPI = M._INV_SQRT_PI; const SQLN2 = M._SQRT_LN2
const Nair = 2.1209e22

function line_peaks(ll, T, p)
    n=length(ll.lines); pk=Vector{Float64}(undef,n)
    for (j,line) in enumerate(ll.lines)
        S=M.temperature_scaled_intensity(line,T)
        gl,gd=M.pressure_broadened_width(line,p,T); gd=max(gd,1e-10); f=SQLN2/gd
        pk[j]=(S*f*ISPI)*M.faddeeva_voigt(0.0, gl*f)
    end
    pk
end

ν1,ν2,cutoff,Δhi = 645.0,800.0,25.0,0.001
iasi = M.IASIInstrument(ν1,ν2,0.25, round(Int,(ν2-ν1)/0.25)+1, 2.0,0.5)   # real IASI ILS
ν_hi = M.wavenumber_grid(ν1,ν2,Δhi)
ll   = M.load_linelist(joinpath("data","co2_645_2760"),1:3; ν_min=ν1-cutoff, ν_max=ν2+cutoff)
prof = M.afgl_us_standard_50lev(); layers=M.layer_properties(prof)
nlay = length(layers.p_mid); nν=ν_hi.n; n_L=length(ll.lines)
@printf("15µm band  grid=%d  lines=%d  layers=%d  threads=%d\n", nν, n_L, nlay, Threads.nthreads())

xsec(sub,T,p) = M.compute_voigt_cross_sections(ν_hi, sub, T, p; cutoff=cutoff, method=M.FullFaddeeva)

# grid index of each line centre (pressure shift ~1e-3 cm⁻¹ → negligible for σ_local)
center_idx = clamp.(round.(Int,(getfield.(ll.lines,:wavenumber).-ν1)./Δhi).+1, 1, nν)

# ── Pass 1: full τ + per-layer σ_local, peaks, σ_max ──
τ_full = zeros(nν,nlay); σloc=zeros(n_L,nlay); peaks=zeros(n_L,nlay); σmax=zeros(nlay)
Tk=zeros(nlay); pk_=zeros(nlay); coef=zeros(nlay)
for k in 1:nlay
    vmr=layers.vmr_cg[M.CO2][k]; vmr==0 && continue
    T=layers.T_cg[M.CO2][k]; p=layers.p_cg[M.CO2][k]/1013.25; Tk[k]=T; pk_[k]=p
    coef[k]=vmr*layers.Δp[k]*Nair
    σ=xsec(ll,T,p)
    τ_full[:,k].=σ.*coef[k]; σmax[k]=maximum(σ); σloc[:,k]=σ[center_idx]; peaks[:,k]=line_peaks(ll,T,p)
end

T_ave=[M.cg_temperature_mass(prof.temperature[k],prof.temperature[k+1],prof.pressure[k],prof.pressure[k+1]) for k in 1:nlay]
Tsfc=prof.temperature[1]
function rt_bt(τ)
    R=M.schwarzschild_rte(ν_hi,τ,prof.temperature,Tsfc; μ=1.0,ε_sfc=1.0,source_function=:cim,T_ave=T_ave)
    δν,kern=M.ils_kernel(Δhi,iasi.opd_max,iasi.fwhm_gauss; apodization=:gaussian)
    Rap=M.apply_ils(ν_hi,R,δν,kern)
    νi=M.wavenumber_grid(ν1,ν2,iasi.Δν)
    νi, M.brightness_temperature(νi, M._resample_to_iasi(ν_hi,Rap,νi))
end
νi, BT_full = rt_bt(τ_full)
@printf("full BT: %.2f–%.2f K (mean %.2f), %d channels\n",
        minimum(BT_full),maximum(BT_full),mean(BT_full),length(BT_full))

# ── Sweep DPTMN (absolute OD floor) / DPTFC (relative to local OD) ──
# keep line j iff  τ_peak_j > DPTMN + DPTFC·τ_local   (LBLRTM SPEAK > DPTMN+DPTFC·R4)
# with τ_peak = peak_σ·coef[k], τ_local = σ_local·coef[k].
combos=[(1e-6,0.0),(1e-5,0.0),(1e-4,0.0),(1e-5,1e-2),(1e-4,1e-2),(1e-4,3e-2)]
@printf("\n DPTMN  DPTFC   mean%%kept   max|ΔBT|   RMS ΔBT   (target<0.01K)\n")
for (DPTMN,DPTFC) in combos
    τ=zeros(nν,nlay); kept=0
    for k in 1:nlay
        coef[k]==0 && continue
        keep = (peaks[:,k].*coef[k]) .> DPTMN .+ DPTFC.*(σloc[:,k].*coef[k])
        kept += count(keep)
        σ=xsec(M.HITRANLinelist(ll.lines[keep]), Tk[k], pk_[k])
        τ[:,k].=σ.*coef[k]
    end
    _,BT=rt_bt(τ)
    dmax=maximum(abs.(BT.-BT_full)); drms=sqrt(mean((BT.-BT_full).^2))
    @printf("  %.0e  %.0e  %6.2f%%    %.2e %s  %.2e\n",
            DPTMN,DPTFC, 100kept/(nlay*n_L), dmax, dmax<0.01 ? "✓" : "✗", drms)
end
