# Phase 2b — validate ∂σ/∂p and ∂σ/∂vmr_self vs FD.
using IRSounderLBL
using Printf

mkC(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                           Float32(0.07), Float32(0.08), E, Float32(0.75), Float32(-0.006))
mkW(ν0, S, E) = HITRANLine(Int8(1), Int8(1), ν0, S, 1.0,    # H2O: self-broadening active
                           Float32(0.09), Float32(0.45), E, Float32(0.70), Float32(0.005))
g = wavenumber_grid(698.0, 705.0, 0.002)

function check(ll, T, p, vs, tag)
    gr = compute_voigt_cross_sections_grad(g, ll, T, p; vmr_self=vs)
    σf = compute_voigt_cross_sections(g, ll, T, p; vmr_self=vs)
    nz = findall(>(maximum(gr.σ)*1e-6), gr.σ)
    rel(a, fd) = maximum(abs.(a[nz] .- fd[nz])) / (maximum(abs.(fd[nz])) + 1e-300)
    # FD in p
    hp = 1e-4
    fdp = (compute_voigt_cross_sections(g, ll, T, p+hp; vmr_self=vs) .-
           compute_voigt_cross_sections(g, ll, T, p-hp; vmr_self=vs)) ./ (2hp)
    # FD in vmr_self
    relv = NaN
    if vs > 0
        hv = 1e-5
        fdv = (compute_voigt_cross_sections(g, ll, T, p; vmr_self=vs+hv) .-
               compute_voigt_cross_sections(g, ll, T, p; vmr_self=vs-hv)) ./ (2hv)
        relv = rel(gr.dself, fdv)
    end
    @printf("%-10s σ≡fwd=%.1e | dσ/dp rel=%.2e | dσ/dvs rel=%s\n",
            tag, maximum(abs.(gr.σ .- σf)), rel(gr.dp, fdp),
            vs > 0 ? @sprintf("%.2e", relv) : "n/a")
end

llC = HITRANLinelist([mkC(700.3, 2.0e-21, 250.0), mkC(701.4, 8.0e-22, 600.0)])
llW = HITRANLinelist([mkW(700.6, 3.0e-21, 200.0), mkW(702.3, 1.2e-21, 450.0)])
check(llC, 260.0, 0.5, 0.0,  "CO2")
check(llC, 240.0, 0.8, 0.0,  "CO2 hi-p")
check(llW, 280.0, 0.9, 0.03, "H2O self")
check(llW, 295.0, 1.0, 0.02, "H2O self2")
