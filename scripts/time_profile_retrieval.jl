# Timing breakdown for the T(p) profile retrieval, to explain the ~30 min/pixel and
# locate the levers. Measures (after a warmup that pays the JIT):
#   1. one full forward  (iasi_forward_model) at internal_dnu = 0.0025 and 0.005;
#   2. one full analytic_jacobian (51-col, T(50)+T_sfc) at each grid;
#   3. the σ-only vs full-grad (σ+dT+dp+dself) cross-section kernel cost, to size the
#      dp/dself work the Jacobian computes but this VMR-free retrieval never uses;
#   4. the BT error from coarsening 0.0025 → 0.005 (the grid-halving gate).
#
# Run:  julia -t auto --project=. scripts/time_profile_retrieval.jl

using IRSounderLBL
using Printf

const ν_LO, ν_HI = 645.0, 800.0
const CO2_PPM, ε_SEA = 432.0, 0.98
const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"

println("threads = ", Threads.nthreads())

g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0,5.0), sralim=(15.0,180.0), zenlim=(0.0,25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
geom = ViewingGeometry(g.zen[ifov])

base = afgl_us_standard_50lev()
base.vmr[CO2] .*= (CO2_PPM*1e-6)/base.vmr[CO2][1]
nlev = n_levels(base)

co2 = load_linelist("data/co2_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
h2o = load_linelist("data/h2o_645_2760", 1:3; ν_min=ν_LO-25, ν_max=ν_HI+25)
linelists = Dict(CO2 => co2, H2O => h2o)
nchan = round(Int,(ν_HI-ν_LO)/0.25)+1
spec  = StateVectorSpec(nlev, GasSpecies[]; include_temperature=true, include_tsfc=true,
                        include_emissivity=false)
@printf("FOV #%d zen=%.1f°; state n=%d; CO2 lines=%d H2O lines=%d\n",
        ifov, g.zen[ifov], spec.n, length(co2.lines), length(h2o.lines))

fmkw(dnu) = (iasi=IASIInstrument(ν_LO,ν_HI,0.25,nchan,2.0,0.5), geom=geom, with_ils=true,
             apodization=:gaussian, apply_continuum=true, internal_dnu=dnu)

fwd(dnu) = iasi_forward_model(base, linelists; T_sfc=base.temperature[1], ε_sfc=ε_SEA,
                              dptmn=0.0, fmkw(dnu)...)
jac(dnu) = analytic_jacobian(base, linelists, spec; T_sfc=base.temperature[1], ε_sfc=ε_SEA,
                             observable=:bt, fmkw(dnu)...)

println("\n── warmup (JIT) ──"); @time fwd(0.0025); @time jac(0.0025)

println("\n── forward vs jacobian @ internal_dnu=0.0025 ──")
r25 = fwd(0.0025); tf25 = @elapsed (r25 = fwd(0.0025)); BT25 = r25[3]
tj25 = @elapsed jac(0.0025)
@printf("  forward         : %6.1f s\n", tf25)
@printf("  analytic_jacobian: %6.1f s   (%.2f× forward)\n", tj25, tj25/tf25)

println("\n── forward vs jacobian @ internal_dnu=0.005 ──")
r50 = fwd(0.005); tf50 = @elapsed (r50 = fwd(0.005)); BT50 = r50[3]
tj50 = @elapsed jac(0.005)
@printf("  forward         : %6.1f s   (%.2f× the 0.0025 forward)\n", tf50, tf50/tf25)
@printf("  analytic_jacobian: %6.1f s   (%.2f× the 0.0025 jac)\n", tj50, tj50/tj25)

dBT = abs.(BT50 .- BT25)
@printf("  BT error 0.005 vs 0.0025:  max %.1f mK  rms %.1f mK\n",
        1e3*maximum(dBT), 1e3*sqrt(sum(abs2,dBT)/length(dBT)))

# Kernel cost: σ-only vs full 4-way grad, on a representative mid-trop layer.
println("\n── cross-section kernel: σ-only vs full grad (σ+dT+dp+dself) ──")
grid = wavenumber_grid(ν_LO-16, ν_HI+16, 0.0025)   # ≈ padded internal grid
Tk, pk = 260.0, 0.5
compute_voigt_cross_sections(grid, co2, Tk, pk); compute_voigt_cross_sections_grad(grid, co2, Tk, pk)  # warm
ts = @elapsed for _ in 1:3; compute_voigt_cross_sections(grid, co2, Tk, pk); end
tg = @elapsed for _ in 1:3; compute_voigt_cross_sections_grad(grid, co2, Tk, pk); end
@printf("  σ only          : %6.1f ms/call\n", 1e3*ts/3)
@printf("  σ+dT+dp+dself   : %6.1f ms/call   (%.2f× σ-only; derivatives add %.0f%%)\n",
        1e3*tg/3, tg/ts, 100*(tg-ts)/ts)

# Model the full retrieval: 5 jacobians + 4 forwards (from the actual run: 4 accepted
# iters, no rejected line-search steps) + diagnostics.
@printf("\nmodelled run @0.0025 ≈ 5·jac + 4·fwd = %.0f s (actual 1876 s)\n", 5tj25 + 4tf25)
@printf("modelled run @0.005  ≈ 5·jac + 4·fwd = %.0f s\n", 5tj50 + 4tf50)
