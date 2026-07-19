# Dump the temperature profile from the baseline vs spike-masked joint retrievals
# for the blacklist experiment plot (plot_iasi_joint_qmask.py).
# Honors JOINT_BASE (default us_standard) so filenames + pressure grid match the run.
#   JOINT_BASE=tropical julia --project=. scripts/dump_qmask_Tp.jl
using IRSounderLBL
using JLD2: load
using Printf

const BASE_ATM = Symbol(get(ENV, "JOINT_BASE", "us_standard"))
const BTAG     = BASE_ATM == :us_standard ? "" : "_$(BASE_ATM)"

base = afgl_atmosphere(BASE_ATM)
p = base.pressure
nlev = n_levels(base)

rB = load("data/iasi_joint$(BTAG).jld2")["rJ"]         # all channels IN the fit
rM = load("data/iasi_joint$(BTAG)_qmask.jld2")["rJ"]   # spike regions HELD OUT
TB = rB.x[1:nlev]                                        # temp block is absolute K
TM = rM.x[1:nlev]

open("data/iasi_joint_qmask_Tp.csv", "w") do io
    println(io, "level,pressure,T_baseline,T_masked")
    for i in 1:nlev
        @printf(io, "%d,%.5f,%.4f,%.4f\n", i, p[i], TB[i], TM[i])
    end
end
@printf("wrote data/iasi_joint_qmask_Tp.csv  (AFGL %s, max|ΔT|=%.3f K)\n",
        BASE_ATM, maximum(abs.(TM .- TB)))
