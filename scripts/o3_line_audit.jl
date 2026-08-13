# Audit the O3 linelist: per-isotope counts and what the S>1e-23 trim removes, overall
# and specifically in the 710-730 residual window. Also the cumulative Σ(S) fraction kept
# by the trim in-window (is the trim BT-lossless HERE, or did it cut lines that matter?).
using IRSounderLBL
using Printf
const ν_LO, ν_HI = 645.0, 800.0
const O3_SMIN = 1e-23

o3f = load_linelist("data/o3_645_2760", 1:4; ν_min=ν_LO-25, ν_max=ν_HI+25)
lines = o3f.lines
@printf("Loaded %d O3 lines over [%.0f,%.0f]\n\n", length(lines), ν_LO-25, ν_HI+25)

isos = sort(unique(Int(l.iso_id) for l in lines))
@printf("%-6s %10s %10s %12s %12s\n", "iso", "total", "S>1e-23", "ΣS_all", "ΣS_kept")
for iso in isos
    li = [l for l in lines if Int(l.iso_id) == iso]
    kept = [l for l in li if l.intensity > O3_SMIN]
    @printf("iso %d  %10d %10d   %.3e   %.3e\n",
            iso, length(li), length(kept),
            sum(l.intensity for l in li; init=0.0), sum(l.intensity for l in kept; init=0.0))
end

# In the 710-730 window specifically
println("\n── 710–730 cm⁻¹ window ─────────────────────────────────")
win = [l for l in lines if 710.0 <= l.wavenumber <= 730.0]
@printf("%-6s %8s %8s   %10s %10s   %10s\n",
        "iso", "total", "kept", "ΣS_all", "ΣS_kept", "kept%")
for iso in isos
    wi = [l for l in win if Int(l.iso_id) == iso]
    isempty(wi) && continue
    kept = [l for l in wi if l.intensity > O3_SMIN]
    Sall = sum(l.intensity for l in wi; init=0.0)
    Skep = sum(l.intensity for l in kept; init=0.0)
    @printf("iso %d  %8d %8d   %.3e %.3e   %6.3f%%\n",
            iso, length(wi), length(kept), Sall, Skep, Sall>0 ? 100*Skep/Sall : 0.0)
end
wtot = sum(l.intensity for l in win; init=0.0)
wkep = sum(l.intensity for l in win if l.intensity > O3_SMIN; init=0.0)
@printf("ALL    %8d %8d   %.3e %.3e   %6.4f%%\n",
        length(win), count(l->l.intensity>O3_SMIN, win), wtot, wkep, 100*wkep/wtot)

# How much intensity sits just below the cut in-window, and how close to the residual pts?
println("\n── strongest DROPPED lines (S≤1e-23) in 710–730 ────────")
dropped = sort([l for l in win if l.intensity <= O3_SMIN], by=l->-l.intensity)
@printf("%-6s %10s %10s %8s\n", "iso", "ν", "S(296)", "E''")
for l in dropped[1:min(12, length(dropped))]
    @printf("iso %d  %9.4f  %.3e  %7.1f\n", Int(l.iso_id), l.wavenumber, l.intensity, l.lower_energy)
end
@printf("(%d dropped lines in-window; strongest %.2e vs cut %.0e)\n",
        length(dropped), isempty(dropped) ? 0.0 : dropped[1].intensity, O3_SMIN)
