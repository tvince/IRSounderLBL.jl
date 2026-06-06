"""
Visualize LBLRTM's panel-based Voigt sampling vs Julia's uniform grid.

LBLRTM tabulates the Lorentz line shape on three nested panels (SHAPEL in
LBLRTM/src/oprop.f90), each capturing the part of L(x) the coarser panel
cannot represent:
  Panel 1 (F1):  0 .. HWF1=4   half-widths, step DXF1=0.002 HW  (fine)
  Panel 2 (F2):  0 .. HWF2=16  half-widths, step DXF2=0.008 HW  (4x coarser)
  Panel 3 (F3):  0 .. HWF3=64  half-widths, step DXF3=0.032 HW  (16x coarser)
  Beyond 64 HW: only the slowly-varying parabolic interpolant Q3(x), no table.

At the OUTPUT grid the same hierarchy applies: contributions from |dnu|<4 a_V
land on the fine panel at spacing DV, 4..16 a_V on a 4*DV panel, 16..64 a_V
on a 16*DV panel. So the effective Voigt sampling step grows in steps with
distance from line center, then the panel cuts off at 25 cm-1.

Julia (IRSounderLBL voigt.jl) evaluates the exact Faddeeva at every output
point at a single uniform DV out to the same 25 cm-1 cutoff.

We plot the effective dnu(x) for both for two regimes:
  - LT (lower troposphere):  pressure-broadened, a_V ~ 0.05 cm-1
  - US (upper stratosphere): Doppler-limited at 2364 cm-1 CO2, a_V ~ 0.0017 cm-1

LBLRTM DV is taken from the actual 4.3 um run TAPE12 grid.

Usage:
  python scripts/plot_lblrtm_voigt_grid.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sampling constants
DV_LBLRTM = 0.000514   # cm-1, auto-set in TAPE5 (SAMPLE=4, alpha_min ~ 0.002)
DV_JULIA  = 0.005      # cm-1, fixed in scripts/julia_bt_43um_export.jl
CUTOFF    = 25.0       # cm-1, line-wing cutoff used by both codes

# Panel half-width boundaries (in units of a_V, from oprop.f90 DATA stmts)
HWF1, HWF2, HWF3 = 4.0, 16.0, 64.0

REGIMES = [
    ("Lower troposphere\np ~ 1 atm, T ~ 270 K",
     0.05,   "tab:orange"),
    ("Upper stratosphere @ 2364 cm$^{-1}$\np ~ 5 hPa, T ~ 250 K (CO$_2$ $\\nu_3$ Doppler-limited)",
     0.0017, "tab:purple"),
]


def lblrtm_dx(x_cm1, a_V, dv):
    """Effective LBLRTM Voigt sampling step (cm-1) at offset x_cm1 from line center."""
    out = np.full_like(x_cm1, np.nan)
    out[(x_cm1 >= 0)         & (x_cm1 <  HWF1*a_V)] = dv
    out[(x_cm1 >= HWF1*a_V)  & (x_cm1 <  HWF2*a_V)] = 4 * dv
    out[(x_cm1 >= HWF2*a_V)  & (x_cm1 <  HWF3*a_V)] = 16 * dv
    # Beyond 64 a_V LBLRTM uses only the analytic Q3(x) - no table lookup -
    # so the effective step is "infinite" in the sense that the line shape
    # is treated as a smooth polynomial there. Render as a dotted ceiling.
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (label, a_V, color) in zip(axes, REGIMES):
        x = np.linspace(1e-5, CUTOFF, 200_000)

        # LBLRTM step function (table lookup)
        dx_lbl = lblrtm_dx(x, a_V, DV_LBLRTM)
        ax.plot(x, dx_lbl, color="k", lw=2.2, label="LBLRTM tabulated (F1/F2/F3)")

        # Julia uniform grid
        ax.axhline(DV_JULIA, color=color, lw=2.2, ls="--",
                   label=f"Julia uniform   $\\Delta\\nu$ = {DV_JULIA} cm$^{{-1}}$")
        ax.axhline(DV_LBLRTM, color="k", lw=0.8, ls=":", alpha=0.6,
                   label=f"LBLRTM grid DV = {DV_LBLRTM:.4f} cm$^{{-1}}$")

        # Panel boundary annotations
        for hw, name in [(HWF1, "F1/F2"), (HWF2, "F2/F3"), (HWF3, "F3/Q$_3$")]:
            bx = hw * a_V
            if bx < CUTOFF:
                ax.axvline(bx, color="grey", lw=0.6, alpha=0.4)
                ax.text(bx, 3.5e-2, f" {name}\n ({hw:.0f}$\\alpha_V$)",
                        fontsize=8, color="grey", rotation=90,
                        verticalalignment="top")

        # Cutoff
        ax.axvline(CUTOFF, color="firebrick", lw=0.8, alpha=0.5)
        ax.text(CUTOFF, 1.3e-4, " 25 cm$^{-1}$\n cutoff",
                fontsize=8, color="firebrick", rotation=90,
                verticalalignment="bottom")

        # Shaded region beyond F3 where LBLRTM uses only the analytic Q3
        # interpolant (no table lookup at all)
        if HWF3 * a_V < CUTOFF:
            ax.axvspan(HWF3 * a_V, CUTOFF, color="khaki", alpha=0.18,
                       label="LBLRTM: analytic Q$_3$ only")

        # samples/FWHM annotation
        fwhm = 2 * a_V
        spf_l = DV_LBLRTM / fwhm   # step per FWHM
        spf_j = DV_JULIA  / fwhm
        ax.text(0.02, 0.04,
                f"FWHM $\\approx$ {fwhm:.4f} cm$^{{-1}}$\n"
                f"samples per FWHM at line peak:\n"
                f"  LBLRTM: {1/spf_l:6.1f}\n"
                f"  Julia : {1/spf_j:6.2f}",
                transform=ax.transAxes, fontsize=9,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 50)
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xlabel("offset from line center $|\\nu - \\nu_0|$ [cm$^{-1}$]")
        ax.set_title(label + f"\n$\\alpha_V$ = {a_V} cm$^{{-1}}$", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    axes[0].set_ylabel("effective Voigt sampling step $\\Delta\\nu$ [cm$^{-1}$]")
    fig.suptitle(
        "Voigt sampling step vs offset from line center — "
        "LBLRTM 3-panel scheme vs Julia uniform grid  "
        f"(4.3 $\\mu$m run, cutoff {CUTOFF:.0f} cm$^{{-1}}$)",
        fontsize=11.5)
    plt.tight_layout()
    out = "data/lblrtm/lblrtm_voigt_grid_vs_julia.png"
    plt.savefig(out, dpi=140)
    print(f"wrote {out}")

    # Also print a numerical summary
    print("\n=== Effective Voigt sampling at line CENTER (offset 0) ===")
    for label, a_V, _ in REGIMES:
        fwhm = 2 * a_V       # rough; Voigt FWHM ~ 2 a_V when one width dominates
        per_fwhm_l = fwhm / DV_LBLRTM
        per_fwhm_j = fwhm / DV_JULIA
        regime_short = label.split("\n")[0]
        print(f"  {regime_short:40s} a_V={a_V:.4f}  FWHM~{fwhm:.4f} cm-1")
        print(f"     LBLRTM samples/FWHM = {per_fwhm_l:6.2f}")
        print(f"     Julia  samples/FWHM = {per_fwhm_j:6.2f}")


if __name__ == "__main__":
    main()
