"""
Does CO₂ line mixing close the 2364 cm⁻¹ LBL residual?

Overlays three cont-OFF spectra over the 4.3 µm band:
  - LBLRTM            (COUPLED=0, χ disabled -> pure Voigt)
  - Julia VP_Y OFF    (julia_bt_43um_contOFF.csv)
  - Julia VP_Y ON     (julia_bt_43um_contOFF_lm.csv)
and the Julia−LBLRTM difference for LM-off vs LM-on.

If LM (which LBLRTM does NOT have on here) makes Julia AGREE BETTER, that is odd
(LBLRTM has no LM); if it makes Julia agree WORSE, that quantifies how much of
the 2364 structure is line mixing that LBLRTM is simply missing.

Usage: python scripts/plot_julia_lblrtm_43um_lm.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"
ZOOM = (2300, 2420)


def load(path, nu, bt):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu], a[bt]


def rms(x):
    return float(np.sqrt(np.mean(x**2)))


def main():
    nj, j_off = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")
    _,  j_on  = load(f"{DATA}/julia_bt_43um_contOFF_lm.csv", "nu_cm1", "BT_K")
    nl, l_off = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv", "wavenumber_cm1", "BT_K")
    L = np.interp(nj, nl, l_off)

    d_off, d_on = j_off - L, j_on - L
    for tag, d in [("LM off", d_off), ("LM on ", d_on)]:
        i = np.argmax(np.abs(d))
        print(f"  {tag}: RMS {rms(d):.3f}  min {d.min():+.2f} max {d.max():+.2f} "
              f"| worst {d[i]:+.2f} @ {nj[i]:.2f}")
    # peak at 2364 specifically
    k = np.argmin(np.abs(nj - 2364.11))
    print(f"  at 2364.11: LBLRTM {L[k]:.2f}  Julia-off {j_off[k]:.2f} (Δ{d_off[k]:+.2f})"
          f"  Julia-on {j_on[k]:.2f} (Δ{d_on[k]:+.2f})")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for n, y, c, lab in [(nj, L, "k", "LBLRTM"),
                         (nj, j_off, "tab:red", "Julia VP_Y off"),
                         (nj, j_on, "tab:green", "Julia VP_Y on")]:
        ax[0].plot(n, y, lw=0.5, color=c, alpha=0.8, label=lab)
    ax[0].set_ylabel("BT [K]"); ax[0].set_xlim(*ZOOM)
    ax[0].set_title("4.3 µm cont-OFF BT — effect of CO₂ line mixing at the band head")
    ax[0].legend(loc="best")

    ax[1].plot(nj, d_off, lw=0.4, color="tab:red", label=f"Julia(off)−LBLRTM  RMS {rms(d_off):.2f}")
    ax[1].plot(nj, d_on,  lw=0.4, color="tab:green", label=f"Julia(on)−LBLRTM  RMS {rms(d_on):.2f}")
    ax[1].axhline(0, color="grey", lw=0.5); ax[1].axvline(2364.11, color="grey", ls=":", lw=0.8)
    ax[1].set_xlabel("wavenumber [cm⁻¹]"); ax[1].set_ylabel("ΔBT [K]"); ax[1].set_xlim(*ZOOM)
    ax[1].set_title("difference vs LBLRTM"); ax[1].legend(loc="best")
    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_lm.png"
    plt.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
