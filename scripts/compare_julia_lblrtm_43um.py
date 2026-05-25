"""
Compare Julia vs LBLRTM over the 4.3 µm CO₂ band (2000-2500 cm⁻¹).

Two configs were run in each code:
  - contOFF : pure CO₂ Voigt LBL (no continuum)
  - co2cont : + MT-CKD CO₂ continuum only (LBLRTM ICNTNM=6 XCO2C=1; Julia
              continua=(:co2,)), isolating the continuum from the N₂/O₂
              datasets that differ between the codes near 2330 cm⁻¹.

The headline diagnostic is the *continuum effect* on BT, (co2cont − contOFF),
computed separately in each code. Differencing within each code cancels the
common line-by-line features — including a ~+7 K Julia−LBLRTM discrepancy at
the 2364 cm⁻¹ Q-branch/band-head that is an LBL (line-shape/mixing) effect, not
a continuum one — so the comparison tests the XFACCO2 + bandhead corrections
directly.

Usage:
  python scripts/compare_julia_lblrtm_43um.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"
REGIONS = [(2000, 2386, "xfac wing"),
           (2386, 2434, "bandhead (T-corr)"),
           (2434, 2500, "high wing")]


def load(path, nu_col, bt_col):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu_col], a[bt_col]


def main():
    nj, j_off = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")
    _,  j_on  = load(f"{DATA}/julia_bt_43um_co2cont.csv", "nu_cm1", "BT_K")
    nl, l_off = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv", "wavenumber_cm1", "BT_K")
    _,  l_on  = load(f"{DATA}/lblrtm_bt_43um_co2cont.csv", "wavenumber_cm1", "BT_K")
    L_off = np.interp(nj, nl, l_off)
    L_on  = np.interp(nj, nl, l_on)

    # Absolute agreement per config (dominated by the 2364 LBL feature).
    print("=== absolute BT agreement (Julia − LBLRTM) ===")
    for tag, bj, bl in [("contOFF", j_off, L_off), ("co2cont", j_on, L_on)]:
        d = bj - bl
        print(f"  {tag}: RMS {np.sqrt((d**2).mean()):.3f}  "
              f"min {d.min():+.2f} max {d.max():+.2f} K")

    # Headline: continuum effect on BT, isolated within each code.
    dJ, dL = j_on - j_off, L_on - L_off
    err = dJ - dL
    print("\n=== CO₂ continuum EFFECT on BT (co2cont − contOFF), isolated ===")
    print(f"  Julia  effect: mean {dJ.mean():+.3f}  min {dJ.min():+.3f} K")
    print(f"  LBLRTM effect: mean {dL.mean():+.3f}  min {dL.min():+.3f} K")
    print(f"  reproduction error: mean {err.mean():+.4f}  RMS "
          f"{np.sqrt((err**2).mean()):.4f}  max|{np.abs(err).max():.3f}| K")
    for lo, hi, name in REGIONS:
        m = (nj >= lo) & (nj < hi)
        print(f"  {name:18s} {lo}-{hi}: LBLRTM peak {dL[m].min():+.2f} K  "
              f"err RMS {np.sqrt((err[m]**2).mean()):.4f} max|{np.abs(err[m]).max():.3f}|")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(nj, dL, lw=0.4, color="k", label="LBLRTM")
    ax[0].plot(nj, dJ, lw=0.4, color="tab:red", alpha=0.8, label="Julia")
    ax[0].set_ylabel("continuum ΔBT [K]")
    ax[0].set_title("CO₂ MT-CKD continuum effect on BT (co2cont − contOFF)")
    ax[0].legend(loc="lower right")
    ax[1].plot(nj, err, lw=0.3, color="tab:blue")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_title(f"reproduction error (Julia − LBLRTM), "
                    f"RMS={np.sqrt((err**2).mean()):.3f} K")
    ax[1].set_xlabel("wavenumber [cm⁻¹]"); ax[1].set_ylabel("ΔΔBT [K]")
    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_compare.png"
    plt.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
