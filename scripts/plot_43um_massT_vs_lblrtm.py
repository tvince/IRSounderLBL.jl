"""
Phase-3 plot: does the +5..+7 K Julia−LBLRTM per-line core comb in 2355-2375
cm⁻¹ collapse when Julia switches from T linearly-interpolated-in-log(p) to
the LBLATM-style mass-weighted layer T?

Three panels:
  Top:    BT — LBLRTM, Julia(linlogp), Julia(massT)
  Middle: Julia(linlogp) − LBLRTM   (existing comb)
  Bottom: Julia(massT)   − LBLRTM   (does it shrink?)

Usage:
  python scripts/plot_43um_massT_vs_lblrtm.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"
W_MIN, W_MAX = 2355.0, 2375.0


def load(path, nu_col, bt_col):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu_col], a[bt_col]


def window(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    return nu[m], y[m]


def main():
    nl, bt_l = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv",
                    "wavenumber_cm1", "BT_K")
    nj_a, bt_ja = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")
    nj_m, bt_jm = load(f"{DATA}/julia_bt_43um_massT_contOFF.csv", "nu_cm1", "BT_K")

    nl,   bt_l   = window(nl,   bt_l,   W_MIN, W_MAX)
    nj_a, bt_ja  = window(nj_a, bt_ja,  W_MIN, W_MAX)
    nj_m, bt_jm  = window(nj_m, bt_jm,  W_MIN, W_MAX)

    bt_l_on_a = np.interp(nj_a, nl, bt_l)
    bt_l_on_m = np.interp(nj_m, nl, bt_l)
    d_a = bt_ja - bt_l_on_a
    d_m = bt_jm - bt_l_on_m

    def stats(label, d):
        rms = np.sqrt((d**2).mean())
        i = np.argmax(np.abs(d))
        print(f"  {label:30s} RMS {rms:6.3f} K  worst {d[i]:+6.2f} K")
        return rms, i

    print("=== 2355–2375 cm⁻¹, cont-OFF (Julia − LBLRTM) ===")
    rms_a, ia = stats("Julia T linear-in-log(p)", d_a)
    rms_m, im = stats("Julia T mass-weighted",    d_m)
    print(f"\n  ratio massT/linlogp RMS = {rms_m/rms_a:.3f}")

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax[0].plot(nl,   bt_l,   lw=0.5, color="k",        label="LBLRTM")
    ax[0].plot(nj_a, bt_ja,  lw=0.5, color="tab:red",  alpha=0.85,
               label="Julia (T at p$_{cg}$, log-p interp)")
    ax[0].plot(nj_m, bt_jm,  lw=0.5, color="tab:blue", alpha=0.85,
               label="Julia (T mass-weighted, LBLATM-style)")
    ax[0].set_ylabel("BT [K]")
    ax[0].set_title("CO$_2$ ν₃ band head, 4.3 µm, cont-OFF — interpolation test")
    ax[0].legend(loc="best", fontsize=9)

    ax[1].plot(nj_a, d_a, lw=0.4, color="tab:red")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("ΔBT [K]")
    ax[1].set_title(f"Julia (log-p T) − LBLRTM    "
                    f"RMS = {rms_a:.3f} K   worst {d_a[ia]:+.2f} K @ {nj_a[ia]:.2f}")

    ax[2].plot(nj_m, d_m, lw=0.4, color="tab:blue")
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].set_xlabel("wavenumber [cm⁻¹]")
    ax[2].set_ylabel("ΔBT [K]")
    ax[2].set_title(f"Julia (mass-weighted T) − LBLRTM    "
                    f"RMS = {rms_m:.3f} K   worst {d_m[im]:+.2f} K @ {nj_m[im]:.2f}")

    ymax = max(abs(d_a).max(), abs(d_m).max()) * 1.1
    ax[1].set_ylim(-ymax, ymax)
    ax[2].set_ylim(-ymax, ymax)

    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_massT_vs_linlogp.png"
    plt.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
