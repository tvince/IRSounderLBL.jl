"""
Phase-6 plot: does the +5..+7 K Julia−LBLRTM comb collapse when Julia uses the
LBLRTM-style Clough-Iacono-Moncet source function with mass-weighted T?

Four panels:
  Top:    BT — LBLRTM + 3 Julia variants
  Then for each Julia variant: ΔBT vs LBLRTM

Usage:
  python scripts/plot_43um_cim_vs_lblrtm.py
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
    nj_a, bt_ja = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")          # default: Toon LIT + log-p T
    nj_m, bt_jm = load(f"{DATA}/julia_bt_43um_massT_contOFF.csv", "nu_cm1", "BT_K")    # Toon LIT + mass-wt T
    nj_c, bt_jc = load(f"{DATA}/julia_bt_43um_cim_contOFF.csv", "nu_cm1", "BT_K")      # CIM Pade + mass-wt T

    nl,   bt_l   = window(nl,   bt_l,   W_MIN, W_MAX)
    nj_a, bt_ja  = window(nj_a, bt_ja,  W_MIN, W_MAX)
    nj_m, bt_jm  = window(nj_m, bt_jm,  W_MIN, W_MAX)
    nj_c, bt_jc  = window(nj_c, bt_jc,  W_MIN, W_MAX)

    bt_l_on_a = np.interp(nj_a, nl, bt_l)
    bt_l_on_m = np.interp(nj_m, nl, bt_l)
    bt_l_on_c = np.interp(nj_c, nl, bt_l)

    d_a = bt_ja - bt_l_on_a
    d_m = bt_jm - bt_l_on_m
    d_c = bt_jc - bt_l_on_c

    def stats(label, d):
        rms = np.sqrt((d**2).mean())
        i = np.argmax(np.abs(d))
        print(f"  {label:46s} RMS {rms:6.3f} K  worst {d[i]:+6.2f} K @ {nj_a[i]:.3f}")
        return rms, i

    print("=== 2355–2375 cm⁻¹, cont-OFF (Julia − LBLRTM) ===")
    rms_a, ia = stats("Julia Toon LIT + T_cg at p_cg  (baseline)", d_a)
    rms_m, im = stats("Julia Toon LIT + T_AVE         (T-mean swap)", d_m)
    rms_c, ic = stats("Julia CIM Pade + T_AVE         (CG-consistent)", d_c)
    print(f"\n  CIM/baseline RMS ratio = {rms_c/rms_a:.3f}")
    print(f"  CIM/baseline worst ratio= {abs(d_c[ic])/abs(d_a[ia]):.3f}")

    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    ax[0].plot(nl,   bt_l,  lw=0.5, color="k",        label="LBLRTM")
    ax[0].plot(nj_a, bt_ja, lw=0.4, color="tab:red",  alpha=0.7, label="Julia Toon+log-p (default)")
    ax[0].plot(nj_m, bt_jm, lw=0.4, color="tab:orange",alpha=0.7, label="Julia Toon+T_AVE")
    ax[0].plot(nj_c, bt_jc, lw=0.4, color="tab:blue", alpha=0.85, label="Julia CIM+T_AVE")
    ax[0].set_ylabel("BT [K]")
    ax[0].set_title("CO$_2$ ν₃ band head, 4.3 µm, cont-OFF — source-function test")
    ax[0].legend(loc="best", fontsize=8)

    ymax = max(abs(d_a).max(), abs(d_m).max(), abs(d_c).max()) * 1.05

    for axi, d, label, color, rms, ii in [
        (ax[1], d_a, "Julia (Toon + log-p T) − LBLRTM", "tab:red",    rms_a, ia),
        (ax[2], d_m, "Julia (Toon + T_AVE)   − LBLRTM", "tab:orange", rms_m, im),
        (ax[3], d_c, "Julia (CIM  + T_AVE)   − LBLRTM", "tab:blue",   rms_c, ic),
    ]:
        axi.plot(nj_a, d, lw=0.4, color=color)
        axi.axhline(0, color="grey", lw=0.5)
        axi.set_ylabel("ΔBT [K]")
        axi.set_title(f"{label}    RMS = {rms:.3f} K   worst {d[ii]:+.2f} K")
        axi.set_ylim(-ymax, ymax)
    ax[3].set_xlabel("wavenumber [cm⁻¹]")

    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_cim_vs_toon.png"
    plt.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
