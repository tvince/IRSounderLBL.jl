"""
Render the README figure: a simulated IASI brightness-temperature spectrum
(AFGL US Standard, nadir) with the CO2, O3, CH4/N2O and H2O bands labelled.
Writes docs/assets/spectrum-light.png and docs/assets/spectrum-dark.png, which
are committed and served by the <picture> element at the top of README.md.

Inputs are two validation exports, neither of which is committed (data/ is
gitignored apart from the small redistributable tables):

  data/julia_bt_645_800.csv   this package, full 645-2760 cm-1 line-by-line run
                              -> scripts/validation/julia_bt_export.jl
  data/arts_bt_iasi.csv       the matching ARTS 2.6 reference run
                              -> scripts/arts_validation.py on the
                                 `validation-scripts` branch

So this script documents how the committed figure was produced rather than
being runnable from a fresh clone. Regenerate the two CSVs first if you need to
rebuild it. The RMS quoted in the subtitle is computed here from those files, so
it cannot drift away from the figure it labels.

Colours are Solarized (Ethan Schoonover): base3/base03 surface, base01/base1
headings, base00/base0 ticks, blue #268bd2 for the trace, and a single neutral
tone for every band highlight. The bands are NOT given a hue each: Solarized's
eight accents are near-equiluminant by design, and no four of them clear the
CVD separation gates used for categorical palettes (best case dE 5.5 against a
hard floor of 6). Each band carries a direct text label, so colour is not
carrying identity and one tone loses nothing.

The plotted range stops at 2390 cm-1. Above roughly 2400 both this package and
ARTS produce a flat ~288 K line, since no species in this configuration absorbs
there; it is a real result but reads as a broken axis in a figure.

    python3 scripts/provenance/make_readme_figure.py
"""

import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NU_HI = 2390.0

def load(p):
    nu, bt = [], []
    with open(f"{ROOT}/{p}") as f:
        for r in csv.DictReader(f):
            nu.append(float(r["nu_cm1"])); bt.append(float(r["BT_K"]))
    return np.array(nu), np.array(bt)

nu, bt   = load("data/julia_bt_645_800.csv")
nu_a, ba = load("data/arts_bt_iasi.csv")
assert np.allclose(nu, nu_a)
rms_full = math.sqrt(np.mean((bt - ba) ** 2))
m = nu <= NU_HI

BANDS = [(645, 800,  "CO$_2$ $\\nu_2$\n15 µm"),
         (980, 1080, "O$_3$ $\\nu_3$\n9.6 µm"),
         (1200,1340, "CH$_4$, N$_2$O"),
         (1360,1750, "H$_2$O $\\nu_2$\n6.3 µm"),
         (2200,2390, "CO$_2$ $\\nu_3$\n4.3 µm")]

SOLARIZED = {
  "light": dict(surface="#fdf6e3", shade="#93a1a1", shade_a=0.13, rule="#93a1a1", grid="#93a1a1",
                spine="#93a1a1", tick="#657b83", head="#586e75",
                muted="#657b83", line="#268bd2"),
  "dark":  dict(surface="#002b36", shade="#93a1a1", shade_a=0.10, rule="#586e75", grid="#586e75",
                spine="#586e75", tick="#839496", head="#93a1a1",
                muted="#839496", line="#268bd2"),
}

def render(theme, out):
    c = SOLARIZED[theme]
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=200)
    fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["spine"]); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=c["tick"], labelsize=9.5, width=0.8)
    ax.grid(True, color=c["grid"], lw=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    for lo, hi, lab in BANDS:
        ax.axvspan(lo, hi, color=c["shade"], alpha=c["shade_a"], lw=0, zorder=0)
        ax.plot([lo, hi], [1.0, 1.0], transform=ax.get_xaxis_transform(),
                color=c["rule"], lw=1.4, solid_capstyle="butt", zorder=4, clip_on=False)
        ax.text((lo + hi) / 2, 1.015, lab, ha="center", va="bottom", fontsize=8.5,
                color=c["tick"], linespacing=1.25,
                transform=ax.get_xaxis_transform())
    for x in (890, 1140):
        ax.text(x, 199, "window", ha="center", va="center", fontsize=8.5,
                color=c["muted"], style="italic")

    ax.plot(nu[m], bt[m], lw=0.7, color=c["line"], solid_joinstyle="round", zorder=3)
    ax.set_xlim(645, NU_HI); ax.set_ylim(190, 296)
    ax.set_yticks([200, 225, 250, 275])
    ax.xaxis.set_major_locator(MultipleLocator(250))
    ax.set_xlabel("Wavenumber  (cm$^{-1}$)", color=c["tick"], fontsize=10.5)
    ax.set_ylabel("Brightness temperature  (K)", color=c["tick"], fontsize=10.5)
    ax.set_title("Line-by-line IASI spectrum — AFGL US Standard, nadir",
                 color=c["head"], fontsize=14, pad=58, loc="left", fontweight="medium")
    ax.text(0, 1.135, f"agrees with ARTS 2.6 to {rms_full:.3f} K RMS over the full "
                      f"645–2760 cm$^{{-1}}$ range", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.5, color=c["muted"])

    fig.savefig(out, facecolor=c["surface"], bbox_inches="tight", pad_inches=0.28)
    plt.close(fig); print("wrote", out)

print(f"full-range RMS vs ARTS = {rms_full:.4f} K; plotting {int(m.sum()):,} of {len(nu):,} channels")
render("light", f"{ROOT}/docs/assets/spectrum-light.png")
render("dark",  f"{ROOT}/docs/assets/spectrum-dark.png")
