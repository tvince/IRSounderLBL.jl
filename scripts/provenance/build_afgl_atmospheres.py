"""
Download the six AFGL (1986) model atmospheres — tropical, midlatitude summer/winter,
subarctic summer/winter, and US standard — from the ARTS XML data repository and write
data/afgl_<name>_50lev.csv, one per atmosphere. Generalizes build_afgl_50lev.py (which
produced only us-standard) to the full set; the us-standard output is byte-for-byte the
same as before.

Source: https://gitlab.rrz.uni-hamburg.de/atmtools/arts-xml-data/-/tree/main/planets/Earth/afgl
Each atmosphere is on its OWN native 50-level grid (shared AFGL altitude levels, 0-120 km;
pressure differs per atmosphere because T differs) — no interpolation, each profile is
self-consistent and directly usable as a retrieval prior.

Columns: p_hPa, T_K, z_km, vmr_H2O, vmr_CO2, vmr_O3, vmr_N2O, vmr_CH4, vmr_CO
Surface-first (index 0 = surface). CO2 scaled to 415 ppm surface (as the original did);
the retrieval re-scales CO2 to its own ppm anyway.

    python3 scripts/build_afgl_atmospheres.py
"""

import urllib.request, re, csv, os

REPO = ("https://gitlab.rrz.uni-hamburg.de/atmtools/arts-xml-data/-/raw/main"
        "/planets/Earth/afgl")
# repo directory name -> our CSV slug
ATMOSPHERES = {
    "us-standard":        "us_standard",
    "tropical":           "tropical",
    "midlatitude-summer": "midlatitude_summer",
    "midlatitude-winter": "midlatitude_winter",
    "subarctic-summer":   "subarctic_summer",
    "subarctic-winter":   "subarctic_winter",
}
CO2_SURFACE_PPM = 415.0


def _blocks(txt):
    return re.findall(r'<(?:Tensor3|Vector)[^>]*>(.*?)</(?:Tensor3|Vector)>', txt, re.DOTALL)


def fetch(base, name):
    with urllib.request.urlopen(f"{base}/{name}") as r:
        txt = r.read().decode()
    m = _blocks(txt)
    if not m:
        raise ValueError(f"No Tensor3/Vector in {name}")
    return txt, [float(x) for x in m[-1].split()]


def build(dirname, slug):
    base = f"{REPO}/{dirname}"
    print(f"\n{dirname} -> data/afgl_{slug}_50lev.csv")
    txt_p, p_Pa = fetch(base, "p.xml")
    z_m = [float(x) for x in _blocks(txt_p)[0].split()]   # altitude grid (m), first block
    _, T_K = fetch(base, "t.xml")
    prof = {sp: fetch(base, f"{sp}.xml")[1] for sp in ("H2O", "CO2", "O3", "N2O", "CH4", "CO")}

    n = len(p_Pa)
    assert all(len(v) == n for v in [T_K, z_m, *prof.values()]), f"length mismatch in {dirname}"
    scale = (CO2_SURFACE_PPM * 1e-6) / prof["CO2"][0]
    co2 = [v * scale for v in prof["CO2"]]
    print(f"  {n} levels · p {p_Pa[0]/100:.1f}-{p_Pa[-1]/100:.4g} hPa · "
          f"T_sfc {T_K[0]:.1f} K · O3 surf {prof['O3'][0]:.2e}")

    outpath = os.path.join("data", f"afgl_{slug}_50lev.csv")
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_hPa", "T_K", "z_km",
                    "vmr_H2O", "vmr_CO2", "vmr_O3", "vmr_N2O", "vmr_CH4", "vmr_CO"])
        for i in range(n):
            w.writerow([round(p_Pa[i]/100.0, 6), round(T_K[i], 4), round(z_m[i]/1000.0, 3),
                        f"{prof['H2O'][i]:.6e}", f"{co2[i]:.6e}", f"{prof['O3'][i]:.6e}",
                        f"{prof['N2O'][i]:.6e}", f"{prof['CH4'][i]:.6e}", f"{prof['CO'][i]:.6e}"])
    return outpath


if __name__ == "__main__":
    for dirname, slug in ATMOSPHERES.items():
        build(dirname, slug)
    print("\nDone.")
