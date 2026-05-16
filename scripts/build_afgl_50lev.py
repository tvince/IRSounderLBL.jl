"""
Download AFGL US Standard Atmosphere (50 levels, 0-120 km) from the ARTS XML
data repository and write data/afgl_us_standard_50lev.csv.

Source: https://gitlab.rrz.uni-hamburg.de/atmtools/arts-xml-data/-/tree/main/planets/Earth/afgl/us-standard

Columns: p_hPa, T_K, z_km, vmr_H2O, vmr_CO2, vmr_O3, vmr_N2O, vmr_CH4, vmr_CO
All values are surface-first (index 0 = surface, index 49 = 120 km).
"""

import urllib.request, re, csv, os

BASE = ("https://gitlab.rrz.uni-hamburg.de/atmtools/arts-xml-data/-/raw/main"
        "/planets/Earth/afgl/us-standard")

def fetch_tensor(name):
    """Download an ARTS XML file and return the Tensor3 / Vector data as a list of floats."""
    url = f"{BASE}/{name}"
    print(f"  Fetching {name} ...")
    with urllib.request.urlopen(url) as r:
        txt = r.read().decode()
    # Extract content of the last Tensor3 or Vector block
    m = re.findall(r'<(?:Tensor3|Vector)[^>]*>(.*?)</(?:Tensor3|Vector)>',
                   txt, re.DOTALL)
    if not m:
        raise ValueError(f"No Tensor3/Vector found in {name}")
    return [float(x) for x in m[-1].split()]

print("Downloading AFGL US Standard Atmosphere (50 levels)...")
z_m  = fetch_tensor("p.xml")[: 0]          # altitude grid is first Vector
# Re-fetch to get altitude separately
with urllib.request.urlopen(f"{BASE}/p.xml") as r:
    txt_p = r.read().decode()
vectors = re.findall(r'<Vector[^>]*>(.*?)</Vector>', txt_p, re.DOTALL)
z_m   = [float(x) for x in vectors[0].split()]   # altitude grid (m)
p_Pa  = fetch_tensor("p.xml")                      # pressure (Pa)
T_K   = fetch_tensor("t.xml")                      # temperature (K)
h2o   = fetch_tensor("H2O.xml")
co2   = fetch_tensor("CO2.xml")
o3    = fetch_tensor("O3.xml")
n2o   = fetch_tensor("N2O.xml")
ch4   = fetch_tensor("CH4.xml")
co    = fetch_tensor("CO.xml")

n = len(p_Pa)
assert all(len(v) == n for v in [T_K, h2o, co2, o3, n2o, ch4, co, z_m]), \
    "Length mismatch between profiles"
print(f"  {n} levels, z = {z_m[0]/1e3:.0f} – {z_m[-1]/1e3:.0f} km")
print(f"  p = {p_Pa[0]/100:.1f} – {p_Pa[-1]/100:.5f} hPa")
print(f"  T = {min(T_K):.1f} – {max(T_K):.1f} K")

# Scale CO2 from original 330 ppm to current 415 ppm (preserve upper-atm shape)
co2_ref = co2[0]          # surface value in original profile
co2_new = 4.15e-4         # target surface VMR (415 ppm)
scale   = co2_new / co2_ref
co2_scaled = [v * scale for v in co2]
print(f"  CO2 scaled from {co2_ref*1e6:.0f} ppm to {co2_new*1e6:.0f} ppm")

outpath = os.path.join("data", "afgl_us_standard_50lev.csv")
with open(outpath, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p_hPa", "T_K", "z_km",
                "vmr_H2O", "vmr_CO2", "vmr_O3", "vmr_N2O", "vmr_CH4", "vmr_CO"])
    for i in range(n):
        w.writerow([
            round(p_Pa[i] / 100.0, 6),   # Pa → hPa
            round(T_K[i], 4),
            round(z_m[i] / 1000.0, 3),   # m → km
            f"{h2o[i]:.6e}",
            f"{co2_scaled[i]:.6e}",
            f"{o3[i]:.6e}",
            f"{n2o[i]:.6e}",
            f"{ch4[i]:.6e}",
            f"{co[i]:.6e}",
        ])

print(f"Saved → {outpath}  ({n} levels)")
