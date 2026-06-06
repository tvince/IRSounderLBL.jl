"""
Extract LBLRTM per-layer Curtis-Godson properties from TAPE6.

Two tables of interest:

(1) "LAYER BOUNDARIES PBAR TBAR ... MOLECULAR MIXING RATIOS BY LAYER"
    Each layer row:
       L  z_from  z_to  IY IT IH IP  PBAR(MB)  TBAR(K)  AIR  H2O  CO2  O3  N2O  CO  CH4  OTHER

(2) "MOLECULAR AMOUNTS (MOL/CM**2) BY LAYER"
    Each layer row:
       L  z_from TO  z_to KM  P(MB)  T(K)  IPATH  H2O  CO2  O3  N2O  CO  CH4  O2  OTHER

(3) "LAYER ... P(MB) T(K) ALPHL ALPHD ALPHV ZETA  CALC DV ..."  (reference line widths)
    Each layer row:
       L  z_from TO  z_to KM  P(MB)  T(K)  ALPHL  ALPHD  ALPHV  ZETA  CALC DV  ...

We unify the three by layer index L. Outputs CSV:
  layer, z_from_km, z_to_km, p_eff_hPa, T_eff_K, N_CO2_molec_cm2,
  alpha_L_ref, alpha_D_ref, alpha_V_ref

The ALPHL/ALPHD/ALPHV in (3) are the LBLRTM reference half-widths at the layer's
(P_eff, T_eff) for a representative line of width ALFAL0 (default 0.04 cm-1/atm,
typical CO2-ish). They directly show what LBLRTM ASSUMES the broadening is at
that layer.

Usage:
  python scripts/parse_lblrtm_tape6_layers.py \\
      --tape6 data/lblrtm/lblrtm_run_43um/TAPE6 \\
      --out   data/lblrtm/lblrtm_layers_43um.csv
"""

import argparse
import re
import csv


def parse_table1(lines, start, n_lay=49):
    """PBAR/TBAR/mixing-ratio table starting at `start` (0-indexed).
    Returns dict[L] -> {z_from, z_to, p_eff, T_eff, vmr_CO2, n_air_col}."""
    out = {}
    for ln in lines[start:start + n_lay + 5]:
        m = re.match(
            r"^0?\s*(\d{1,2})\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+\s+"
            r"([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+"
            r"([\d.eE+\-]+)\s+[\d.eE+\-]+\s+([\d.eE+\-]+)",
            ln)
        if not m:
            continue
        L = int(m.group(1))
        if L < 1 or L > n_lay:
            continue
        out[L] = {
            "z_from_km": float(m.group(2)),
            "z_to_km":   float(m.group(3)),
            "p_eff_hPa": float(m.group(4)),
            "T_eff_K":   float(m.group(5)),
            "n_air_col": float(m.group(6)),
            "vmr_CO2":   float(m.group(7)),
        }
    return out


def parse_table2(lines, start, n_lay=49):
    """MOLECULAR AMOUNTS table (column densities). Header is
        L  z_from TO z_to KM  P(MB)  T(K)  IPATH  H2O CO2 O3 N2O CO CH4 O2 OTHER
    Returns dict[L] -> N_CO2 column density (mol/cm2)."""
    out = {}
    for ln in lines[start:start + n_lay + 5]:
        m = re.match(
            r"^0?\s*(\d{1,2})\s*(\d+\.\d+)\s*TO\s*(\d+\.\d+)\s+KM\s+"
            r"([\d.eE+\-]+)\s+([\d.]+)\s+\d+\s+"
            r"[\d.eE+\-]+\s+([\d.eE+\-]+)",
            ln)
        if not m:
            continue
        L = int(m.group(1))
        if L < 1 or L > n_lay:
            continue
        out[L] = float(m.group(6))
    return out


def parse_table3(lines, start, n_lay=49):
    """ALPHL/ALPHD/ALPHV table. Header is
        L  z_from TO z_to KM  P(MB)  T(K)  ALPHL  ALPHD  ALPHV  ZETA  CALC DV ...
    Returns dict[L] -> (alpha_L, alpha_D, alpha_V, calc_dv)."""
    out = {}
    for ln in lines[start:start + n_lay + 5]:
        # Match the per-line width table — layer ID followed by "z_from TO z_to KM"
        m = re.match(
            r"^0?\s*(\d{1,2})\s*(\d+\.\d+)\s*TO\s*(\d+\.\d+)\s+KM\s+"
            r"([\d.eE+\-]+)\s+([\d.]+)\s+"
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            ln)
        if not m:
            continue
        L = int(m.group(1))
        if L < 1 or L > n_lay:
            continue
        out[L] = {
            "alpha_L":   float(m.group(6)),
            "alpha_D":   float(m.group(7)),
            "alpha_V":   float(m.group(8)),
            "zeta":      float(m.group(9)),
            "calc_dv":   float(m.group(10)),
        }
    return out


def find_sections(lines):
    """Return (line index of header) for each of the three tables we want."""
    i_t1 = i_t2 = i_t3 = None
    for i, ln in enumerate(lines):
        if "PBAR" in ln and "TBAR" in ln and "MIXING RATIOS BY LAYER" in ln:
            i_t1 = i + 2     # skip header and "FROM TO" line
        elif "MOLECULAR AMOUNTS (MOL/CM**2) BY LAYER" in ln:
            i_t2 = i + 2     # skip the column-name line that follows
        elif "ALPHL" in ln and "ALPHD" in ln and "ALPHV" in ln:
            i_t3 = i + 2
    return i_t1, i_t2, i_t3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape6", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--nlay",  type=int, default=49)
    args = ap.parse_args()

    with open(args.tape6) as f:
        lines = f.readlines()

    i_t1, i_t2, i_t3 = find_sections(lines)
    print(f"section starts (1-indexed):  PBAR/TBAR table @ {i_t1+1 if i_t1 else None}, "
          f"column amounts @ {i_t2+1 if i_t2 else None}, "
          f"ALPHL/D/V @ {i_t3+1 if i_t3 else None}")

    t1 = parse_table1(lines, i_t1, args.nlay) if i_t1 else {}
    t2 = parse_table2(lines, i_t2, args.nlay) if i_t2 else {}
    t3 = parse_table3(lines, i_t3, args.nlay) if i_t3 else {}
    print(f"layers parsed:  PBAR/TBAR {len(t1)}, column amounts {len(t2)}, "
          f"ALPHL/D/V {len(t3)}")

    rows = []
    for L in sorted(t1.keys()):
        r = dict(layer=L,
                 z_from_km   = t1[L]["z_from_km"],
                 z_to_km     = t1[L]["z_to_km"],
                 p_eff_hPa   = t1[L]["p_eff_hPa"],
                 T_eff_K     = t1[L]["T_eff_K"],
                 vmr_CO2     = t1[L]["vmr_CO2"],
                 N_CO2_cm2   = t2.get(L, float("nan")),
                 alpha_L_ref = t3.get(L, {}).get("alpha_L", float("nan")),
                 alpha_D_ref = t3.get(L, {}).get("alpha_D", float("nan")),
                 alpha_V_ref = t3.get(L, {}).get("alpha_V", float("nan")),
                 calc_dv     = t3.get(L, {}).get("calc_dv", float("nan")))
        rows.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
