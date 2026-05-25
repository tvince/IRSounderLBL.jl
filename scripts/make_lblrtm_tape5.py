"""
Generate an LBLRTM TAPE5 control file from data/afgl_us_standard_50lev.csv.

Reproduces the same scenario the Julia `iasi_forward_model` call encodes for the
ARTS validation: AFGL US Standard 50-level profile, CO2 15 um window 645-800 cm-1,
nadir downlooking from TOA, blackbody surface, MT-CKD continuum ON, line coupling
OFF (pure Voigt LBL).

LBLRTM TAPE5 is column-exact fixed-format Fortran. Record formats are taken from
LBLRTM/docs/html/lblrtm_instructions.html (v12.17).

Key alignment choices:
  - MODEL=0 (user profile); IBMAX=nlev with our altitudes as layer boundaries
    (Record 3.3B) so LBLRTM uses the SAME 49 layers Julia does.
  - Molecule numbering is LBLRTM's fixed order: 5=CO, 6=CH4. The CSV stores
    CH4 before CO, so VMOL is reordered to [H2O, CO2, O3, N2O, CO, CH4].
  - VMR given as ppmv (JCHAR='A'); CSV stores dry-air volume fractions -> x1e6.
  - ILBLF4=1 (25 cm-1 line-by-line bound) to match Julia cutoff=25.

Usage:
  python scripts/make_lblrtm_tape5.py [--v1 645.0] [--v2 800.0] [-o data/lblrtm/TAPE5]
"""

import os, csv, argparse

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# LBLRTM molecule numbers we populate (1..6), in LBLRTM order.
# CSV column -> LBLRTM slot. Note CO(5) before CH4(6).
MOL_FROM_CSV = [
    ("vmr_H2O", "H2O"),   # 1
    ("vmr_CO2", "CO2"),   # 2
    ("vmr_O3",  "O3"),    # 3
    ("vmr_N2O", "N2O"),   # 4
    ("vmr_CO",  "CO"),    # 5  <- CO before CH4 (LBLRTM numbering)
    ("vmr_CH4", "CH4"),   # 6
]
NMOL = len(MOL_FROM_CSV)


# ── Fortran field formatters ──────────────────────────────────────────────────
def E10_3(x):
    """Fortran E10.3: 10-wide, e.g. ' 6.450E+02'."""
    s = f"{x:10.3E}"
    if len(s) != 10:
        raise ValueError(f"E10.3 overflow for {x!r}: {s!r} ({len(s)} chars)")
    return s


def F10_3(x):
    s = f"{x:10.3f}"
    if len(s) != 10:
        raise ValueError(f"F10.3 overflow for {x!r}: {s!r} ({len(s)} chars)")
    return s


def I5(n):
    return f"{int(n):5d}"


def place(buf, col1, text):
    """Place `text` so it ENDS at 1-indexed column col1 (right-justified field
    end). buf is a list of chars (0-indexed)."""
    end = col1                       # 1-indexed inclusive end
    start = end - len(text)          # 0-indexed start
    for i, ch in enumerate(text):
        buf[start + i] = ch


def record_1_2(flags):
    """Record 1.2: control flags, placed at exact end-columns per the spec.
    Cols: IHIRAC5 ILBLF410 ICNTNM15 IAERSL20 IEMIT25 ISCAN30 IFILTR35 IPLOT40
    ITEST45 IATM50 IMRG54-55 ILAS60 IOD65 IXSECT70 MPTS72-75 NPTS77-80
    ISOTPL85 IBRD90."""
    buf = [" "] * 90
    place(buf, 5,  str(flags["IHIRAC"]))
    place(buf, 10, str(flags["ILBLF4"]))
    place(buf, 15, str(flags["ICNTNM"]))
    place(buf, 20, str(flags["IAERSL"]))
    place(buf, 25, str(flags["IEMIT"]))
    place(buf, 30, str(flags["ISCAN"]))
    place(buf, 35, str(flags["IFILTR"]))
    place(buf, 40, str(flags["IPLOT"]))
    place(buf, 45, str(flags["ITEST"]))
    place(buf, 50, str(flags["IATM"]))
    place(buf, 55, f"{flags['IMRG']:>2}")    # 3X,A2 -> 2 chars ending col 55
    place(buf, 60, str(flags["ILAS"]))
    place(buf, 65, str(flags["IOD"]))
    place(buf, 70, str(flags["IXSECT"]))
    place(buf, 75, f"{flags['MPTS']:4d}")    # 1X,I4 ending col 75
    place(buf, 80, f"{flags['NPTS']:4d}")    # 1X,I4 ending col 80
    place(buf, 85, str(flags["ISOTPL"]))
    place(buf, 90, str(flags["IBRD"]))
    return "".join(buf).rstrip()


def build_tape5(profile, v1, v2, title, angle=180.0):
    p_mb  = profile["p_hPa"]
    T_K   = profile["T_K"]
    z_km  = profile["z_km"]
    vmr   = profile["vmr"]            # dict: csv_col -> list (fractions)
    nlev  = len(z_km)
    z_top = z_km[-1]
    T_sfc = T_K[0]

    lines = []

    # Record 1.1 — title, must start with '$'
    lines.append(("$ " + title)[:80])

    # Record 1.2 — control flags
    flags = dict(
        IHIRAC=1,    # Voigt
        ILBLF4=1,    # line-by-line bound 25 cm-1 (matches Julia cutoff)
        ICNTNM=1,    # all continua (MT-CKD); Rayleigh negligible in the IR
        IAERSL=0,
        IEMIT=1,     # radiance + transmittance
        ISCAN=0,     # no instrument scan (no ILS), like with_ils=false
        IFILTR=0,
        IPLOT=0,
        ITEST=0,
        IATM=1,      # use LBLATM
        IMRG=0,
        ILAS=0,
        IOD=0,
        IXSECT=0,    # no cross-section molecules
        MPTS=0, NPTS=0, ISOTPL=0, IBRD=0,
    )
    lines.append(record_1_2(flags))

    # Record 1.3 — V1, V2, SAMPLE, then defaults (zeros)
    lines.append(E10_3(v1) + E10_3(v2) + E10_3(4.0)
                 + E10_3(0.0) * 5)   # DVSET, ALFAL0, AVMASS, DPTMIN, DPTFAC

    # Record 1.4 — surface boundary: TBOUND, SREMIS(1-3), SRREFL(1-3)
    # emissivity 1, reflectivity 0 (blackbody)
    lines.append(E10_3(T_sfc) + E10_3(1.0) + E10_3(0.0) * 2
                 + E10_3(0.0) * 3)

    # Record 3.1 — atmosphere control
    #  MODEL=0, ITYPE=2 (H1->H2), IBMAX=nlev (layers from our boundaries),
    #  ZERO=0, NOPRNT=1, NMOL, IPUNCH=0; IFXTYP(I2)+1X+MUNITS(I2); RE/HSPACE/VBAR;
    #  10X; REF_LAT
    rec31 = (I5(0) + I5(2) + I5(nlev) + I5(0) + I5(1) + I5(NMOL) + I5(0)
             + f"{0:2d}" + " " + f"{0:2d}"
             + F10_3(0.0) + F10_3(z_top) + F10_3(0.0)
             + " " * 10 + F10_3(0.0))
    lines.append(rec31)

    # Record 3.2 — geometry: H1=TOA, H2=surface, ANGLE (zenith at H1; nadir down)
    lines.append(F10_3(z_top) + F10_3(0.0) + F10_3(angle))

    # Record 3.3B — layer boundary altitudes (IBMAX>0), 8F10.3 per line
    for i in range(0, nlev, 8):
        lines.append("".join(F10_3(z) for z in z_km[i:i + 8]))

    # Record 3.4 — IMMAX, HMOD
    lines.append(I5(nlev) + f"{'AFGL US Std 50lev':<24}")

    # Records 3.5 / 3.6 — per level
    jchar = "AAAAAA"[:NMOL]           # all VMR in ppmv
    for k in range(nlev):
        # 3.5: ZM PM TM | 5X JCHARP JCHART 1X JLONG 1X JCHAR(1..NMOL)
        rec35 = (E10_3(z_km[k]) + E10_3(p_mb[k]) + E10_3(T_K[k])
                 + " " * 5 + "A" + "A" + " " + " " + " " + jchar)
        lines.append(rec35)
        # 3.6: VMOL(1..NMOL) in ppmv, 8E10.3
        ppmv = [vmr[col][k] * 1.0e6 for col, _ in MOL_FROM_CSV]
        lines.append("".join(E10_3(v) for v in ppmv))

    # Terminate the input stream (CXID(1)='%')
    lines.append("%")

    return "\n".join(lines) + "\n"


def load_profile(path):
    cols = ["p_hPa", "T_K", "z_km"] + [c for c, _ in MOL_FROM_CSV]
    data = {c: [] for c in cols}
    with open(path) as f:
        for row in csv.DictReader(f):
            for c in cols:
                data[c].append(float(row[c]))
    return dict(
        p_hPa=data["p_hPa"], T_K=data["T_K"], z_km=data["z_km"],
        vmr={c: data[c] for c, _ in MOL_FROM_CSV},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=float, default=645.0)
    ap.add_argument("--v2", type=float, default=800.0)
    ap.add_argument("--angle", type=float, default=180.0,
                    help="zenith ANGLE at H1 (Record 3.2); 180=nadir-down, 0=nadir-up")
    ap.add_argument("--profile", default=os.path.join(DATA_DIR, "afgl_us_standard_50lev.csv"))
    ap.add_argument("-o", "--out", default=os.path.join(DATA_DIR, "lblrtm", "TAPE5"))
    ap.add_argument("--title", default="IRSounderLBL validation: AFGL US Std, CO2 15um, no LM")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    text = build_tape5(prof, args.v1, args.v2, args.title, angle=args.angle)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)

    print(f"Wrote TAPE5 -> {args.out}")
    print(f"  range {args.v1}-{args.v2} cm-1, {len(prof['z_km'])} levels, "
          f"{NMOL} molecules, T_sfc={prof['T_K'][0]:.1f} K")


if __name__ == "__main__":
    main()
