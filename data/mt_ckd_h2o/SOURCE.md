# MT-CKD H₂O continuum — provenance and AER license

`mt_ckd43_h2o_coeffs.csv` holds the MT-CKD 4.3 water-vapour continuum reference
coefficients (self and foreign, `nu_cm1, Cs_cm2_molec_cm1, Cf_cm2_molec_cm1,
self_texp`) used by `h2o_continuum` in `src/CrossSections/continuum.jl`. The
table is **derived** from AER's MT_CKD distribution (the reference coefficients
in `absco-ref_wv-mt-ckd.nc` / `mt_ckd_h2o_module.f90`); see
`scripts/provenance/extract_ckd43_csv.py`.

This is third-party data. It is **not** covered by the package's MIT license and
carries AER's own terms, reproduced verbatim below as that license requires.

## AER copyright notice (reproduced per the license)

```
                 The MT_CKD Water Vapor Continuum

 Copyright ©, Atmospheric and Environmental Research, Inc., 2022

 All rights reserved. This source code was developed as part of the
 LBLRTM software and is designed for scientific and research purposes.
 Atmospheric and Environmental Research Inc. (AER) grants USER the right
 to download, install, use and copy this software for scientific and
 research purposes only. This software may be redistributed as long as
 this copyright notice is reproduced on any copy made and appropriate
 acknowledgment is given to AER. This software or any modified version
 of this software may not be incorporated into proprietary software or
 commercial software offered for sale without the express written
 consent of AER.

 This software is provided as is without any express or implied warranties.
```

Questions to AER: aer_contnm@aer.com

## Acknowledgment / citation

Continuum coefficients from AER's MT_CKD model. Please cite:

> Mlawer, E. J., Payne, V. H., Moncet, J.-L., Delamere, J. S., Alvarado, M. J.,
> & Tobin, D. C. (2012). *Development and recent evaluation of the MT_CKD model
> of continuum absorption.* Phil. Trans. R. Soc. A, 370, 2520–2556.
> https://doi.org/10.1098/rsta.2011.0295

Upstream: AER-RC/MT_CKD and AER-RC/LBLRTM (https://github.com/AER-RC).
