# `docs/` — development records

**This is not the user documentation.** For usage, start with the top-level
[`README.md`](../README.md), the runnable examples in [`scripts/`](../scripts),
and the docstrings (`?forward_model`, `?analytic_jacobian`, `?optimal_estimation`,
… at the REPL). A
Documenter site is planned; until it exists, the docstrings are the API
reference.

What is kept here is the paper trail behind the physics — why the model is
believed to be correct, and what was checked to establish that:

| File | What it is |
| --- | --- |
| `jacobian_roadmap.md` | The design the analytic-Jacobian and optimal-estimation work was built from. Implemented; kept as a record of the derivations. |
| `ARTS_BUG_REPORT.md` | Diagnosis of wrong-sign/inflated CO₂ line-mixing Y coefficients near 665 cm⁻¹ in ARTS (upstream issue #1130, since fixed). Retained as validation evidence. |
| `ARTS_CIA_BUG_REPORT.md` | Diagnosis of the collision-induced-absorption discrepancies found while validating against ARTS. Never filed upstream; retained as validation evidence. |
| `code_review_2026-07-02.md` | A full-source review from 2026-07-02, focused on the Jacobian and line-mixing hot paths. A snapshot of that date, not a current description of the code. |
| `assets/` | Figures used by the top-level README. |

The two ARTS reports document discrepancies against a specific ARTS version and
are frozen at the date they were written; they are provenance for the validation
figures quoted in the main README, not maintained documentation.
