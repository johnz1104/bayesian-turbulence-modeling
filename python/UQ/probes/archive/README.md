Archived diagnostic probe scripts from the 2026-07-20 coupled-round
diagnosis (the runaway-to-vacuum investigation and its exoneration chain:
mask, prepare-properties, mask floor, force ramp, frozen-k). They document
the rounds exactly as run and target the PRE-REPAIR injection interface
(set_target_correction(b, dq, ...)); the repaired stored-discrepancy
interface is set_target_correction(db, b, dq, ...), so these scripts are
provenance, not runnable tools. Their raw logs live in
UQ-RANS_research/shock_interaction/diagnosis_logs/. The maintained
three-solve pilot is python/UQ/sbli_member_pilot.py.
