# Task 016 - Capacitor-Stack (Internal) Fault Signatures

Rounded DDB-123, healthy |Vtap| = 11829.5 V, tap phase -0.12 mdeg. C1 = elements
1..10 (HV section), C2 = elements 11..12 (LV).
Cap faults +20% eps_r; loss faults tan-delta -> 0.05;
disc short = element 5 (C1) shorted (sigma=1 S/m). d-theta vs healthy.

| case | |Vtap| [V] | dVtap [%] | d-theta [mdeg] | tan(delta) | leakage [A] |
|------|-----------|-----------|----------------|------------|-------------|
| healthy | 11829.5 | +0.00 | +0.00 | 2.000e-03 | 2.505e-04 |
| C1_cap | 13737.8 | +16.13 | -0.06 | 2.002e-03 | 2.911e-04 |
| C2_cap | 10140.3 | -14.28 | +0.06 | 2.000e-03 | 2.577e-04 |
| C1_loss | 11843.7 | +0.12 | -2289.79 | 4.182e-02 | 5.242e-03 |
| C2_loss | 11819.1 | -0.09 | +2289.24 | 9.941e-03 | 1.246e-03 |
| disc_short | 12905.0 | +9.09 | -0.54 | 2.010e-03 | 2.746e-04 |

## Fault separation (internal family vs pollution family)
- **Capacitance faults (C1/C2 cap, disc short)** move |Vtap| (the divider ratio) with
  little tan-delta or leakage: C1-cap / C1-short raise Vtap, C2-cap lowers it.
- **Loss faults (C1/C2 loss)** raise tan-delta and shift phase with ~no Vtap change;
  C1-loss vs C2-loss differ in phase (loss above vs below the tap).
- **External pollution** (Task 015) raises tan-delta + leakage with little Vtap change;
  it lives on its own creepage manifold in (loss, phase).
- Key separators: **|Vtap| shift -> capacitive/ratio faults**; **leakage with tan-delta
  -> external pollution**; **tan-delta with ~zero leakage and ~zero Vtap -> internal loss**.
  See `stack_faults_state_space.png` (left: loss-phase; right: Vtap-loss).

## Artifacts
- `stack_faults.csv`, `stack_faults_state_space.png`
