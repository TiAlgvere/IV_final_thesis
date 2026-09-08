# Diagnostic Matrix - Fault -> Signature

Baseline (healthy/clean): tan-delta 2.00e-03, volume loss current 2.51e-04 A,
surface leakage 0. d|Vtap| and d-theta are vs that baseline. Split leakage:
surface_leak = creepage surface-conductance current; volume_leak = bulk loss current.

| family | case | d|Vtap| [%] | d-theta [mdeg] | tan-delta | surface leak [A] | volume leak [A] | diagnosis |
|--------|------|-------------|----------------|-----------|------------------|-----------------|-----------|
| baseline | healthy | +0.00 | +0.0 | 2.00e-03 | 0.00e+00 | 2.51e-04 | healthy / nominal |
| internal | C1_cap | +16.13 | -0.1 | 2.00e-03 | 0.00e+00 | 2.91e-04 | capacitor ratio fault (C1 cap up / disc short) |
| internal | C1_loss | +0.12 | -2289.8 | 4.18e-02 | 0.00e+00 | 5.24e-03 | internal dielectric loss (C1) |
| internal | C2_cap | -14.28 | +0.1 | 2.00e-03 | 0.00e+00 | 2.58e-04 | capacitor ratio fault (C2 cap up / disc short) |
| internal | C2_loss | -0.09 | +2289.2 | 9.94e-03 | 0.00e+00 | 1.25e-03 | internal dielectric loss (C2) |
| internal | disc_short | +9.09 | -0.5 | 2.01e-03 | 0.00e+00 | 2.75e-04 | capacitor ratio fault (C1 cap up / disc short) |
| pollution | bottom25 | -0.21 | +2.5 | 2.01e-03 | 1.69e-06 | 2.51e-04 | localized pollution patch (position by phase sign) |
| pollution | full | +0.06 | -2.1 | 1.63e-01 | 2.01e-02 | 2.51e-04 | external creepage pollution / bridge |
| pollution | mid25 | +0.02 | -0.1 | 2.00e-03 | 2.40e-07 | 2.51e-04 | healthy / nominal |
| pollution | streamer | +0.06 | -0.3 | 8.95e-01 | 1.12e-01 | 2.51e-04 | external creepage pollution / bridge |
| pollution | top25 | +0.08 | -0.8 | 2.02e-03 | 2.54e-06 | 2.51e-04 | localized pollution patch (position by phase sign) |

## Classifier (from the split signature)
1. **|d|Vtap|| > 3%**  -> capacitor ratio fault / disc short  (Vtap up = C1 side, down = C2).
2. **surface leakage high** (>> baseline volume loss) -> external creepage pollution / bridge.
3. **tan-delta up + volume loss up + degree-scale phase** -> internal dielectric loss
   (phase sign: negative = C1 / above tap, positive = C2 / below tap).
4. **phase shift with low tan-delta and ~zero surface leakage** -> localized pollution patch
   (position from the phase sign).

The split leakage is the key new axis: it cleanly separates the EXTERNAL path
(surface leakage, pollution) from the INTERNAL path (volume loss, dielectric aging) -
previously both collapsed into one 'leakage' number. See `diagnostic_matrix.png`.
