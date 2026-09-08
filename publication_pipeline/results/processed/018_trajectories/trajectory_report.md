# Task 018 - State-Space Trajectory & Degradation-Vector Formulation

## 1. State point
Observable state vector (relative to the healthy/clean baseline x0):

    x = [ d|Vtap| (%),  dθ (mdeg),  tanδ,  surface_leak (A),  volume_leak (A) ]

Baseline x0: tanδ0 = 2.000e-03, volume_leak0 = 2.505e-04 A, surface_leak0 = 0,
d|Vtap| = dθ = 0. A fault moves the device to a point x in this 5-D space.

## 2. Fault-family manifolds  x = M_f(s)
Each family is a 1-parameter curve through x0, parameterised by severity s:
- pollution  M_poll(log10 σ_s): full-creepage surface conductance (Task 10/15)
- C1/C2 cap  M_cap(eps factor): capacitor permittivity (Task 16)
- C1/C2 loss M_loss(tanδ): element dielectric loss (Task 16)
- disc short M_short(n): n shorted C1 elements (Task 16)

## 3. Pollution trajectory & velocity (per decade of σ_s)
Velocity v = dx/d(log10 σ_s) in the (log10 tanδ, dθ) projection:

| σ_s decade | d(log10 tanδ)/dec | d(dθ)/dec [mdeg] |
|---|---|---|
| 1e-09->1e-08 | +0.21 | -4.4 |
| 1e-08->1e-07 | +0.70 | -11.9 |
| 1e-07->1e-06 | +0.95 | +14.6 |
| 1e-06->1e-05 | +1.00 | +1.9 |
| 1e-05->1e-04 | +1.00 | +0.2 |

tanδ and surface_leak rise ~one decade per decade of σ_s (slope ~ +1); dθ first
dives (capacitive-polluting branch) then recovers (resistive-bridge branch) - the
critical point dθ' = 0 is the relaxation corner. See pollution_trajectory_arrows.png.

## 4. Internal-fault degradation vectors  v = dx/ds
Direction (unit vector in detection-normalised coordinates; dominant observable):

| fault | d|Vtap| | dθ | tanδ | surf | vol | dominant |
|---|---|---|---|---|---|---|
| C1_cap | +1.00 | -0.00 | +0.00 | +0.00 | +0.01 | dvtap_pct |
| C2_cap | -1.00 | +0.00 | +0.00 | +0.00 | +0.00 | dvtap_pct |
| C1_loss | +0.00 | -0.96 | +0.17 | +0.00 | +0.21 | dtheta_mdeg |
| C2_loss | -0.00 | +1.00 | +0.03 | +0.00 | +0.04 | dtheta_mdeg |
| disc_short | +1.00 | -0.00 | +0.00 | +0.00 | +0.01 | dvtap_pct |

Cap/short vectors point almost purely along d|Vtap|; loss vectors along (dθ, tanδ,
volume_leak) with dθ sign = section; pollution points along (surface_leak, tanδ).
The families are near-orthogonal in detection-normalised space -> separable.

## 5. Classification rule
Given measured dx, in order:
1. |d|Vtap|| > 3%  -> capacitor ratio fault  (+ = C1 / short, − = C2)
2. surface_leak ≫ baseline  -> external creepage pollution (bridge);
   phase-only with ~0 leak -> localized pollution patch (sign = position)
3. tanδ↑ + volume_leak↑ + degree-scale dθ  -> internal dielectric loss (dθ<0 C1, >0 C2)
4. else -> healthy / below detection. See decision_tree.png.

## 6. Detection limits (trend-monitoring thresholds)
Thresholds: d|Vtap| 0.05%, dθ 1.0 mdeg, tanδ 1e-04,
surface_leak 1e-06 A, volume_leak 1e-05 A (deltas vs baseline).
Smallest severity that crosses any threshold:

- **pollution**: sigma_s = 1e-09  (first via tan_delta, surface_leak)
- **C1_cap**: severity = 1.1  (first via dvtap_pct, volume_leak)
- **C2_cap**: severity = 1.1  (first via dvtap_pct)
- **C1_loss**: severity = 0.01  (first via dtheta_mdeg, tan_delta, volume_leak)
- **C2_loss**: severity = 0.01  (first via dtheta_mdeg, tan_delta, volume_leak)
- **disc_short**: severity = 1  (first via dvtap_pct, volume_leak)

Blind spot: a mid-creepage non-bridging pollution patch stays below all thresholds
(sub-mdeg phase, baseline leak/tanδ) -> not identifiable from terminal observables.

## Figures
- pollution_trajectory_arrows.png, diagnostic_matrix_arrows.png, decision_tree.png
