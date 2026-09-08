# Task 002 - Numerical Verification Framework for the Elmer Baseline

## Purpose
Verify the clean Elmer baseline by sweeping mesh levels without introducing pollution, fault, EQS phase, or thermal physics.

## Quantities Tracked
- `Vtap`
- relative `Vtap` error
- `Emax` if it can be extracted from the solved field

## Future Extension
- Richardson extrapolation
- GCI

## Scope Limits
- No pollution physics yet
- No fault physics yet
- No complex admittivity or phase analysis yet
- No thermal physics yet
