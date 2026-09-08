# Task 009 - Representative CVT Validation

Axisymmetric ComplexEQS on the self-contained DDB-123 geometry, validated
against the datasheet and cross-checked against the ctfem scikit-fem backend
on the identical mesh.

- Elmer terminal C: **5616.4 pF** (datasheet 5600 pF)
- Divider ratio: **0.1666** (nominal 0.1667 = n_c2/n)
- |Vtap|: 11829.5 V at U0=71014.1 V; phase -0.0001 deg
- Creepage (geometry): 3062 mm (datasheet 3075 mm)

| check | status | detail |
|-------|--------|--------|
| Terminal C ~ datasheet 5600 pF | PASS | Elmer=5616.4 pF, rated=5600 pF, rel=0.003 (<0.05) |
| Divider ratio ~ nominal n_c2/n | PASS | Elmer=0.1666, nominal=0.1667, rel=0.001 (<0.02) |
| C cross-backend (Elmer vs skfem, same mesh) | PASS | Elmer=5616.4 pF, skfem=5609.9 pF, rel=0.001 (<0.03) |
| Divider cross-backend (Elmer vs skfem) | PASS | Elmer=0.1666, skfem=0.1666, rel=0.000 (<0.03) |

**Result: ALL CHECKS PASS.**
