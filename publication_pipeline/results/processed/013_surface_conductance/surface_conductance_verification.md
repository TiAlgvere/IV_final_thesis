# Task 013 - Surface-Conductance Term Verification (ComplexEQS vs ctfem skfem)

Same clean CVT mesh, surface pollution sigma_s on insulator_surface, both backends.

| sigma_s [S] | C Elmer/skfem [pF] | ratio E/sk | phase E/sk [deg] | tand E/sk |
|---|---|---|---|---|
| 1e-09 | 5617.2 / 5610.8 | 0.1666 / 0.1666 | -5.195e-04 / -5.328e-04 | 2.247e-03 / 2.244e-03 |
| 1e-07 | 5618.0 / 5611.6 | 0.1667 / 0.1666 | -1.680e-02 / -1.683e-02 | 1.809e-02 / 1.811e-02 |
| 1e-05 | 5618.1 / 5611.6 | 0.1667 / 0.1667 | -2.845e-04 / -2.869e-04 | 1.609e+00 / 1.610e+00 |

**Result: ALL CHECKS PASS** (C<1%, ratio<2%, phase<10%, tand<10%; phase/tand looser for P1-vs-P2 + small signal).
