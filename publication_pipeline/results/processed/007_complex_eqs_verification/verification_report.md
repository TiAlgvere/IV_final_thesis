# Task 007 - ComplexEQS Verification Report

Verification audit of `publication_pipeline/elmer/solvers/ComplexEQS.F90`.
No new physics, no new fault cases, no new thesis claims - this only checks
that the solver reproduces the intended weak form
`div((sigma + i*omega*eps0*eps_r) grad phi) = 0` on analytically tractable cases.

**Result: 6/6 tests passed.**

| Test | Status | Tolerance |
|------|--------|-----------|
| T1 sigma=0 reduces to StatElec baseline | PASS | max|dRe|/V0 < 1e-6 and max|Im|/V0 < 1e-6 (same mesh) |
| T2 sign convention (positive dissipation, +jwe reactive sign) | PASS | P_loss>0; |P_loss-Re(1/2 V I*)|/P_loss<1e-6; Im(Y)>0; sign & value of complex Vi match analytic |
| T3 analytic lossy capacitor admittance | PASS | |Y_num - Y_analytic|/|Y_analytic| < 1e-3 and field-uniformity rel-std < 1e-6 |
| T4 mesh convergence of lossy-capacitor admittance | PASS | rel_err_Y < 1e-3 at every refinement level |
| T5 two-region continuity (potential + normal current) | PASS | |J_n1-J_n2|/|J_n1| < 1e-3; |Vi_num-Vi_analytic|/|Vi_analytic| < 1e-3; potential bounded/continuous |
| T6 axisymmetric coaxial capacitor | PASS | |C_num - C_analytic|/C_analytic < 1e-3 (axisymmetric metric); Im/V0 < 1e-9 |

## T1 sigma=0 reduces to StatElec baseline - PASS
Tolerance: max|dRe|/V0 < 1e-6 and max|Im|/V0 < 1e-6 (same mesh)

| metric | value |
|--------|-------|
| same_mesh_node_order | True |
| max_abs_dRe_V | 6.548362e-11 |
| rel_dRe | 5.953056e-16 |
| max_abs_Im_V | 0.000000e+00 |
| rel_Im | 0.000000e+00 |
| n_nodes | 86 |

## T2 sign convention (positive dissipation, +jwe reactive sign) - PASS
Tolerance: P_loss>0; |P_loss-Re(1/2 V I*)|/P_loss<1e-6; Im(Y)>0; sign & value of complex Vi match analytic

| metric | value |
|--------|-------|
| p_loss_W_per_m | 1.000000e+00 |
| p_terminal_W_per_m | 1.000000e+00 |
| power_balance_rel | 0.000000e+00 |
| im_Y_positive_capacitive | True |
| Vi_num_imag | 1.332796e+01 |
| Vi_analytic_imag | 1.332796e+01 |
| Vi_sign_and_value_match | True |

## T3 analytic lossy capacitor admittance - PASS
Tolerance: |Y_num - Y_analytic|/|Y_analytic| < 1e-3 and field-uniformity rel-std < 1e-6

| metric | value |
|--------|-------|
| g_num | 2.000000e-06 |
| c_num | 7.083350e-11 |
| y_num | 2.000000e-06 + 2.225300e-08j |
| y_analytic | 2.000000e-06 + 2.225300e-08j |
| g_analytic | 2.000000e-06 |
| c_analytic | 7.083350e-11 |
| rel_err_Y | 1.654259e-18 |
| p_loss | 1.000000e+00 |
| ey_mean_abs | 1.000000e+05 |
| ey_expected | 1.000000e+05 |
| ey_rel_std | 4.411697e-15 |
| nelem | 1208 |
| width | 2.000000e-02 |
| gap | 1.000000e-02 |
| eps_r | 4.000000e+00 |
| sigma | 1.000000e-06 |
| v0 | 1.000000e+03 |

## T4 mesh convergence of lossy-capacitor admittance - PASS
Tolerance: rel_err_Y < 1e-3 at every refinement level
Note: Parallel-plate field is uniform, so linear FE is exact at every level (error sits at round-off); this verifies mesh-independence.

| metric | value |
|--------|-------|
| worst_rel_err_Y | 1.654259e-18 |
| levels | [4, 8, 16, 32] |

## T5 two-region continuity (potential + normal current) - PASS
Tolerance: |J_n1-J_n2|/|J_n1| < 1e-3; |Vi_num-Vi_analytic|/|Vi_analytic| < 1e-3; potential bounded/continuous

| metric | value |
|--------|-------|
| j_continuity_rel | 7.563092e-16 |
| abs_j1 | 1.842982e-02 |
| abs_j2 | 1.842982e-02 |
| vi_num | 9.117872e+01 + 1.332796e+01j |
| vi_analytic | 9.117872e+01 + 1.332796e+01j |
| vi_rel | 9.271152e-16 |
| vi_std_over_nodes | 2.128468e-13 |
| potential_continuous_C0 | True |

## T6 axisymmetric coaxial capacitor - PASS
Tolerance: |C_num - C_analytic|/C_analytic < 1e-3 (axisymmetric metric); Im/V0 < 1e-9
Note: Curved 1/r field -> genuine convergence check; validates the radial weight.

| metric | value |
|--------|-------|
| C_num_F | 1.203937e-11 |
| C_analytic_F | 1.203911e-11 |
| rel_err_C | 2.120591e-05 |
| a_m | 2.000000e-02 |
| b_m | 4.000000e-02 |
| length_m | 5.000000e-02 |
| eps_r | 3.000000e+00 |
| nelem | 9224 |
| rel_Im | 0.000000e+00 |

## Artifacts
- `mesh_convergence.csv`, `admittance_lossy_capacitor.csv`
- `verification_plots.png` (convergence + two-region potential profile)
- Raw Elmer runs: `results/raw/007_complex_eqs_verification/`
