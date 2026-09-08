!------------------------------------------------------------------------------
! ComplexEQS - electroquasistatic solver with complex permittivity.
!
! Solves the time-harmonic charge-continuity equation for a lossy dielectric at
! a single (power) frequency:
!
!       div( ( sigma + i*omega*eps0*eps_r ) grad(phi) ) = 0
!
! where phi = phi_re + i*phi_im is the complex electric scalar potential, sigma
! is the electric conductivity, eps_r the relative permittivity, eps0 the vacuum
! permittivity and omega = 2*pi*f the angular frequency.
!
! The complex coefficient kappa = sigma + i*omega*eps couples conduction (Re)
! and dielectric displacement (Im). That coupling is exactly what lets conductive
! pollution perturb a capacitive divider - it cannot in a real-only electrostatic
! solve (Elmer's StatElecSolve), which is why Task 006 needs this module.
!
! Assembly idiom (complex element stiffness + Linear System Complex) follows
! Elmer's HelmholtzSolve, with the Helmholtz mass/wavenumber term dropped and the
! diffusion coefficient replaced by kappa.
!
! Validation contract: with sigma = 0 everywhere this must reproduce the real
! StatElecSolve baseline (study 001), with phase ~ 0 and tan(delta) ~ 0.
!
! Materials (per Body, in the SIF):
!   Relative Permittivity   - eps_r            (default 1)
!   Electric Conductivity   - sigma  [S/m]     (default 0)
! Frequency (Simulation or Solver):
!   Frequency / Angular Frequency  (read via GetAngularFrequency)
!
! Variable: complex scalar "Potential" with components "Potential Re"/"Potential Im".
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
SUBROUTINE ComplexEQSSolver_init( Model, Solver, dt, TransientSimulation )
!------------------------------------------------------------------------------
! Set safe defaults so a minimal SIF works: declare the complex 2-component
! variable and flag the linear system complex. ListAddNew* only set the value if
! the user has not already specified it in the SIF, so explicit SIF wins.
!------------------------------------------------------------------------------
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t)  :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp)  :: dt
  LOGICAL        :: TransientSimulation

  CALL ListAddNewString( Solver % Values, 'Variable', &
       'Potential[Potential Re:1 Potential Im:1]' )
  CALL ListAddNewLogical( Solver % Values, 'Linear System Complex', .TRUE. )
!------------------------------------------------------------------------------
END SUBROUTINE ComplexEQSSolver_init
!------------------------------------------------------------------------------


!------------------------------------------------------------------------------
SUBROUTINE ComplexEQSSolver( Model, Solver, dt, TransientSimulation )
!------------------------------------------------------------------------------
  USE DefUtils
  IMPLICIT NONE
  TYPE(Model_t)  :: Model
  TYPE(Solver_t) :: Solver
  REAL(KIND=dp)  :: dt
  LOGICAL        :: TransientSimulation
!------------------------------------------------------------------------------
  TYPE(Element_t), POINTER   :: Element
  TYPE(ValueList_t), POINTER :: Material, Constants, BC
  COMPLEX(KIND=dp), ALLOCATABLE :: STIFF(:,:), FORCE(:)
  REAL(KIND=dp), ALLOCATABLE :: Sigma(:), Epsr(:)
  REAL(KIND=dp) :: Norm, Omega, Eps0, SigmaS
  INTEGER :: t, n, nd, Nmax, active
  LOGICAL :: Found, AxiSym
  SAVE STIFF, FORCE, Sigma, Epsr
!------------------------------------------------------------------------------

  Nmax = Solver % Mesh % MaxElementDOFs
  IF ( .NOT. ALLOCATED( STIFF ) ) THEN
    ALLOCATE( STIFF(Nmax,Nmax), FORCE(Nmax), Sigma(Nmax), Epsr(Nmax) )
  END IF

  Omega = GetAngularFrequency()

  Constants => GetConstants()
  Eps0 = GetConstReal( Constants, 'Permittivity Of Vacuum', Found )
  IF ( .NOT. Found ) Eps0 = 8.8541878128e-12_dp

  CALL Info( 'ComplexEQSSolver', '-------------------------------------', Level=4 )
  WRITE( Message, '(A,ES12.4,A)' ) 'Angular frequency omega = ', Omega, ' rad/s'
  CALL Info( 'ComplexEQSSolver', Message, Level=4 )

  CALL DefaultInitialize()

  active = GetNOFActive()
  DO t=1,active
    Element => GetActiveElement(t)
    n  = GetElementNOFNodes()
    nd = GetElementNOFDOFs()
    Material => GetMaterial()

    Sigma = 0.0_dp
    Epsr  = 1.0_dp
    Sigma(1:n) = GetReal( Material, 'Electric Conductivity', Found )
    Epsr(1:n)  = GetReal( Material, 'Relative Permittivity', Found )
    IF ( .NOT. Found ) Epsr(1:n) = 1.0_dp

    CALL LocalMatrix( STIFF, FORCE, Element, n, nd, Sigma, Epsr, Omega, Eps0 )
    CALL DefaultUpdateEquations( STIFF, FORCE )
  END DO

  ! ---- Surface conductance (pollution layer on a creepage surface) ----------
  ! A sheet conductance sigma_s [S] on a boundary adds a tangential (surface)
  ! Laplace-Beltrami leakage term  a_s(phi,v) = INT_Gamma sigma_s grad_s(phi).
  ! grad_s(v) dS  to the system. For a 1-D boundary element in 2-D, ElementInfo's
  ! dBasisdx is already the tangential gradient, so the same stiffness pattern
  ! applies. Set "Surface Conductance = Real <sigma_s>" on the BC (e.g. the
  ! insulator_surface creepage curve). This is the surface-pollution model
  ! (cf. ctfem skfem_solver surf_form); it follows the creepage exactly.
  AxiSym = ( CurrentCoordinateSystem() == AxisSymmetric .OR. &
             CurrentCoordinateSystem() == CylindricSymmetric )
  DO t=1, GetNOFBoundaryElements()
    Element => GetBoundaryElement(t)
    IF ( .NOT. ActiveBoundaryElement() ) CYCLE
    BC => GetBC()
    IF ( .NOT. ASSOCIATED( BC ) ) CYCLE
    n  = GetElementNOFNodes()
    nd = GetElementNOFDOFs()
    ! nodal sigma_s (GetReal) -> may be spatially windowed via MATC in the SIF
    Sigma = 0.0_dp
    Sigma(1:n) = GetReal( BC, 'Surface Conductance', Found )
    IF ( .NOT. Found .OR. ALL( Sigma(1:n) == 0.0_dp ) ) CYCLE
    CALL SurfaceMatrix( STIFF, FORCE, Element, n, nd, Sigma, AxiSym )
    CALL DefaultUpdateEquations( STIFF, FORCE )
  END DO

  CALL DefaultFinishBulkAssembly()
  CALL DefaultFinishAssembly()
  CALL DefaultDirichletBCs()
  Norm = DefaultSolve()

CONTAINS

!------------------------------------------------------------------------------
  SUBROUTINE SurfaceMatrix( STIFF, FORCE, Element, n, nd, SigmaS, AxiSym )
!------------------------------------------------------------------------------
! Tangential (Laplace-Beltrami) leakage stiffness for a (possibly spatially
! varying) sheet conductance SigmaS on a boundary element. Real coefficient ->
! contributes to the conduction (real) part of the complex admittance.
!------------------------------------------------------------------------------
    COMPLEX(KIND=dp), INTENT(INOUT) :: STIFF(:,:), FORCE(:)
    TYPE(Element_t), POINTER        :: Element
    INTEGER, INTENT(IN)             :: n, nd
    REAL(KIND=dp), INTENT(IN)       :: SigmaS(:)
    LOGICAL, INTENT(IN)             :: AxiSym
!------------------------------------------------------------------------------
    REAL(KIND=dp) :: Basis(nd), dBasisdx(nd,3), DetJ, weight, r_ip, sigmas_ip
    LOGICAL :: stat
    INTEGER :: t, p, q
    TYPE(GaussIntegrationPoints_t) :: IP
    TYPE(Nodes_t) :: Nodes
    SAVE Nodes
!------------------------------------------------------------------------------
    CALL GetElementNodes( Nodes, Element )
    STIFF = CMPLX( 0.0_dp, 0.0_dp, KIND=dp )
    FORCE = CMPLX( 0.0_dp, 0.0_dp, KIND=dp )

    IP = GaussPoints( Element )
    DO t=1,IP % n
      stat = ElementInfo( Element, Nodes, IP % U(t), IP % V(t), IP % W(t), &
                          DetJ, Basis, dBasisdx )
      weight = IP % s(t) * DetJ
      IF ( AxiSym ) THEN
        r_ip = SUM( Basis(1:n) * Nodes % x(1:n) )
        weight = weight * r_ip
      END IF
      sigmas_ip = SUM( Basis(1:n) * SigmaS(1:n) )
      DO p=1,nd
        DO q=1,nd
          STIFF(p,q) = STIFF(p,q) + weight * CMPLX( sigmas_ip, 0.0_dp, KIND=dp ) * &
                       SUM( dBasisdx(q,1:3) * dBasisdx(p,1:3) )
        END DO
      END DO
    END DO
!------------------------------------------------------------------------------
  END SUBROUTINE SurfaceMatrix
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
  SUBROUTINE LocalMatrix( STIFF, FORCE, Element, n, nd, Sigma, Epsr, Omega, Eps0 )
!------------------------------------------------------------------------------
    COMPLEX(KIND=dp), INTENT(INOUT) :: STIFF(:,:), FORCE(:)
    TYPE(Element_t), POINTER        :: Element
    INTEGER, INTENT(IN)             :: n, nd
    REAL(KIND=dp), INTENT(IN)       :: Sigma(:), Epsr(:), Omega, Eps0
!------------------------------------------------------------------------------
    REAL(KIND=dp) :: Basis(nd), dBasisdx(nd,3), DetJ, weight, r_ip
    REAL(KIND=dp) :: sigma_ip, epsr_ip
    COMPLEX(KIND=dp) :: Kappa_ip
    LOGICAL :: stat, AxiSym
    INTEGER :: t, p, q
    TYPE(GaussIntegrationPoints_t) :: IP
    TYPE(Nodes_t) :: Nodes
    SAVE Nodes
!------------------------------------------------------------------------------
    CALL GetElementNodes( Nodes, Element )
    STIFF = CMPLX( 0.0_dp, 0.0_dp, KIND=dp )
    FORCE = CMPLX( 0.0_dp, 0.0_dp, KIND=dp )

    ! Axisymmetric (r,z): the weak form carries a radial metric, integral over
    ! (2*pi*r) dr dz. ElementInfo returns the Cartesian DetJ, so we apply the
    ! radial weight r here. The constant 2*pi cancels in the homogeneous solve
    ! and is restored in the Python capacitance post-processing. Cartesian runs
    ! (Tasks 006-008) are unaffected.
    AxiSym = ( CurrentCoordinateSystem() == AxisSymmetric .OR. &
               CurrentCoordinateSystem() == CylindricSymmetric )

    IP = GaussPoints( Element )
    DO t=1,IP % n
      stat = ElementInfo( Element, Nodes, IP % U(t), IP % V(t), IP % W(t), &
                          DetJ, Basis, dBasisdx )
      weight = IP % s(t) * DetJ
      IF ( AxiSym ) THEN
        r_ip = SUM( Basis(1:n) * Nodes % x(1:n) )
        weight = weight * r_ip
      END IF

      sigma_ip = SUM( Basis(1:n) * Sigma(1:n) )
      epsr_ip  = SUM( Basis(1:n) * Epsr(1:n) )
      Kappa_ip = CMPLX( sigma_ip, Omega * Eps0 * epsr_ip, KIND=dp )

      DO p=1,nd
        DO q=1,nd
          STIFF(p,q) = STIFF(p,q) + weight * Kappa_ip * &
                       SUM( dBasisdx(q,1:3) * dBasisdx(p,1:3) )
        END DO
      END DO
    END DO
!------------------------------------------------------------------------------
  END SUBROUTINE LocalMatrix
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
END SUBROUTINE ComplexEQSSolver
!------------------------------------------------------------------------------
