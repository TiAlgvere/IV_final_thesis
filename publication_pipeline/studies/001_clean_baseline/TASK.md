# Task 001 - Clean Baseline ElmerFEM Pipeline

## Goal
Establish a publication-oriented baseline workflow without modifying the legacy scikit-fem prototype.

## Scope
- Build a clean Gmsh mesh with stable physical names
- Convert the mesh with ElmerGrid
- Provide a baseline SIF template for an electrostatic / EQS solve
- Prepare the study for later validation against capacitive divider theory

## Acceptance Criteria
- Mesh builds successfully from `publication_pipeline/scripts/build_mesh.py`
- Physical groups survive in the generated `.msh` file
- ElmerGrid conversion succeeds from `publication_pipeline/scripts/convert_mesh_elmer.py`
- The next study step is clean Vtap validation against capacitive divider theory

## Notes
- Pollution sweep is out of scope for this task
- Mesh convergence is out of scope for this task
- Thesis text changes are out of scope for this task
