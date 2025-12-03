# 2D Poisson Equation – Finite Difference Method (PETSc + MPI)

This project solves the 2D Poisson equation on the unit square (0, 1) × (0, 1):

- Problem:  -Δu = f in Ω = (0,1)²
- Boundary condition: u = g on ∂Ω

The equation is discretized using a standard 5-point finite difference stencil on a structured (N+1) × (N+1) grid.  
Dirichlet boundary conditions are imposed using the exact solution.

The computation is parallelized using PETSc and MPI, and the code visualizes both the numerical and exact solutions.

---

## File Overview

- `mkd2d.py`  
  Main script that:
  - constructs the grid and spacing,
  - defines the exact solution, right-hand side, and boundary data,
  - assembles the Poisson matrix using PETSc sparse storage,
  - applies Dirichlet boundary conditions,
  - solves the resulting linear system with PETSc KSP,
  - compares the result with the exact analytical solution,
  - visualizes both fields and computes the sup-norm error.

---

## Mathematical Setup

- Exact solution:  
  u(x, y) = sin(x) · cos(y)

- Right-hand side (from −Δu):  
  f(x, y) = 2 · sin(x) · cos(y)

- Boundary data:  
  g(x, y) = u(x, y) on the boundary of the unit square.

Finite difference stencil (for interior points):

- grid spacing: h = 1 / N
- central coefficient: a = 4 / h²  
- neighbor coefficients: b = c = −1 / h²  
- row pattern: [c, b, a, b, c] corresponding to (up, left, center, right, down).

---

## Requirements

- Python 3
- `petsc4py`
- `mpi4py`
- `numpy`
- `matplotlib`

Example installation (assuming PETSc and MPI are already installed on the system):

```bash
pip install numpy matplotlib mpi4py petsc petsc4py
