# Import necessary libraries
import petsc4py 
import numpy as np
import sys

petsc4py.init(sys.argv)

from mpi4py import MPI   
from petsc4py import PETSc
import matplotlib.pyplot as plt

opts = PETSc.Options()
N = opts.getInt('-NX', 10)
comm = MPI.COMM_WORLD
# N is the number of grid cells in the x and y directions
# The total number of grid points is N+1
h = 1.0 / N

x_points = np.linspace(0, 1, N+1)
y_points = np.linspace(0, 1, N+1)
# ---------------------------
# Definition of the exact solution and the right-hand side
def exact(x, y):
    return np.sin(x) * np.cos(y)

def f(x,y):
    return 2 * np.sin(x) * np.cos(y)

def g(x,y):
    return exact(x,y)
# ---------------------------
# Solution vector
x = PETSc.Vec().create(comm)
x.setSizes([PETSc.DECIDE, (N+1)**2])
x.setFromOptions()

# ---------------------------
# Right-hand-side vector
b = x.duplicate()
# Boundary conditions on y = 0 and y = 1
downEdge = g(x_points[0:N+1], np.repeat(y_points[0], N+1))  
upperEdge = g(x_points[0:N+1], np.repeat(y_points[N], N+1))
b.setValues(np.arange(N+1,dtype=np.int32), downEdge, PETSc.InsertMode.INSERT_VALUES)
b.setValues(np.arange((N+1)**2-1-N, (N+1)**2, dtype=np.int32), upperEdge, PETSc.InsertMode.INSERT_VALUES)

# Boundary conditions on x = 0 and x = 1
# Values of G_k
for i in range(1, N):
    leftEdge = g(0, y_points[i])
    rightEdge = g(1, y_points[i])
    b.setValue(i*(N+1), leftEdge, PETSc.InsertMode.INSERT_VALUES)
    b.setValue(i*(N+1)+N, rightEdge, PETSc.InsertMode.INSERT_VALUES)

# Values of F_k
for i in range(1, N):
    for j in range(1, N):
        rhs_value = f(x_points[j], y_points[i])
        row = i*(N+1) + j
        b.setValue(row, rhs_value, PETSc.InsertMode.INSERT_VALUES)
b.assemble()
# ---------------------------
# System matrix
A = PETSc.Mat().create(comm)
A.setSizes((N+1)**2)
# Coefficients a, b, c from the finite-difference stencil
# b is already used for the RHS vector, so a different name is used
a = 4 / (h**2) # h_x = h_y = h
b_float = -1 / (h**2)
c = -1 / (h**2)

# First block equals the identity matrix
A.setValues( 
    np.arange(N+1,dtype=np.int32), 
    np.arange(N+1,dtype=np.int32),  
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)  

# Last block equals the identity matrix
A.setValues(
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)

for i in range(1, N):
    # Identity entries at the start and end of each diagonal block
    A.setValues(i*(N+1), i*(N+1), 1.0, PETSc.InsertMode.INSERT_VALUES)
    A.setValues(i*(N+1)+N, i*(N+1)+N, 1.0, PETSc.InsertMode.INSERT_VALUES)
    # Remaining stencil entries [c, b, a, b, c] in each row
    for j in range(1, N):
        row = i*(N+1) + j
        A.setValues( 
            row, [row - (N+1), row - 1, row, row + 1, row + (N+1)],
            [c, b_float, a, b_float, c],
            PETSc.InsertMode.INSERT_VALUES
        )
A.assemble()

# ---------------------------
# Solving the system
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setTolerances(1e-12,PETSc.CURRENT,PETSc.CURRENT, 1000)
ksp.setFromOptions()

ksp.solve(b, x)

# ---------------------------
# Compute error and display results
approximation = np.array(x.getArray()).reshape((N+1, N+1))

X, Y = np.meshgrid(x_points, y_points, indexing='xy')
exact_solution = exact(X, Y)

err_inf = np.max(np.abs(approximation - exact_solution))
print(f"Maximum error in the sup norm: {err_inf}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(x_points, y_points, exact_solution, cmap='viridis')
plt.colorbar()
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf(x_points, y_points, approximation, cmap='viridis')
plt.colorbar()
plt.title('Numerical Approximation')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
