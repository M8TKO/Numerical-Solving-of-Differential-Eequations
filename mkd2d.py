import petsc4py 
import numpy as np
import sys

petsc4py.init(sys.argv)

from mpi4py import MPI   
from petsc4py import PETSc
import matplotlib.pyplot as plt

opts = PETSc.Options()
N = opts.getInt('-NX', 10)
# N je broj ćelija mreže u smjeru x i y
# Broj vrhova mreže je N+1
h = 1.0 / N

x_points = np.linspace(0, 1, N+1)
y_points = np.linspace(0, 1, N+1)
# ---------------------------

def exact(x, y):
    return np.sin(np.pi*x) * np.sin(np.pi*y)

def f(x,y):
    return 2.0 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

def g(x,y):
    return (x+y)*0.0  

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{value:10.4f}" for value in row))
# ---------------------------
# Vektor rješenja
comm = MPI.COMM_WORLD
x = PETSc.Vec().create(comm)
x.setSizes([PETSc.DECIDE, (N+1)**2])
x.setFromOptions()

# ---------------------------
# Vektor desne strane
b = x.duplicate()
downEdge = g(x_points[0:N+1], np.repeat(y_points[0], N+1))  
upperEdge = g(x_points[0:N+1], np.repeat(y_points[N], N+1))
b.setValues(np.arange(N+1,dtype=np.int32), downEdge, PETSc.InsertMode.INSERT_VALUES)
b.setValues( np.arange((N+1)**2-1-N, (N+1)**2, dtype=np.int32), upperEdge, PETSc.InsertMode.INSERT_VALUES)
# Documentation: https://numpy.org/doc/2.3/reference/generated/numpy.arange.html 
for i in range(1, N):
    leftEdge = g(0, y_points[i])
    rightEdge = g(1, y_points[i])
    b.setValue(i*(N+1), leftEdge, PETSc.InsertMode.INSERT_VALUES)
    b.setValue(i*(N+1)+N, rightEdge, PETSc.InsertMode.INSERT_VALUES)

for i in range(1, N):
    for j in range(1, N):
        rhs_value = f(x_points[j], y_points[i])
        row = i*(N+1) + j
        b.setValue(row, rhs_value, PETSc.InsertMode.INSERT_VALUES)
b.assemble()
b.view()
# ---------------------------
# Matrica sustava
A = PETSc.Mat().create(comm)
A.setSizes((N+1)**2)
a = 4 / (h**2) # h_x = h_y = h
b_float = -1 / (h**2)
c = -1 / (h**2)

A.setValues( 
    np.arange(N+1,dtype=np.int32), 
    np.arange(N+1,dtype=np.int32),  
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)  

A.setValues(
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)

for i in range(1, N):
    A.setValues(i*(N+1), i*(N+1), 1.0, PETSc.InsertMode.INSERT_VALUES)
    A.setValues(i*(N+1)+N, i*(N+1)+N, 1.0, PETSc.InsertMode.INSERT_VALUES)
    for j in range(1, N):
        row = i*(N+1) + j
        A.setValues( 
            row, [row - (N+1), row - 1, row, row + 1, row + (N+1)],
            [c, b_float, a, b_float, c],
            PETSc.InsertMode.INSERT_VALUES
        )
A.assemble()
viewer = PETSc.Viewer().createASCII('matricaSustava.txt', comm=PETSc.COMM_WORLD)
A.view(viewer)

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setTolerances(1e-9,PETSc.CURRENT,PETSc.CURRENT, 1000)
ksp.setFromOptions()

ksp.solve(b, x)

approximation = np.array(x.getArray()).reshape((N+1, N+1))
exact_solution = np.array([[exact(i*h, j*h) for j in range(N+1)] for i in range(N+1)])

error = np.linalg.norm(approximation - exact_solution, ord=np.inf)
print(f"Maximum error (infinity norm): {error}")

X, Y = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, exact_solution, cmap='viridis')
plt.colorbar()
plt.title('Egzaktno rješenje')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, approximation, cmap='viridis')
plt.colorbar()
plt.title('Aproksimacija rješenja')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# ---------------------------
