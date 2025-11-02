import petsc4py 
import numpy as np
import sys

petsc4py.init(sys.argv)

from mpi4py import MPI   # mora biti iza inicijalizacije 
from petsc4py import PETSc

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
    return x + y#(x+y)*0.0  

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
b.assemble()

# ---------------------------
# Matrica sustava
A = PETSc.Mat().create(comm)
A.setSizes((N+1)**2)
a = 4 / (h**2) # h_x = h_y = h
b = -1 / (h**2)
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

A.assemble()
viewer = PETSc.Viewer().createASCII('matrix_output.txt', comm=PETSc.COMM_WORLD)
A.view(viewer)