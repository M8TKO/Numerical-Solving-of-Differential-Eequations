# Dodavanje potrebnih biblioteka
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
# N je broj ćelija mreže u smjeru x i y
# Broj vrhova mreže je N+1
h = 1.0 / N

x_points = np.linspace(0, 1, N+1)
y_points = np.linspace(0, 1, N+1)
# ---------------------------
# Definicija egzaktog rješenja i desne strane
def exact(x, y):
    return np.pow(x,3) + np.power(y,3)

def f(x,y):
    return -6*x - 6*y

def g(x,y):
    return exact(x,y)
# ---------------------------
# Vektor rješenja
x = PETSc.Vec().create(comm)
x.setSizes([PETSc.DECIDE, (N+1)**2])
x.setFromOptions()

# ---------------------------
# Vektor desne strane
b = x.duplicate()

# Rubni uvjeti na y = 0 i y = 1
downEdge = g(x_points[0:N+1], np.repeat(y_points[0], N+1))  
upperEdge = g(x_points[0:N+1], np.repeat(y_points[N], N+1))
b.setValues(np.arange(N+1,dtype=np.int32), downEdge, PETSc.InsertMode.INSERT_VALUES)
b.setValues( np.arange((N+1)**2-1-N, (N+1)**2, dtype=np.int32), upperEdge, PETSc.InsertMode.INSERT_VALUES)

# Rubni uvjeti na x = 0 i x = 1
    # Vrijednosti G_k
for i in range(1, N):
    leftEdge = g(0, y_points[i])
    rightEdge = g(1, y_points[i])
    b.setValue(i*(N+1), leftEdge, PETSc.InsertMode.INSERT_VALUES)
    b.setValue(i*(N+1)+N, rightEdge, PETSc.InsertMode.INSERT_VALUES)
    # Vrijednosti F_k
for i in range(1, N):
    for j in range(1, N):
        rhs_value = f(x_points[j], y_points[i])
        row = i*(N+1) + j
        b.setValue(row, rhs_value, PETSc.InsertMode.INSERT_VALUES)
b.assemble()
# ---------------------------
# Matrica sustava
A = PETSc.Mat().create(comm)
A.setSizes((N+1)**2)
# Brojevi a,b,c kao iz skripte, b je već definiran kao vektor 
# pa koristimo drugi naziv b_float
a = 4 / (h**2) # h_x = h_y = h
b_float = -1 / (h**2)
c = -1 / (h**2)

# Prvi blok je jednak jediničnoj matrici
A.setValues( 
    np.arange(N+1,dtype=np.int32), 
    np.arange(N+1,dtype=np.int32),  
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)  

# Zadnji blok je jednak jediničnoj matrici
A.setValues(
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.arange((N+1)**2 - (N+1), (N+1)**2,dtype=np.int32), 
    np.eye(N+1),
    PETSc.InsertMode.INSERT_VALUES
)

for i in range(1, N):
    # Jedinice na početku i kraju svakog dijagonalnog bloka
    A.setValues(i*(N+1), i*(N+1), 1.0, PETSc.InsertMode.INSERT_VALUES)
    A.setValues(i*(N+1)+N, i*(N+1)+N, 1.0, PETSc.InsertMode.INSERT_VALUES)
    # Preostali elementi [c, b, a, b, c] u svakom retku
    for j in range(1, N):
        row = i*(N+1) + j
        A.setValues( 
            row, [row - (N+1), row - 1, row, row + 1, row + (N+1)],
            [c, b_float, a, b_float, c],
            PETSc.InsertMode.INSERT_VALUES
        )
A.assemble()

# ---------------------------
# Rješavanje sustava
ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setTolerances(1e-9,PETSc.CURRENT,PETSc.CURRENT, 1000)
ksp.setFromOptions()

ksp.solve(b, x)

# ---------------------------
# Izračun greške i ispis rezultata
approximation = np.array(x.getArray()).reshape((N+1, N+1))
X, Y = np.meshgrid(x_points, y_points, indexing='xy')
exact_solution = exact(X, Y)


error = np.linalg.norm(approximation - exact_solution, ord=np.inf)
print(f"Maksimalna greška u sup normi: {error}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf( x_points, y_points, exact_solution, cmap='viridis')
plt.colorbar()
plt.title('Egzaktno rješenje')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.contourf( x_points, y_points, approximation, cmap='viridis')
plt.colorbar()
plt.title('Aproksimacija rješenja')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()