
from dolfin import *

# mesh = UnitSquareMesh(32, 32)


mesh = Mesh()
with XDMFFile("msh.xdmf") as file:
    file.read(mesh)


# mesh = UnitSquareMesh(32, 32)
# File("mesh.pvd") << mesh

tol = 1e-14
if has_linear_algebra_backend("Epetra"):
    parameters["linear_algebra_backend"] = "Epetra"

def u_D(t):
    return Expression('1.0', degree=1, A=1, a=0.5, t=t)



# Define Dirichlet boundary (x = 0 or x = 1)
def u_boundary(x, on_boundary):
    return on_boundary and (near(x[0], 0, tol) or near(x[0], 1, tol))


V = FunctionSpace(mesh, "CG", 1)


# Define boundary condition
g = Constant(1.0)
bc = DirichletBC(V, u_D(1), u_boundary)

# Define variational problem
u = Function(V)
v = TestFunction(V)
D = Constant(0.01)
r = Constant(1)
f = Expression("x[0]*sin(x[1])",element = V.ufl_element())
#F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx
F = -D * dot(grad(u), grad(v)) * dx + r * u * u * v * dx - r * u * v * dx


# Compute solution
solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                        {"relative_tolerance": 1e-6}})

# plot(u, title="Solution")
# plot(grad(u), title="Solution gradient")


# Save solution in VTK format
file = File("fkpp_mesh_custom.pvd")
file << u

# Plot solution
#import matplotlib.pyplot as plt
#plot(u)
#plt.show()
