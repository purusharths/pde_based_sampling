import numpy as np
import fenics as fncs

from numpy.random import standard_normal
from scipy.linalg import cholesky, solve_triangular
from scipy.special import gamma
from dolfin import File


mesh = fncs.Mesh()
with fncs.XDMFFile("msh.xdmf") as file:
    file.read(mesh)


# ---------------------- RANDOM FIELD PARAMETERS ------------------------------

corrLength = 0.1
nu = 1.

kappaSq = 2. * nu / corrLength**2
beta = 0.5 * nu + 0.25


# -------------------------- DISCRETISATION -----------------------------------

V = fncs.FunctionSpace(mesh, "Lagrange", 1)

u = fncs.TrialFunction(V)
v = fncs.TestFunction(V)

# bilinear form
coeff = fncs.Constant(kappaSq)
a = fncs.dot(fncs.grad(u), fncs.grad(v)) * fncs.dx + coeff * u * v * fncs.dx

# assemble the stiffness matrix beforehand, as it stays the same for each
# computation. It can be reused.
A = fncs.assemble(a)

"""
    I choose homogeneous Neumann boundary conditions so we don't need the
    boundary condition part.
"""

solver = fncs.LUSolver(A)

fncs.set_log_level(fncs.LogLevel.ERROR)

# only works for p.w. linear Lagrange
nDof = mesh.num_vertices()


# ------------------ WHITE NOISE FACTORISATION --------------------------------

massMatForm = u * v * fncs.dx
M = fncs.assemble(massMatForm)

useMassLumping = (nDof > 2000)

if (useMassLumping):

    U = np.zeros(nDof)

    U[0] = np.sum(M.getrow(0)[1])
    U[nDof - 1] = np.sum(M.getrow(nDof - 1)[1])

    for i in range(1, nDof - 1):
        U[i] = np.sum(M.getrow(i)[1])

    U = 1. / np.sqrt(U)

else:

    U = cholesky(M.array())


# ------------------ COMPUTING THE REALISATION --------------------------------

varScaling = np.sqrt(
    gamma(nu + 0.5) * np.sqrt(4. * np.pi) * np.power(kappaSq, nu) / gamma(nu))

if (useMassLumping):
    feWhiteNoise = varScaling * U * standard_normal(nDof)

else:
    feWhiteNoise = varScaling * solve_triangular(U, standard_normal(nDof))

sol = fncs.Function(V)
b = fncs.Function(V)

# fill Dofs of b with the white noise vector
b.vector().set_local(feWhiteNoise)

solver.solve(sol.vector(), b.vector())
print(dir(u))
# Save solution in VTK format
file = File("mesh_custom.pvd")
file << u.evaluate()
