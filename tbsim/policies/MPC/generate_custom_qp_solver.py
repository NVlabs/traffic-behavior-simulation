import cvxpy as cp

m, n = 3, 2
x = cp.Variable(n, name='x')
y = cp.Variable(n, name='y')
Q = cp.Parameter((2,2), name='Q',PSD=True)
R = cp.Parameter((2,2), name='R',PSD=True)
A = cp.Parameter((m, n), name='A', sparsity=[(0, 0), (0, 1), (1, 1)])
b = cp.Parameter(m, name='b')

G = cp.Parameter((m, n), name='H', sparsity=[(0, 0), (0, 1), (1, 1)])
h = cp.Parameter(m, name='h')
problem = cp.Problem(cp.Minimize(0.5*(cp.sum_squares(Q@x)+cp.sum_squares(R@y))), [A@x+b <= 0, G@y+h <= 0])


import numpy as np

np.random.seed(0)
A.value = np.zeros((m, n))
A.value[0, 0] = np.random.randn()
A.value[0, 1] = np.random.randn()
A.value[1, 1] = np.random.randn()
b.value = np.random.randn(m)

G.value = np.zeros((m, n))
G.value[0, 0] = np.random.randn()
G.value[0, 1] = np.random.randn()
G.value[1, 1] = np.random.randn()
h.value = np.random.randn(m)
Q.value = np.eye(2)
R.value = np.eye(2)

# problem.solve()

from cvxpygen import cpg

# cpg.generate_code(problem, code_dir='nonneg_LS', solver='SCS')

import time
import sys

# import extension module and register custom CVXPY solve method
from nonneg_LS.cpg_solver import cpg_solve
problem.register_solve('cpg', cpg_solve)

# solve problem conventionally
t0 = time.time()
val = problem.solve(solver='GUROBI')
t1 = time.time()
sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
sys.stdout.write('Objective function value: %.6f\n' % val)

# solve problem with C code via python wrapper
t0 = time.time()
problem.param_dict["b"].value = -b.value
val = problem.solve(method='cpg', updated_params=['A', 'b'])
t1 = time.time()
sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
sys.stdout.write('Objective function value: %.6f\n' % val)