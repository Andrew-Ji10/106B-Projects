import sympy
from sympy.core.function import diff

"""
Description of SimpleArm, including
- dimensions
- dynamics
- body Jacobian
"""

# Constants
l1 = sympy.Symbol("l1") # length of each arm
l2 = sympy.Symbol("l2")
m1 = sympy.Symbol("m1") # mass of each arm
m2 = sympy.Symbol("m2") 
I1 = sympy.Symbol("I1") # moment of inertia of each arm
I2 = sympy.Symbol("I2") 
g = sympy.Symbol("g") # acceleration due to gravity

# Define generalized coordinates as functions of time
# and specify their derivatives for prettier computation
t = sympy.Symbol("t")
dim = 2
q_funcs = [sympy.Function(f"q{i}") for i in range(1, 1 + dim)]
q_dot_funcs = [sympy.Function(f"qdot{i}") for i in range(1, 1 + dim)]
q_ddot_funcs = [sympy.Function(f"qddot{i}") for i in range(1, 1 + dim)]

# Putting this in a loop didn't work for god knows what reason
q_funcs[0].fdiff = lambda self, argindex=1: q_dot_funcs[0](self.args[argindex-1])
q_funcs[1].fdiff = lambda self, argindex=1: q_dot_funcs[1](self.args[argindex-1])

# Same with this
q_dot_funcs[0].fdiff = lambda self, argindex=1: q_ddot_funcs[0](self.args[argindex-1])
q_dot_funcs[1].fdiff = lambda self, argindex=1: q_ddot_funcs[1](self.args[argindex-1])

# Generalized coordinates
q = sympy.Matrix(
    [[q_funcs[i](t)] for i in range(dim)]
)
q1, q2 = q

# First derivatives
q_dot = sympy.diff(q, t)
q1_dot, q2_dot  = q_dot

# Second derivatives
q_ddot = sympy.diff(q, t, t)
q1_ddot, q2_ddot = q_ddot

vars = [l1, l2, m1, m2, I1, I2, g, q, q_dot]

# Find the xy position of each center of mass
cm1_xy = sympy.Matrix([
        [-(l1/2)*sympy.sin(q1)],
        [(l1/2)*sympy.cos(q1)]
    ])
cm2_xy = sympy.Matrix([
    [-l1*sympy.sin(q1) - (l2/2)*sympy.sin(q1+q2)],
    [l1*sympy.cos(q1) + (l2/2)*sympy.cos(q1+q2)]
])

# Find the velocity vector for each center of mass
# sympy.diff(arg1, arg2) will be useful here (takes derivative of first argument with respect to the second)
cm1_xy_vel = sympy.diff(cm1_xy, t)
cm2_xy_vel = sympy.diff(cm2_xy, t)

# Now, we need to find the kinetic energy T
# This will be the sum of the translational kinetic energy and rotational kinetic energy

# Get the squared magnitude of the velocity for each center of mass for the translational kinetic energy
# Use the dot product to do this
cm1_vel_squared = cm1_xy_vel.dot(cm1_xy_vel)
cm2_vel_squared = cm2_xy_vel.dot(cm2_xy_vel)

# Get the translational kinetic energy
T_t = (1/2)*m1*cm1_vel_squared + (1/2)*m2*cm2_vel_squared

# Get the rotional kinetic energy
T_r = (1/2)*I1*q1_dot**2 + (1/2)*I2*(q1_dot + q2_dot)**2

# Get total kinetic energy
T = T_t + T_r

# Next, we'll get the potential energy
V = m1*g*(l1/2)*sympy.cos(q1) + m2*g*((l1)*sympy.cos(q1) + (l2/2)*sympy.cos(q1+q2))

# Helper function to compute Lagrangian dynamics given energy functions
def LagrangianDynamics(T, V, q, dq):
    f_diff = sympy.diff(T, dq).T
    M = sympy.Matrix.hstack(*[sympy.diff(f_diff[i], dq) for i in range(dim)])
    C = sympy.zeros(dim, dim)
    # Don't ask me how this works
    for k in range(dim):
        for j in range(dim):
            for i in range(dim):
                C[k, j] = C[k, j] + (1/2)*(sympy.diff(M[k, j], q[i]) + sympy.diff(M[k, i], q[j]) - sympy.diff(M[i, j], q[k]))*dq[i]
    G = sympy.diff(V, q)
    return M, C, G

# We can now directly compute our dynamics
M, C, G = LagrangianDynamics(T, V, q, q_dot)
M_func = sympy.lambdify(vars, M, "numpy")
C_func = sympy.lambdify(vars, C, "numpy")
G_func = sympy.lambdify(vars, G, "numpy")

# Body jacobian for manipulator
vel1 = sympy.Matrix([
    [-l1*sympy.cos(q1)],
    [-l1*sympy.sin(q1)]
])

vel2 = sympy.Matrix([
    [-l2*sympy.cos(q1 + q2)],
    [-l2*sympy.sin(q1 + q2)]
])

J_body = sympy.Matrix.hstack(vel1 + vel2, vel2)
J_body_func = sympy.lambdify(vars, J_body, "numpy")