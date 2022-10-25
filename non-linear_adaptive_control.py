import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot


# length
l = 12e-2 # [m]


# Parallel robot with two DOF moving in the horizontal plane.


q_initial = np.array([0.785, 0.785])

# time span
tspan = np.linspace(0, 1, 20)
q = q_initial

for t in tspan:
    # non linear adaptive control
    




    s12 = np.sin(q[0] + q[1])
    c12 = np.cos(q[0] + q[1])
    psi = [[s12*

    ]]