import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, multi_dot

"""
3 linked planlar robot manipulator
inverse kinematics soultion using jacobian pseudoinverse
"""


## Link 1 ##
# length 
Ls = L1 = 0.31 # [m] 
# mass
ms = m1 = 1.93 # [kg]
# Inertia
Is = I1 = 0.0141 # [kg*m^2]
# Center of mass from proximal joint
Lms = Lm1 = 0.165 # [m]

## Link 2 ##
Le = L2 = 0.34
me = m2 = 1.52
Ie = 0.0141
Lme = 0.19

## Link 3 ##
Lh = L3 = 0.08
mh = m3 = 0.52
Ih = 0.0003
Lmh = 0.055

def forward_kinematics(q1,q2,q3):
    """
    forward kinematics
    Arg:
        q1,q2,q3 - joint angles
    Return:
        [X,Y] - end effector position
    """
    X = L1*np.cos(q1) + L2*np.cos(q1+q2) + L3*np.cos(q1+q2+q3)
    Y = L1*np.sin(q1) + L2*np.sin(q1+q2) + L3*np.sin(q1+q2+q3)
    return X,Y


# three joint angles inital values
q_initial = np.array([0, 1.745, 2.09])

# end effector position inital values
X_inital = forward_kinematics(q_initial[0],q_initial[1],q_initial[2])

# end effector position final values
X_final = np.array([0.3,0.45])


# time span
tspan = np.linspace(0, 1, 20)



# before starting the loop, set the initial values
q = q_initial
X = X_inital

# close previous figure
plt.clf()
for t in tspan:

    qs = q[0]
    qe = q[1]
    qh = q[2]
 
    # Jacobian matrix 
    J11 = -L1*np.sin(q[0]) - L2*np.sin(q[0]+q[1]) - L3*np.sin(q[0]+q[1]+q[2])
    J12 = -L2*np.sin(q[0]+q[1]) - L3*np.sin(q[0]+q[1]+q[2])
    J13 = -L3*np.sin(q[0]+q[1]+q[2])
    J21 = L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]) + L3*np.cos(q[0]+q[1]+q[2])
    J22 = L2*np.cos(q[0]+q[1]) + L3*np.cos(q[0]+q[1]+q[2])
    J23 = L3*np.cos(q[0]+q[1]+q[2])

    J = np.array([[J11,J12,J13],[J21,J22,J23]])

    # Mass matrix
    H = np.array([[ms*Lms**2 + me*Lme**2 + mh*Lmh**2 + 2*me*Lms*Lme*np.cos(qe) + 2*mh*Lms*Lmh*np.cos(qh) + 2*mh*Lme*Lmh*np.cos(qe+qh) + Is + Ie + Ih, me*Lme**2 + mh*Lmh**2 + me*Lms*Lme*np.cos(qe) + mh*Lms*Lmh*np.cos(qh) + mh*Lme*Lmh*np.cos(qe+qh) + Ie + Ih, mh*Lmh**2 + mh*Lms*Lmh*np.cos(qh) + mh*Lme*Lmh*np.cos(qe+qh) + Ih],
                    [me*Lme**2 + mh*Lmh**2 + me*Lms*Lme*np.cos(qe) + mh*Lms*Lmh*np.cos(qh) + mh*Lme*Lmh*np.cos(qe+qh) + Ie + Ih, me*Lme**2 + mh*Lmh**2 + mh*Lme*Lmh*np.cos(qe+qh) + Ie + Ih, mh*Lmh**2 + mh*Lme*Lmh*np.cos(qe+qh) + Ih],
                    [mh*Lmh**2 + mh*Lms*Lmh*np.cos(qh) + mh*Lme*Lmh*np.cos(qe+qh) + Ih, mh*Lmh**2 + mh*Lme*Lmh*np.cos(qe+qh) + Ih, mh*Lmh**2 + Ih]])


    # pseudoinverse of Jacobian matrix with mass matrix
    JH = multi_dot([inv(H),J.T,inv(multi_dot([J,inv(H),J.T]))])
    
    # end effector velocity
    X_dot = ((X_final-X)/tspan[-1])

    # calculate qdot using pseudoinverse of Jacobian matrix times the end effector velocity
    q_dot = JH.dot(X_dot)
  
    # calculate q using qdot times t
    new_q = q_dot*t

    # update q
    q = q + new_q

    # update end effector position
    X = forward_kinematics(q[0],q[1],q[2])



    # plot end effector position
    plt.plot(X[0],X[1],'ro')

    #plot the links
    plt.plot([0,L1*np.cos(q[0])],[0,L1*np.sin(q[0])],'b')
    plt.plot([L1*np.cos(q[0]),L1*np.cos(q[0])+L2*np.cos(q[0]+q[1])],[L1*np.sin(q[0]),L1*np.sin(q[0])+L2*np.sin(q[0]+q[1])],'b')
    plt.plot([L1*np.cos(q[0])+L2*np.cos(q[0]+q[1]),L1*np.cos(q[0])+L2*np.cos(q[0]+q[1])+L3*np.cos(q[0]+q[1]+q[2])],[L1*np.sin(q[0])+L2*np.sin(q[0]+q[1]),L1*np.sin(q[0])+L2*np.sin(q[0]+q[1])+L3*np.sin(q[0]+q[1]+q[2])],'b')
    
    plt.xlim(-0.1,0.6)
    plt.ylim(-0.1,0.6)
    plt.pause(0.01)  
plt.show()

    
