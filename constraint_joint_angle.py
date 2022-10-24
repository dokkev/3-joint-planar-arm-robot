import numpy as np
import matplotlib.pyplot as plt

"""
3 linked planlar robot manipulator
inverse kinematics soultion using jacobian pseudoinverse
"""


## Link 1 ##
# length 
L1 = 0.31 # [m] 
# mass
m1 = 1.93 # [kg]
# Inertia
I1 = 0.0141 # [kg*m^2]
# Center of mass from proximal joint
Lm1 = 0.165 # [m]

## Link 2 ##
# length
L2 = 0.34
# mass
m2 = 1.52
# Inertia


## Link 3 ##
L3 = 0.08

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
q_initial = np.array([2, 1.0472, 1.0472])

# end effector position inital values
X_inital = forward_kinematics(q_initial[0],q_initial[1],q_initial[2])
print(X_inital)

# end effector position final values
X_final = np.array([0,0.7])


# time span
tspan = np.linspace(0, 1, 30)



# before starting the loop, set the initial values
q = q_initial
X = X_inital

# close previous figure
plt.clf()
for t in tspan:
 

    # Jacobian matrix with joint angles constraints
    J11 = -L1*np.sin(q[0]) - L2*np.sin(q[0]+q[1]) - L3*np.sin(q[0]+q[1]+q[2])
    J12 = -L2*np.sin(q[0]+q[1]) - L3*np.sin(q[0]+q[1]+q[2])
    J13 = -L3*np.sin(q[0]+q[1]+q[2])
    J21 = L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]) + L3*np.cos(q[0]+q[1]+q[2])
    J22 = L2*np.cos(q[0]+q[1]) + L3*np.cos(q[0]+q[1]+q[2])
    J23 = L3*np.cos(q[0]+q[1]+q[2])

    J = np.array([[J11,J12,J13],[J21,J22,J23],[1,1,1]])
    

    # pseudoinverse of Jacobian matrix
    J_pinv = np.linalg.pinv(J)


    
    # bell-shaped velocity profile vp to produce staright trajectory
    # movement duration, T
    T = tspan[-1]
    tn = t/T
    # amplitude, A
    A = 1
    vp = (tn**2-2*tn+1)*30*A*tn**2/T

    # end effector velocity
    X_dot = vp*((X_final-X)/tspan[-1])
    # add 1 element to X_dot to match the shape of J_pinv
    X_dot = np.append(X_dot,0)


    # calculate qdot using pseudoinverse of Jacobian matrix times the end effector velocity
    q_dot = J_pinv.dot(X_dot)

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
    
    plt.xlim(-0.5,0.5)
    plt.ylim(-0.5,1.5)
    plt.pause(0.01)
plt.show()
    
