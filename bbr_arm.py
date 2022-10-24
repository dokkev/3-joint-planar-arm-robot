import numpy as np
import matplotlib.pyplot as plt

"""
3 linked planlar robot manipulator
inverse kinematics soultion using jacobian pseudoinverse
"""


# Link 1
L1 = 0.31
# Link 2
L2 = 0.34
# Link 3
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


# three joint angles inital values and 
q_initial = np.array([2, 1.0472, 1.0472])

# end effector position inital values
X_inital = forward_kinematics(q_initial[0],q_initial[1],q_initial[2])

# end effector position final values
X_final = np.array([-0.05,0.7])


# time span
tspan = np.linspace(0, 1, 20)


for t in tspan:

    # Jacobian matrix
    J11 = -L1*np.sin(q_initial[0])-L2*np.sin(q_initial[0]+q_initial[1])-L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2])
    J12 = -L2*np.sin(q_initial[0]+q_initial[1])-L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2])
    J13 = -L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2])
    J21 = L1*np.cos(q_initial[0])+L2*np.cos(q_initial[0]+q_initial[1])+L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2])
    J22 = L2*np.cos(q_initial[0]+q_initial[1])+L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2])
    J23 = L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2])


    J = np.array([[J11,J12,J13],[J21,J22,J23]])

    # Jacobian matrix with joint angles constraints
    J = np.array([[-L1*np.sin(q_initial[0])-L2*np.sin(q_initial[0]+q_initial[1])-L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2]), -L2*np.sin(q_initial[0]+q_initial[1])-L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2]), -L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2])],
                [L1*np.cos(q_initial[0])+L2*np.cos(q_initial[0]+q_initial[1])+L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2]), L2*np.cos(q_initial[0]+q_initial[1])+L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2]), L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2])]
                ,[1,1,1]])
  

    # pseudoinverse of Jacobian matrix
    J_pinv = np.linalg.pinv(J)


    # Using redundancy, it is possible to modify a manipulator ’ s configuration without changing the end effector position or orientation. This procedure involves “ internal motions ”restricted to the manipulator ’ s nullspace : 

    
    # joint angles computing

    X_dot = (X_final - X_inital)/tspan[-1]
    print(X_dot)

    

    # add 1 element to X_dot to match the shape of J_pinv
    X_dot = np.append(X_dot,0)

  
    # joint angles update
    q_dot = J_pinv.dot(X_dot)
    q_initial = q_initial + q_dot*t




    # end effector position update
    X_inital = forward_kinematics(q_initial[0],q_initial[1],q_initial[2])

    # plot end effector position
    plt.plot(X_inital[0],X_inital[1],'ro')
   

    plt.plot([0,L1*np.cos(q_initial[0]),L1*np.cos(q_initial[0])+L2*np.cos(q_initial[0]+q_initial[1]),L1*np.cos(q_initial[0])+L2*np.cos(q_initial[0]+q_initial[1])+L3*np.cos(q_initial[0]+q_initial[1]+q_initial[2])],
            [0,L1*np.sin(q_initial[0]),L1*np.sin(q_initial[0])+L2*np.sin(q_initial[0]+q_initial[1]),L1*np.sin(q_initial[0])+L2*np.sin(q_initial[0]+q_initial[1])+L3*np.sin(q_initial[0]+q_initial[1]+q_initial[2])])
    plt.axis([-0.9, 0.9, -0.5, 1.5])

   
    plt.pause(0.01)
    plt.clf()
