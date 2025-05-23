import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# Define joint symbols
q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5')
joint_symbols = [q1, q2, q3, q4, q5]

# Define DH parameters (theta, alpha, r, d)
DH_params = [
    [q1,         sp.pi/2,  0,     170.5],
    [q2,         0,        83,    0],
    [q3,         0,        83,    0],
    [q4 + sp.pi/2, sp.pi/2, 0,     0],
    [q5,         0,        0,     188.5]
]

# Function to compute a transformation matrix from DH parameters
def dh_transform(theta, alpha, r, d):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), r*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), r*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

# Compute symbolic transformation
def compute_symbolic_fk(DH_params):
    T = sp.eye(4)
    for param in DH_params:
        T_i = dh_transform(*param)
        T = T * T_i
    return T

symbolic_T = compute_symbolic_fk(DH_params)

# Function to evaluate FK numerically
def forward_kinematics(joint_vals, symbolic_T, joint_symbols):
    subs = dict(zip(joint_symbols, joint_vals))
    T_numeric = symbolic_T.evalf(subs=subs)
    return np.array(T_numeric).astype(np.float64)

# Plotting function
def plot_fk(q1_val, q2_val, q3_val, q4_val, q5_val):
    joint_vals = [q1_val, q2_val, q3_val, q4_val, q5_val]
    T = forward_kinematics(joint_vals, symbolic_T, joint_symbols)

    # Extract position
    x, y, z = T[:3, 3]
    
    # Simple 3D plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='red', label='End Effector')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(0, 500)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("FK Visualization")
    plt.show()

# Create interactive sliders
interact(plot_fk,
         q1_val=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0),
         q2_val=FloatSlider(min=-np.pi/2, max=np.pi/2, step=0.1, value=0),
         q3_val=FloatSlider(min=-np.pi/2, max=np.pi/2, step=0.1, value=0),
         q4_val=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0),
         q5_val=FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=0)
)
