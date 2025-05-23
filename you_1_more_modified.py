import sympy as sp
import numpy as np
import pickle
import os

# Define symbolic joint variables
q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5')
joint_symbols = [q1, q2, q3, q4, q5]
DOF = 5

# DH Parameters
DH_params = [
    [q1,         sp.pi/2,  0,     170.5],
    [q2,         0,        83,    0],
    [q3,         0,        83,    0],
    [q4 + sp.pi/2, sp.pi/2, 0,     0],
    [q5,         0,        0,     188.5]
]

PICKLE_PATH = "cached_transformation.pkl"

# --- FORWARD KINEMATICS ---
def dh_transform(theta, alpha, a, d):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),                sp.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def compute_symbolic_fk(DH_params, cache_path=PICKLE_PATH, changed=False):
    if not changed and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            # print("Loaded symbolic transform from cache.")
            return pickle.load(f)

    print("Computing symbolic FK transform...")
    T = sp.eye(4)
    for params in DH_params:
        T *= dh_transform(*params)

    with open(cache_path, 'wb') as f:
        pickle.dump(T, f)
    print("Saved symbolic transform to cache.")
    return T


def forward_kinematics(joint_values, symbolic_transform):
    subs_dict = dict(zip(joint_symbols, joint_values))
    return symbolic_transform.subs(subs_dict)

# --- INVERSE KINEMATICS IMPROVED ---

def joint_limits(joints):
    limits = [(0, sp.pi), (0, sp.pi), (0, sp.pi), (0, sp.pi), (0, 3*sp.pi/2)]
    return np.clip(joints, [l[0] for l in limits], [l[1] for l in limits])


def jacobian_expr(DH_params):
    transforms = [dh_transform(*p) for p in DH_params]
    trans_EF = sp.eye(4)
    for M in transforms:
        trans_EF *= M
    pos_EF = trans_EF[:3, 3]
    J = sp.zeros(6, DOF)
    for i in range(DOF):
        Ti = sp.eye(4)
        for M in transforms[:i+1]: Ti *= M
        z = Ti[:3, 2]
        p = Ti[:3, 3]
        J[:3, i] = z.cross(pos_EF - p)
        J[3:, i] = z
    return sp.simplify(J)


def jacobian_subs(joints, J_sym):
    return np.array(J_sym.subs(dict(zip(joint_symbols, joints))), dtype=np.float64)


def damped_pseudo_inverse(J, lam):
    # SVD-based damped pseudo-inverse: J⁺ = V Σ_damped Uᵀ
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_d = S / (S**2 + lam**2)
    return Vt.T @ np.diag(S_d) @ U.T


def compute_lambda(jac, lam0=1e-3, mu0=1e-3):
    # manipulability measure: sqrt(det(J Jᵀ))
    # JJt = jac @ jac.T
    # mu = np.sqrt(np.linalg.det(JJ[JJt.shape[0] > 0])) if JJt.shape[0]>0 else 0
    JJt = jac @ jac.T
    mu = np.sqrt(np.linalg.det(JJt))   if JJt.shape[0] > 0 else 0
    # adaptive law: increase near singularity
    if mu < mu0:
        return lam0 * (1 + (mu0 - mu)/mu0)
    else:
        return lam0 * 0.1


def i_kine(joints_init, target, DH_params,
           max_iters=2000, pos_tol=1e-3, error_tol=1e-3,
           no_rotation=False, joint_lims_flag=True):
    # initial setup
    joints = np.array(joints_init, dtype=np.float64)
    R_des = np.array(target[:3,:3], dtype=np.float64)
    p_des = np.array(target[:3,3], dtype=np.float64)
    J_sym = jacobian_expr(DH_params)

    for i in range(max_iters):
        J = jacobian_subs(joints, J_sym)
        # current pose
        T_cur = np.array(forward_kinematics(joints, compute_symbolic_fk(DH_params)).evalf(), dtype=np.float64)
        R_cur, p_cur = T_cur[:3,:3], T_cur[:3,3]

        # errors
        dp = (p_des - p_cur).reshape(3,1)
        dR = R_des @ R_cur.T
        theta = np.arccos((np.trace(dR)-1)/2)
        axis = 0.5/np.sin(theta) * np.array([[dR[2,1]-dR[1,2]], [dR[0,2]-dR[2,0]], [dR[1,0]-dR[0,1]]]) if not no_rotation else np.zeros((3,1))
        dR_vec = axis * theta

        xdot = np.vstack((dp, dR_vec))
        if np.linalg.norm(dp) < pos_tol:
            print(f"Position within tolerance after {i} iters.")
            break
        if np.linalg.norm(xdot) < error_tol:
            print(f"Converged in task-space change after {i} iters.")
            break

        # adaptive damping
        lam = compute_lambda(J)
        # compute damped pseudo-inverse
        J_pinv = damped_pseudo_inverse(J, lam)
        # joint update
        dq = J_pinv @ xdot
        joints += dq.flatten()

        if joint_lims_flag:
            joints = joint_limits(joints)

    print("Final pose error:", np.linalg.norm(p_des - p_cur))
    return joints

# --- Example Usage ---
if __name__ == "__main__":
    # Define symbolic joint variables
    q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5')
    joint_symbols = [q1, q2, q3, q4, q5]
    DOF = 5

    # DH Parameters
    DH_params = [
        [q1,         sp.pi/2,  0,     170.5],
        [q2,         0,        83,    0],
        [q3,         0,        83,    0],
        [q4 + sp.pi/2, sp.pi/2, 0,     0],
        [q5,         0,        0,     188.5]
    ]

    PICKLE_PATH = "cached_transformation.pkl"
    initial_position = [0, 0, sp.pi/2, 0, 0]
    symbolic_T = compute_symbolic_fk(DH_params)
    fk_result = forward_kinematics(initial_position, symbolic_T)
    fk_result_eval = fk_result.evalf()
    sp.pprint(fk_result_eval, use_unicode=True)
    
    # Define target pose
    joint_init = [sp.pi/2, 0, 0, 0, 0]
    target = np.array(fk_result_eval)

    new_j = i_kine(joint_init, target, DH_params)
    print("New joint configuration:", new_j)
    new_j_floats = np.array([float(val) for val in new_j])
    joint_degs   = np.degrees(new_j_floats)
    print(joint_degs)
