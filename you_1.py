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
            print("Loaded symbolic transform from cache.")
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

# --- INVERSE KINEMATICS ---
def joint_limits(joints):
    limits = [(0, sp.pi), (0, sp.pi), (0, sp.pi), (0, sp.pi), (0, 3*sp.pi/2)]
    return [max(min(joint, lim[1]), lim[0]) for joint, lim in zip(joints, limits)]

def joint_transforms(DH_params):
    return [dh_transform(*params) for params in DH_params]

def trans_EF_eval(joints, DH_params):
    T = sp.eye(4)
    subs_dict = dict(zip(joint_symbols, joints))
    for params in DH_params:
        T *= dh_transform(*params)
    return T.subs(subs_dict)

def jacobian_expr(DH_params):
    transforms = joint_transforms(DH_params)
    trans_EF = sp.eye(4)
    for mat in transforms:
        trans_EF *= mat
    pos_EF = trans_EF[:3, 3]
    J = sp.zeros(6, DOF)

    for i in range(DOF):
        trans_i = sp.eye(4)
        for mat in transforms[:i+1]:
            trans_i *= mat
        z = trans_i[:3, 2]
        p = trans_i[:3, 3]
        J[:3, i] = z.cross(pos_EF - p)
        J[3:, i] = z

    return sp.simplify(J)

def jacobian_subs(joints, jacobian_sym):
    subs = dict(zip(joint_symbols, joints))
    return jacobian_sym.subs(subs)

def compute_adaptive_damping(jac):
    """Compute adaptive damping factor based on Jacobian condition number"""
    s = np.linalg.svd(jac, compute_uv=False)
    condition_number = s[0] / s[-1]
    Lambda_min = 5.0
    Lambda_max = 50.0
    return np.clip(condition_number / 100, Lambda_min, Lambda_max)

def update_joints(jac, x_dot, joints, Lambda):
    """Stable joint update with error handling"""
    try:
        joint_change = np.linalg.solve(
            jac.T @ jac + Lambda**2 * np.eye(DOF), 
            jac.T @ x_dot
        )
        return joints + joint_change.flatten()
    except np.linalg.LinAlgError:
        print("Warning: Singular configuration encountered")
        return joints

def validate_ik_solution(joints, target_pose, DH_params):
    """Validate IK solution against target pose"""
    current_pose = forward_kinematics(joints, DH_params)
    pos_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])
    rot_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3], 'fro')
    
    return pos_error < 1e-3 and rot_error < 1e-2

def i_kine(joints_init, target, DH_params, max_iters=2000, error_threshold=1e-3, no_rotation=False, joint_lims=True):
    joints = np.array(joints_init, dtype=np.float64)
    xr_desired = np.array(target[:3, :3], dtype=np.float64)
    xt_desired = np.array(target[:3, 3], dtype=np.float64)

    x_dot_prev = np.zeros((6, 1))
    iters = 0

    print("Finding symbolic jacobian")
    jacobian_symbolic = jacobian_expr(DH_params)

    print("Starting IK loop")
    while iters < max_iters:
        jac = np.array(jacobian_subs(joints, jacobian_symbolic)).astype(np.float64)
        trans_EF_cur = np.array(trans_EF_eval(joints, DH_params)).astype(np.float64)

        xr_cur = trans_EF_cur[:3, :3]
        xt_cur = trans_EF_cur[:3, 3]
        xt_dot = (xt_desired - xt_cur).reshape((3, 1))

        R = xr_desired @ xr_cur.T
        v = np.arccos((np.trace(R) - 1) / 2)
        r = 0.5 * np.sin(v) * np.array([[R[2,1]-R[1,2]], [R[0,2]-R[2,0]], [R[1,0]-R[0,1]]])
        xr_dot = 0 * r if no_rotation else 200 * r * np.sin(v)

        x_dot = np.vstack((xt_dot, xr_dot))
        x_dot_norm = np.linalg.norm(x_dot)

        if x_dot_norm > 25:
            x_dot *= 25 / x_dot_norm

        if np.linalg.norm(x_dot - x_dot_prev) < error_threshold:
            print(f"Converged with position error: {np.linalg.norm(xt_dot):.6f}")
            break

        x_dot_prev = x_dot

        Lambda = compute_adaptive_damping(jac)
        joints = update_joints(jac, x_dot, joints, Lambda)

        if joint_lims:
            joints = joint_limits(joints)

        iters += 1

    print(f"Done in {iters} iterations")
    print("Final position is:", xt_cur)

    return joints

# --- Example Usage ---
if __name__ == "__main__":
    initial_position = [0, 0, sp.pi/2, 0, 0]
    symbolic_T = compute_symbolic_fk(DH_params)
    fk_result = forward_kinematics(initial_position, symbolic_T)
    fk_result_eval = fk_result.evalf()
    sp.pprint(fk_result_eval, use_unicode=True)
    
    # Define target pose
    joint_init = [sp.pi/2, 0, 0, 0, 0]
    target = np.array(fk_result_eval)

    new_j = i_kine(joint_init, target, DH_params, error_threshold=1e-3, joint_lims=False)
    print("New joint configuration:", new_j)
