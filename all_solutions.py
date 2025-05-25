def analytical_ik(target_pose, DH_params):
    """Analytical inverse kinematics for 5-DOF robot arm"""
    # Extract target position and orientation
    R_des = target_pose[:3,:3]
    p_des = target_pose[:3,3]
    
    # Extract DH parameters
    d1 = DH_params[0][3]  # 170.5
    a2 = DH_params[1][2]  # 83
    a3 = DH_params[2][2]  # 83
    d5 = DH_params[4][3]  # 188.5
    
    # Step 1: Solve for q1 (base rotation)
    # Project end effector position onto XY plane
    q1 = np.arctan2(p_des[1], p_des[0])
    
    # Step 2: Transform target to q1 frame
    R1 = np.array([
        [np.cos(q1), -np.sin(q1), 0],
        [np.sin(q1), np.cos(q1), 0],
        [0, 0, 1]
    ])
    
    # Wrist position (subtract tool length from end effector)
    wrist_vec = np.array([0, 0, -d5, 1])
    wrist_in_ee = np.dot(target_pose, wrist_vec)[:3]
    
    # Step 3: Solve for q2 and q3 using geometric approach
    # Distance from base to wrist in XY plane
    r = np.sqrt(wrist_in_ee[0]**2 + wrist_in_ee[1]**2)
    s = wrist_in_ee[2] - d1  # Height from base
    
    # Distance from joint 2 to wrist
    D = np.sqrt(r**2 + s**2)
    
    # Check if target is reachable
    if D > (a2 + a3) or D < abs(a2 - a3):
        print(f"Target position unreachable: {D} vs arm length {a2+a3}")
        return None
    
    # Law of cosines
    cos_q3 = (D**2 - a2**2 - a3**2) / (2 * a2 * a3)
    if cos_q3 < -1 or cos_q3 > 1:
        print("Cosine out of range, target unreachable")
        return None
        
    q3 = np.arccos(cos_q3)  # Elbow angle
    
    # Shoulder angle
    alpha = np.arctan2(s, r)  # Angle to wrist
    beta = np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))  # Angle in arm triangle
    q2 = alpha - beta
    
    # Step 4: Solve for wrist angles q4 and q5
    # Rotation from base to wrist
    R0_3 = np.array([
        [np.cos(q1) * np.cos(q2+q3), -np.cos(q1) * np.sin(q2+q3), np.sin(q1)],
        [np.sin(q1) * np.cos(q2+q3), -np.sin(q1) * np.sin(q2+q3), -np.cos(q1)],
        [np.sin(q2+q3), np.cos(q2+q3), 0]
    ])
    
    # Desired wrist rotation
    R3_6 = R0_3.T @ R_des
    
    # Extract Euler angles
    q4 = np.arctan2(R3_6[1,2], R3_6[0,2]) - np.pi/2
    q5 = np.arctan2(np.sqrt(R3_6[0,2]**2 + R3_6[1,2]**2), R3_6[2,2])
    
    # Return joint angles
    return np.array([q1, q2, q3, q4, q5])

def ccd_ik(joints_init, target_pose, DH_params, max_iters=100, tol=1e-3):
    """Cyclic Coordinate Descent IK solver"""
    # Initialize joints
    joints = np.array(joints_init, dtype=np.float64)
    symbolic_T = compute_symbolic_fk(DH_params)
    
    # Target position
    p_target = target_pose[:3, 3]
    
    # Extract joint positions and rotation axes
    def get_joint_poses(joints):
        poses = []
        axes = []
        T = np.eye(4)
        
        # Forward pass to get all joint positions and axes
        for i, (joint, dh) in enumerate(zip(joints, DH_params)):
            # Create DH transform with current joint value
            dh_params = dh.copy()
            dh_params[0] = float(joint)  # Replace symbolic with value
            Ti = np.array(dh_transform(*dh_params).evalf(), dtype=np.float64)
            T = T @ Ti
            
            # Store joint position and z-axis (rotation axis)
            poses.append(T[:3, 3].copy())
            axes.append(T[:3, 2].copy())  # z-axis is rotation axis
        
        return poses, axes
    
    for iter in range(max_iters):
        # Get current end effector position
        T_current = np.array(forward_kinematics(joints, symbolic_T).evalf(), dtype=np.float64)
        p_current = T_current[:3, 3]
        
        # Check convergence
        error = np.linalg.norm(p_target - p_current)
        if error < tol:
            print(f"CCD converged after {iter} iterations, error: {error:.6f}mm")
            return joints
            
        if iter % 10 == 0:
            print(f"Iteration {iter}, error: {error:.6f}mm")
        
        # Get joint positions and axes
        joint_positions, joint_axes = get_joint_poses(joints)
        
        # Iterate through joints in reverse order (from end effector to base)
        for i in range(len(joints)-1, -1, -1):
            # Current joint position and axis
            p_joint = joint_positions[i]
            axis = joint_axes[i]
            
            # Vectors from joint to current end effector and target
            v_current = p_current - p_joint
            v_target = p_target - p_joint
            
            # Skip if vectors are too small
            if np.linalg.norm(v_current) < 1e-6 or np.linalg.norm(v_target) < 1e-6:
                continue
                
            # Normalize vectors
            v_current = v_current / np.linalg.norm(v_current)
            v_target = v_target / np.linalg.norm(v_target)
            
            # Compute rotation angle using dot and cross product
            cos_angle = np.clip(np.dot(v_current, v_target), -1.0, 1.0)
            sin_angle = np.linalg.norm(np.cross(v_current, v_target))
            angle = np.arctan2(sin_angle, cos_angle)
            
            # Determine rotation direction
            cross_prod = np.cross(v_current, v_target)
            if np.linalg.norm(cross_prod) > 0:
                cross_prod = cross_prod / np.linalg.norm(cross_prod)
                if np.dot(cross_prod, axis) < 0:
                    angle = -angle
            
            # Update joint angle
            joints[i] += angle
            
            # Apply joint limits
            joints = joint_limits(joints)
            
            # Update end effector position after this joint change
            T_current = np.array(forward_kinematics(joints, symbolic_T).evalf(), dtype=np.float64)
            p_current = T_current[:3, 3]
    
    print(f"CCD reached max iterations, final error: {error:.6f}mm")
    return joints

def optimization_ik(joints_init, target_pose, DH_params, method='L-BFGS-B'):
    """Optimization-based inverse kinematics"""
    from scipy.optimize import minimize
    
    # Convert initial joints to float array
    joints_init = np.array([float(j) for j in joints_init], dtype=np.float64)
    
    # Target position and orientation
    p_target = target_pose[:3, 3]
    R_target = target_pose[:3, :3]
    
    # Pre-compute symbolic transform
    symbolic_T = compute_symbolic_fk(DH_params)
    
    # Cost function: position and orientation error
    def cost_function(joints):
        # Forward kinematics
        T = np.array(forward_kinematics(joints, symbolic_T).evalf(), dtype=np.float64)
        p_current = T[:3, 3]
        R_current = T[:3, :3]
        
        # Position error
        pos_error = np.linalg.norm(p_target - p_current)
        
        # Orientation error (Frobenius norm)
        R_error = R_target @ R_current.T
        orient_error = np.linalg.norm(R_error - np.eye(3), 'fro')
        
        # Combined error with weight
        return pos_error + 0.1 * orient_error
    
    # Joint limits
    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi), (0, 3*np.pi/2)]
    
    # Run optimization
    print("Starting optimization-based IK...")
    result = minimize(cost_function, joints_init, method=method, bounds=bounds, 
                     options={'maxiter': 1000, 'disp': True})
    
    # Check result
    if result.success:
        print(f"Optimization converged: {result.message}")
        final_error = cost_function(result.x)
        print(f"Final error: {final_error:.6f}")
    else:
        print(f"Optimization failed: {result.message}")
    
    return result.x

def multi_start_ik(target_pose, DH_params, method='jacobian'):
    """Try multiple initial guesses and return best result"""
    # Generate diverse initial guesses
    initial_guesses = [
        [np.pi/2, 0, 0, 0, 0],
        [0, np.pi/4, np.pi/4, 0, 0],
        [np.pi/4, np.pi/4, np.pi/4, 0, 0],
        [0, np.pi/2, np.pi/2, 0, 0],
        [np.pi/4, np.pi/2, 0, np.pi/4, 0],
        [0, 0, 0, 0, 0],
        [np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3],
    ]
    
    best_joints = None
    best_error = float('inf')
    symbolic_T = compute_symbolic_fk(DH_params)
    
    for i, guess in enumerate(initial_guesses):
        print(f"\nTrying initial guess {i+1}/{len(initial_guesses)}...")
        
        # Choose IK method
        if method == 'jacobian':
            joints = i_kine(guess, target_pose, DH_params, max_iters=1000, 
                          pos_tol=1e-2, step_size=0.2)
        elif method == 'ccd':
            joints = ccd_ik(guess, target_pose, DH_params)
        elif method == 'optimization':
            joints = optimization_ik(guess, target_pose, DH_params)
        elif method == 'analytical':
            joints = analytical_ik(target_pose, DH_params)
            if joints is None:  # Skip if analytical solution fails
                continue
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate result
        T_result = np.array(forward_kinematics(joints, symbolic_T).evalf(), dtype=np.float64)
        error = np.linalg.norm(target_pose[:3,3] - T_result[:3,3])
        
        print(f"Initial guess {i+1} result: error = {error:.6f}mm")
        
        if error < best_error:
            best_error = error
            best_joints = joints
            print(f"New best solution found! Error: {best_error:.6f}mm")
            
            # Early termination if error is small enough
            if error < 1.0:  # 1mm tolerance
                print("Solution is good enough, stopping early.")
                break
    
    print(f"\nBest solution found with error: {best_error:.6f}mm")
    print(f"Joint angles (rad): {best_joints}")
    print(f"Joint angles (deg): {np.degrees(np.array([float(j) for j in best_joints]))}")
    
    return best_joints

#Usage Example 
if __name__ == "__main__":
    # ... existing code ...
    
    # Define target pose
    initial_position = [0, 0, np.pi/2, 0, 0]  # Use numpy's pi
    symbolic_T = compute_symbolic_fk(DH_params)
    fk_result = forward_kinematics(initial_position, symbolic_T)
    fk_result_eval = fk_result.evalf()
    sp.pprint(fk_result_eval, use_unicode=True)
    
    target = np.array(fk_result_eval)
    
    print("\n=== Trying different IK methods ===")
    
    # Try analytical IK first (fastest)
    print("\n1. Analytical IK:")
    try:
        analytical_joints = analytical_ik(target, DH_params)
        if analytical_joints is not None:
            print("Analytical solution found!")
            print(f"Joint angles (rad): {analytical_joints}")
            print(f"Joint angles (deg): {np.degrees(analytical_joints)}")
            
            # Verify solution
            T_result = np.array(forward_kinematics(analytical_joints, symbolic_T).evalf(), dtype=np.float64)
            error = np.linalg.norm(target[:3,3] - T_result[:3,3])
            print(f"Position error: {error:.6f}mm")
    except Exception as e:
        print(f"Analytical IK failed: {e}")
    
    # Try CCD method
    print("\n2. CCD IK:")
    try:
        ccd_joints = ccd_ik([np.pi/2, 0, 0, 0, 0], target, DH_params)
        print(f"Joint angles (rad): {ccd_joints}")
        print(f"Joint angles (deg): {np.degrees(ccd_joints)}")
        
        # Verify solution
        T_result = np.array(forward_kinematics(ccd_joints, symbolic_T).evalf(), dtype=np.float64)
        error = np.linalg.norm(target[:3,3] - T_result[:3,3])
        print(f"Position error: {error:.6f}mm")
    except Exception as e:
        print(f"CCD IK failed: {e}")
    
    # Try optimization method
    print("\n3. Optimization IK:")
    try:
        opt_joints = optimization_ik([np.pi/2, 0, 0, 0, 0], target, DH_params)
        print(f"Joint angles (rad): {opt_joints}")
        print(f"Joint angles (deg): {np.degrees(opt_joints)}")
        
        # Verify solution
        T_result = np.array(forward_kinematics(opt_joints, symbolic_T).evalf(), dtype=np.float64)
        error = np.linalg.norm(target[:3,3] - T_result[:3,3])
        print(f"Position error: {error:.6f}mm")
    except Exception as e:
        print(f"Optimization IK failed: {e}")
    
    # Try multi-start approach
    print("\n4. Multi-start approach:")
    try:
        best_joints = multi_start_ik(target, DH_params, method='jacobian')
    except Exception as e:
        print(f"Multi-start approach failed: {e}")

'''
## Which Method to Choose?
1. Analytical IK : Fastest and most accurate when available, but requires careful derivation for your specific robot
2. CCD : Simple and robust for position-only IK, but may struggle with orientation constraints
3. Optimization-based : Most flexible and can handle various constraints, but slower
4. Multi-start approach : Most reliable but computationally expensive
I recommend trying the analytical solution first, then falling back to the other methods if needed. The multi-start approach with different methods can give you the most robust results.

Let me know if you'd like me to explain any of these methods in more detail!
'''