import numpy as np
import sympy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class RobotIK:
    def __init__(self):
        # Define symbolic variables
        self.q1, self.q2, self.q3, self.q4, self.q5 = sp.symbols('q1 q2 q3 q4 q5')
        
        # DH parameters [theta, alpha, a, d]
        self.DH_params = [
            [self.q1,         sp.pi/2,  0,     170.5],
            [self.q2,         0,        83,    0],
            [self.q3,         0,        83,    0],
            [self.q4 + sp.pi/2, sp.pi/2, 0,     0],
            [self.q5,         0,        0,     188.5]
        ]
        
        # Precompute forward kinematics
        self.T_total = self._compute_forward_kinematics()
        
        # Create numerical functions for optimization
        self._create_numerical_functions()
    
    def _dh_transform(self, theta, alpha, a, d):
        """Create DH transformation matrix"""
        return sp.Matrix([
            [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
            [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
            [0,              sp.sin(alpha),                sp.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])
    
    def _compute_forward_kinematics(self):
        """Compute forward kinematics symbolically"""
        T = sp.eye(4)
        
        for params in self.DH_params:
            theta, alpha, a, d = params
            T_i = self._dh_transform(theta, alpha, a, d)
            T = T * T_i
        
        return sp.simplify(T)
    
    def _create_numerical_functions(self):
        """Create numerical functions for optimization"""
        # Extract position and orientation
        self.pos_x = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                                self.T_total[0, 3], 'numpy')
        self.pos_y = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                                self.T_total[1, 3], 'numpy')
        self.pos_z = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                                self.T_total[2, 3], 'numpy')
        
        # Rotation matrix elements for orientation
        self.R11 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[0, 0], 'numpy')
        self.R12 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[0, 1], 'numpy')
        self.R13 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[0, 2], 'numpy')
        self.R21 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[1, 0], 'numpy')
        self.R22 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[1, 1], 'numpy')
        self.R23 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[1, 2], 'numpy')
        self.R31 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[2, 0], 'numpy')
        self.R32 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[2, 1], 'numpy')
        self.R33 = sp.lambdify([self.q1, self.q2, self.q3, self.q4, self.q5], 
                              self.T_total[2, 2], 'numpy')
    
    def forward_kinematics(self, joint_angles):
        """Compute forward kinematics numerically"""
        q1, q2, q3, q4, q5 = joint_angles
        
        # Position
        pos = np.array([
            self.pos_x(q1, q2, q3, q4, q5),
            self.pos_y(q1, q2, q3, q4, q5),
            self.pos_z(q1, q2, q3, q4, q5)
        ])
        
        # Rotation matrix
        R = np.array([
            [self.R11(q1, q2, q3, q4, q5), self.R12(q1, q2, q3, q4, q5), self.R13(q1, q2, q3, q4, q5)],
            [self.R21(q1, q2, q3, q4, q5), self.R22(q1, q2, q3, q4, q5), self.R23(q1, q2, q3, q4, q5)],
            [self.R31(q1, q2, q3, q4, q5), self.R32(q1, q2, q3, q4, q5), self.R33(q1, q2, q3, q4, q5)]
        ])
        
        return pos, R
    
    def _cost_function(self, joint_angles, target_pos, target_R=None, pos_weight=1.0, rot_weight=0.1):
        """Cost function for optimization"""
        try:
            current_pos, current_R = self.forward_kinematics(joint_angles)
            
            # Position error
            pos_error = np.sum((current_pos - target_pos)**2)
            
            # Orientation error (if target orientation is provided)
            rot_error = 0
            if target_R is not None:
                # Use Frobenius norm of rotation matrix difference
                rot_error = np.sum((current_R - target_R)**2)
            
            return pos_weight * pos_error + rot_weight * rot_error
            
        except:
            return 1e6  # Return large error if calculation fails
    
    def inverse_kinematics_position_only(self, target_pos, initial_guess=None, bounds=None):
        """Solve IK for position only (3 constraints, 5 DOF - underdetermined)"""
        if initial_guess is None:
            initial_guess = np.zeros(5)
        
        if bounds is None:
            # Default joint limits (adjust based on your robot)
            bounds = [(-np.pi, np.pi)] * 5
        
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(target_pos,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        return result.x, result.success, result.fun
    
    def inverse_kinematics_full(self, target_pos, target_R, initial_guess=None, bounds=None):
        """Solve IK for full pose (6 constraints, 5 DOF - overdetermined)"""
        if initial_guess is None:
            initial_guess = np.zeros(5)
        
        if bounds is None:
            bounds = [(-np.pi, np.pi)] * 5
        
        result = minimize(
            self._cost_function,
            initial_guess,
            args=(target_pos, target_R),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        return result.x, result.success, result.fun
    
    def solve_multiple_solutions(self, target_pos, target_R=None, num_attempts=10):
        """Try multiple random initial guesses to find different solutions"""
        solutions = []
        
        for _ in range(num_attempts):
            # Random initial guess
            initial_guess = np.random.uniform(-np.pi, np.pi, 5)
            
            if target_R is None:
                angles, success, cost = self.inverse_kinematics_position_only(target_pos, initial_guess)
            else:
                angles, success, cost = self.inverse_kinematics_full(target_pos, target_R, initial_guess)
            
            if success and cost < 1e-6:
                # Check if this solution is significantly different from existing ones
                is_new = True
                for existing_sol, _, _ in solutions:
                    if np.allclose(angles, existing_sol, atol=0.1):
                        is_new = False
                        break
                
                if is_new:
                    solutions.append((angles, success, cost))
        
        return solutions

# Example usage
def main():
    # Create robot IK solver
    robot = RobotIK()
    
    # Example 1: Position-only IK
    print("=== Position-Only Inverse Kinematics ===")
    target_position = np.array([100, 100, 200])  # Target position in mm
    
    joint_angles, success, error = robot.inverse_kinematics_position_only(target_position)
    
    if success:
        print(f"Solution found!")
        print(f"Joint angles (rad): {joint_angles}")
        print(f"Joint angles (deg): {np.degrees(joint_angles)}")
        print(f"Error: {error}")
        
        # Verify solution
        achieved_pos, _ = robot.forward_kinematics(joint_angles)
        print(f"Target position: {target_position}")
        print(f"Achieved position: {achieved_pos}")
        print(f"Position error: {np.linalg.norm(achieved_pos - target_position)}")
    else:
        print("No solution found")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Find multiple solutions
    print("=== Multiple Solutions ===")
    solutions = robot.solve_multiple_solutions(target_position, num_attempts=20)
    
    print(f"Found {len(solutions)} different solutions:")
    for i, (angles, success, cost) in enumerate(solutions):
        print(f"Solution {i+1}: {np.degrees(angles)} (degrees)")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Full pose IK (position + orientation)
    print("=== Full Pose Inverse Kinematics ===")
    target_R = np.eye(3)  # Identity rotation (no rotation)
    
    joint_angles, success, error = robot.inverse_kinematics_full(target_position, target_R)
    
    if success:
        print(f"Solution found!")
        print(f"Joint angles (deg): {np.degrees(joint_angles)}")
        print(f"Error: {error}")
        
        # Verify solution
        achieved_pos, achieved_R = robot.forward_kinematics(joint_angles)
        print(f"Position error: {np.linalg.norm(achieved_pos - target_position)}")
        print(f"Rotation error: {np.linalg.norm(achieved_R - target_R)}")
    else:
        print("No solution found for full pose")

if __name__ == "__main__":
    main()