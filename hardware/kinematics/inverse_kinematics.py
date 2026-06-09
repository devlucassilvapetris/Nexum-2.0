"""
Nexum v2.0 Inverse Kinematics
Advanced inverse kinematics for complex robotic movements
"""

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time


class SolverMethod(Enum):
    JACOBIAN_PSEUDO_INVERSE = "jacobian_pseudo_inverse"
    JACOBIAN_TRANSPOSE = "jacobian_transpose"
    DLS = "damped_least_squares"
    CCD = "cyclic_coordinate_descent"
    ANALYTICAL = "analytical"


@dataclass
class JointConfig:
    id: str
    name: str
    min_angle: float  # radians
    max_angle: float  # radians
    max_velocity: float  # rad/s
    max_acceleration: float  # rad/s²
    home_position: float  # radians


@dataclass
class LinkConfig:
    id: str
    name: str
    length: float  # meters
    mass: float  # kg
    center_of_mass: Tuple[float, float, float]  # relative to joint
    inertia: Tuple[float, float, float]  # principal moments of inertia


@dataclass
class EndEffectorTarget:
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Optional[Tuple[float, float, float]]  # roll, pitch, yaw in radians
    velocity: Optional[Tuple[float, float, float]] = None
    acceleration: Optional[Tuple[float, float, float]] = None


@dataclass
class IKSolution:
    joint_angles: Dict[str, float]
    success: bool
    iterations: int
    error: float
    computation_time: float
    method: SolverMethod


class ForwardKinematics:
    """Forward kinematics for computing end-effector position from joint angles"""
    
    def __init__(self, links: List[LinkConfig]):
        self.links = links
        self.logger = logging.getLogger(__name__)
    
    def compute_end_effector(self, joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute end-effector position and orientation from joint angles
        Returns: (position, orientation_matrix)
        """
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.eye(3)
        
        current_position = np.array([0.0, 0.0, 0.0])
        current_orientation = np.eye(3)
        
        for i, (link, joint_id) in enumerate(zip(self.links, joint_angles.keys())):
            angle = joint_angles[joint_id]
            
            # Compute transformation matrix for this joint
            rotation_matrix = self._rotation_matrix(i, angle)
            
            # Update orientation
            current_orientation = current_orientation @ rotation_matrix
            
            # Compute link vector in global frame
            link_vector = current_orientation @ np.array([link.length, 0, 0])
            
            # Update position
            current_position = current_position + link_vector
        
        position = current_position
        orientation = current_orientation
        
        return position, orientation
    
    def _rotation_matrix(self, joint_index: int, angle: float) -> np.ndarray:
        """Compute rotation matrix for joint"""
        # Simplified: alternating rotation axes
        axis = joint_index % 3
        
        if axis == 0:  # X-axis rotation
            return np.array([
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]
            ])
        elif axis == 1:  # Y-axis rotation
            return np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]
            ])
        else:  # Z-axis rotation
            return np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ])
    
    def compute_jacobian(self, joint_angles: Dict[str, float]) -> np.ndarray:
        """
        Compute Jacobian matrix for current configuration
        Jacobian relates joint velocities to end-effector velocities
        """
        num_joints = len(joint_angles)
        jacobian = np.zeros((6, num_joints))  # 6 DOF end-effector, n joints
        
        # Small perturbation for numerical differentiation
        epsilon = 1e-6
        
        # Compute current end-effector state
        position, orientation = self.compute_end_effector(joint_angles)
        
        # Compute Jacobian columns
        for i, joint_id in enumerate(joint_angles.keys()):
            # Perturb joint angle
            perturbed_angles = joint_angles.copy()
            perturbed_angles[joint_id] += epsilon
            
            # Compute perturbed end-effector state
            perturbed_position, perturbed_orientation = self.compute_end_effector(perturbed_angles)
            
            # Compute partial derivatives
            position_derivative = (perturbed_position - position) / epsilon
            
            # For orientation, use small angle approximation
            orientation_diff = perturbed_orientation @ orientation.T
            angle_diff = np.array([
                orientation_diff[2, 1],
                orientation_diff[0, 2],
                orientation_diff[1, 0]
            ])
            orientation_derivative = angle_diff / epsilon
            
            # Fill Jacobian column
            jacobian[:3, i] = position_derivative
            jacobian[3:, i] = orientation_derivative
        
        return jacobian


class InverseKinematics:
    """
    Advanced inverse kinematics solver for complex robotic movements
    Supports multiple solution methods and constraints
    """
    
    def __init__(self, 
                 joints: List[JointConfig], 
                 links: List[LinkConfig],
                 method: SolverMethod = SolverMethod.DLS):
        self.joints = joints
        self.links = links
        self.method = method
        self.fk = ForwardKinematics(links)
        self.logger = logging.getLogger(__name__)
        
        # Solver parameters
        self.max_iterations = 100
        self.tolerance = 1e-3
        self.damping_factor = 0.01
        self.learning_rate = 0.1
        
        # Current joint angles
        self.current_angles = {joint.id: joint.home_position for joint in joints}
    
    def solve(self, target: EndEffectorTarget, 
              initial_angles: Optional[Dict[str, float]] = None) -> IKSolution:
        """
        Solve inverse kinematics for target end-effector pose
        """
        start_time = time.time()
        
        # Set initial angles
        if initial_angles:
            self.current_angles = initial_angles.copy()
        
        # Select solver method
        if self.method == SolverMethod.JACOBIAN_PSEUDO_INVERSE:
            solution = self._solve_jacobian_pseudo_inverse(target)
        elif self.method == SolverMethod.JACOBIAN_TRANSPOSE:
            solution = self._solve_jacobian_transpose(target)
        elif self.method == SolverMethod.DLS:
            solution = self._solve_damped_least_squares(target)
        elif self.method == SolverMethod.CCD:
            solution = self._solve_ccd(target)
        elif self.method == SolverMethod.ANALYTICAL:
            solution = self._solve_analytical(target)
        else:
            solution = self._solve_damped_least_squares(target)
        
        solution.computation_time = time.time() - start_time
        
        return solution
    
    def _solve_jacobian_pseudo_inverse(self, target: EndEffectorTarget) -> IKSolution:
        """Solve using Jacobian pseudo-inverse method"""
        target_position = np.array(target.position)
        iterations = 0
        error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector position
            current_position, _ = self.fk.compute_end_effector(self.current_angles)
            
            # Compute error
            error_vector = target_position - current_position
            error = np.linalg.norm(error_vector)
            
            if error < self.tolerance:
                break
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(self.current_angles)
            
            # Compute pseudo-inverse
            jacobian_pinv = np.linalg.pinv(jacobian[:3, :])  # Position only
            
            # Compute joint angle updates
            delta_angles = jacobian_pinv @ error_vector
            
            # Update joint angles
            self._update_joint_angles(delta_angles)
            
            iterations += 1
        
        success = error < self.tolerance
        return IKSolution(
            joint_angles=self.current_angles.copy(),
            success=success,
            iterations=iterations,
            error=error,
            computation_time=0.0,
            method=self.method
        )
    
    def _solve_jacobian_transpose(self, target: EndEffectorTarget) -> IKSolution:
        """Solve using Jacobian transpose method"""
        target_position = np.array(target.position)
        iterations = 0
        error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector position
            current_position, _ = self.fk.compute_end_effector(self.current_angles)
            
            # Compute error
            error_vector = target_position - current_position
            error = np.linalg.norm(error_vector)
            
            if error < self.tolerance:
                break
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(self.current_angles)
            
            # Compute joint angle updates using transpose
            delta_angles = self.learning_rate * jacobian[:3, :].T @ error_vector
            
            # Update joint angles
            self._update_joint_angles(delta_angles)
            
            iterations += 1
        
        success = error < self.tolerance
        return IKSolution(
            joint_angles=self.current_angles.copy(),
            success=success,
            iterations=iterations,
            error=error,
            computation_time=0.0,
            method=self.method
        )
    
    def _solve_damped_least_squares(self, target: EndEffectorTarget) -> IKSolution:
        """Solve using damped least squares (Levenberg-Marquardt)"""
        target_position = np.array(target.position)
        iterations = 0
        error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector position
            current_position, _ = self.fk.compute_end_effector(self.current_angles)
            
            # Compute error
            error_vector = target_position - current_position
            error = np.linalg.norm(error_vector)
            
            if error < self.tolerance:
                break
            
            # Compute Jacobian
            jacobian = self.fk.compute_jacobian(self.current_angles)
            
            # Damped least squares: (J^T J + λ^2 I)^-1 J^T e
            J = jacobian[:3, :]
            J_T = J.T
            damping = self.damping_factor * (1 + error)
            
            # Compute regularized inverse
            matrix = J_T @ J + damping**2 * np.eye(J.shape[1])
            delta_angles = np.linalg.inv(matrix) @ J_T @ error_vector
            
            # Update joint angles
            self._update_joint_angles(delta_angles)
            
            iterations += 1
        
        success = error < self.tolerance
        return IKSolution(
            joint_angles=self.current_angles.copy(),
            success=success,
            iterations=iterations,
            error=error,
            computation_time=0.0,
            method=self.method
        )
    
    def _solve_ccd(self, target: EndEffectorTarget) -> IKSolution:
        """Solve using Cyclic Coordinate Descent"""
        target_position = np.array(target.position)
        iterations = 0
        error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector position
            current_position, _ = self.fk.compute_end_effector(self.current_angles)
            
            # Compute error
            error = np.linalg.norm(target_position - current_position)
            
            if error < self.tolerance:
                break
            
            # Iterate through joints from end to base
            for i in reversed(range(len(self.joints))):
                joint_id = self.joints[i].id
                
                # Compute position of this joint
                joint_position = self._compute_joint_position(joint_id)
                
                # Compute vector from joint to end-effector
                end_effector_position, _ = self.fk.compute_end_effector(self.current_angles)
                to_end = end_effector_position - joint_position
                
                # Compute vector from joint to target
                to_target = target_position - joint_position
                
                # Compute angle to align vectors
                if np.linalg.norm(to_end) > 1e-6 and np.linalg.norm(to_target) > 1e-6:
                    # Normalize vectors
                    to_end_norm = to_end / np.linalg.norm(to_end)
                    to_target_norm = to_target / np.linalg.norm(to_target)
                    
                    # Compute rotation angle using cross product
                    cross = np.cross(to_end_norm, to_target_norm)
                    dot = np.dot(to_end_norm, to_target_norm)
                    
                    angle = math.atan2(np.linalg.norm(cross), dot)
                    
                    # Determine direction
                    if cross[2] < 0:
                        angle = -angle
                    
                    # Update joint angle
                    self.current_angles[joint_id] += angle
                    
                    # Clamp to limits
                    joint_config = self.joints[i]
                    self.current_angles[joint_id] = max(
                        min(self.current_angles[joint_id], joint_config.max_angle),
                        joint_config.min_angle
                    )
            
            iterations += 1
        
        # Final error check
        current_position, _ = self.fk.compute_end_effector(self.current_angles)
        error = np.linalg.norm(target_position - current_position)
        success = error < self.tolerance
        
        return IKSolution(
            joint_angles=self.current_angles.copy(),
            success=success,
            iterations=iterations,
            error=error,
            computation_time=0.0,
            method=self.method
        )
    
    def _solve_analytical(self, target: EndEffectorTarget) -> IKSolution:
        """Solve using analytical method (for simple 2-3 DOF cases)"""
        # This is a simplified analytical solution for a 2-link planar arm
        # For more complex systems, numerical methods are preferred
        
        if len(self.joints) != 2:
            # Fall back to DLS for complex systems
            return self._solve_damped_least_squares(target)
        
        target_position = np.array(target.position)
        L1 = self.links[0].length
        L2 = self.links[1].length
        
        # Distance to target
        r = np.linalg.norm(target_position)
        
        # Check if target is reachable
        if r > L1 + L2:
            error = r - (L1 + L2)
            return IKSolution(
                joint_angles=self.current_angles.copy(),
                success=False,
                iterations=0,
                error=error,
                computation_time=0.0,
                method=self.method
            )
        
        # Compute joint angles using law of cosines
        # Angle for second joint
        cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = max(-1, min(1, cos_theta2))  # Clamp to valid range
        theta2 = math.acos(cos_theta2)
        
        # Angle for first joint
        theta1 = math.atan2(target_position[1], target_position[0])
        beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
        theta1 -= beta
        
        # Update joint angles
        self.current_angles[self.joints[0].id] = theta1
        self.current_angles[self.joints[1].id] = theta2
        
        # Compute error
        current_position, _ = self.fk.compute_end_effector(self.current_angles)
        error = np.linalg.norm(target_position - current_position)
        
        return IKSolution(
            joint_angles=self.current_angles.copy(),
            success=True,
            iterations=1,
            error=error,
            computation_time=0.0,
            method=self.method
        )
    
    def _compute_joint_position(self, joint_id: str) -> np.ndarray:
        """Compute global position of a specific joint"""
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.eye(3)
        
        for i, (link, current_joint_id) in enumerate(zip(self.links, self.current_angles.keys())):
            if current_joint_id == joint_id:
                break
            
            angle = self.current_angles[current_joint_id]
            rotation_matrix = self.fk._rotation_matrix(i, angle)
            orientation = orientation @ rotation_matrix
            link_vector = orientation @ np.array([link.length, 0, 0])
            position = position + link_vector
        
        return position
    
    def _update_joint_angles(self, delta_angles: np.ndarray):
        """Update joint angles with constraints"""
        for i, joint_id in enumerate(self.current_angles.keys()):
            if i < len(delta_angles):
                self.current_angles[joint_id] += delta_angles[i]
                
                # Clamp to joint limits
                joint_config = self.joints[i]
                self.current_angles[joint_id] = max(
                    min(self.current_angles[joint_id], joint_config.max_angle),
                    joint_config.min_angle
                )
    
    def set_solver_method(self, method: SolverMethod):
        """Change solver method"""
        self.method = method
    
    def set_solver_parameters(self, 
                            max_iterations: Optional[int] = None,
                            tolerance: Optional[float] = None,
                            damping_factor: Optional[float] = None,
                            learning_rate: Optional[float] = None):
        """Set solver parameters"""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if tolerance is not None:
            self.tolerance = tolerance
        if damping_factor is not None:
            self.damping_factor = damping_factor
        if learning_rate is not None:
            self.learning_rate = learning_rate
    
    def get_current_joint_angles(self) -> Dict[str, float]:
        """Get current joint angles"""
        return self.current_angles.copy()
    
    def reset_to_home(self):
        """Reset all joints to home positions"""
        self.current_angles = {joint.id: joint.home_position for joint in self.joints}


class TrajectoryGenerator:
    """Generate smooth trajectories for end-effector movements"""
    
    def __init__(self, ik_solver: InverseKinematics):
        self.ik_solver = ik_solver
        self.logger = logging.getLogger(__name__)
    
    def generate_trajectory(self, 
                          start_target: EndEffectorTarget,
                          end_target: EndEffectorTarget,
                          duration: float,
                          num_points: int = 100) -> List[IKSolution]:
        """
        Generate trajectory between two end-effector targets
        Returns list of IK solutions for each point
        """
        trajectory = []
        
        # Generate waypoints using minimum jerk trajectory
        for t in np.linspace(0, 1, num_points):
            # Interpolate position
            position = self._interpolate_position(
                start_target.position, 
                end_target.position, 
                t, 
                duration
            )
            
            # Interpolate orientation if specified
            orientation = None
            if start_target.orientation and end_target.orientation:
                orientation = self._interpolate_orientation(
                    start_target.orientation,
                    end_target.orientation,
                    t
                )
            
            # Create intermediate target
            intermediate_target = EndEffectorTarget(
                position=position,
                orientation=orientation
            )
            
            # Solve IK for this point
            solution = self.ik_solver.solve(intermediate_target)
            trajectory.append(solution)
        
        return trajectory
    
    def _interpolate_position(self, 
                           start: Tuple[float, float, float],
                           end: Tuple[float, float, float],
                           t: float,
                           duration: float) -> Tuple[float, float, float]:
        """Interpolate position using minimum jerk trajectory"""
        # Minimum jerk profile: 10t³ - 15t⁴ + 6t⁵
        tau = t
        profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        start_pos = np.array(start)
        end_pos = np.array(end)
        
        interpolated = start_pos + profile * (end_pos - start_pos)
        
        return tuple(interpolated)
    
    def _interpolate_orientation(self,
                                start: Tuple[float, float, float],
                                end: Tuple[float, float, float],
                                t: float) -> Tuple[float, float, float]:
        """Interpolate orientation using spherical linear interpolation"""
        start_angles = np.array(start)
        end_angles = np.array(end)
        
        # Simple linear interpolation for angles
        interpolated = start_angles + t * (end_angles - start_angles)
        
        return tuple(interpolated)
