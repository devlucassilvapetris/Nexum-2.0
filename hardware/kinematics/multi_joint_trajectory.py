"""
Nexum v2.0 Multi-Joint Trajectory Planning
Advanced trajectory planning for coordinated multi-articulated movements
"""

import numpy as np
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from .inverse_kinematics import (
    InverseKinematics, 
    EndEffectorTarget, 
    IKSolution,
    JointConfig,
    LinkConfig
)


class TrajectoryType(Enum):
    MINIMUM_JERK = "minimum_jerk"
    MINIMUM_ACCELERATION = "minimum_acceleration"
    MINIMUM_TIME = "minimum_time"
    CUBIC_SPLINE = "cubic_spline"
    LINEAR = "linear"


@dataclass
class TrajectoryPoint:
    time: float
    joint_angles: Dict[str, float]
    joint_velocities: Optional[Dict[str, float]] = None
    joint_accelerations: Optional[Dict[str, float]] = None
    end_effector_position: Optional[Tuple[float, float, float]] = None


@dataclass
class TrajectoryConstraints:
    max_joint_velocities: Dict[str, float]
    max_joint_accelerations: Dict[str, float]
    max_joint_torques: Optional[Dict[str, float]] = None
    collision_obstacles: Optional[List[Tuple[float, float, float, float]]] = None  # (x, y, z, radius)


@dataclass
class Waypoint:
    position: Tuple[float, float, float]
    orientation: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    arrival_time: Optional[float] = None


class MultiJointTrajectoryPlanner:
    """
    Advanced trajectory planner for coordinated multi-joint movements
    Handles constraints, optimization, and smooth trajectories
    """
    
    def __init__(self, 
                 ik_solver: InverseKinematics,
                 joints: List[JointConfig]):
        self.ik_solver = ik_solver
        self.joints = joints
        self.logger = logging.getLogger(__name__)
        
        # Planning parameters
        self.default_trajectory_type = TrajectoryType.MINIMUM_JERK
        self.time_step = 0.01  # 100 Hz
        self.optimization_tolerance = 1e-6
        
        # Current trajectory
        self.current_trajectory: List[TrajectoryPoint] = []
        self.current_trajectory_index = 0
        self.is_executing = False
    
    def plan_trajectory(self,
                       waypoints: List[Waypoint],
                       trajectory_type: TrajectoryType = None,
                       constraints: Optional[TrajectoryConstraints] = None) -> List[TrajectoryPoint]:
        """
        Plan trajectory through multiple waypoints
        """
        if trajectory_type is None:
            trajectory_type = self.default_trajectory_type
        
        self.logger.info(f"Planning {trajectory_type.value} trajectory through {len(waypoints)} waypoints")
        
        # Generate time profile for waypoints
        time_profile = self._generate_time_profile(waypoints, constraints)
        
        # Generate joint space trajectory
        if trajectory_type == TrajectoryType.MINIMUM_JERK:
            trajectory = self._plan_minimum_jerk(waypoints, time_profile)
        elif trajectory_type == TrajectoryType.MINIMUM_ACCELERATION:
            trajectory = self._plan_minimum_acceleration(waypoints, time_profile)
        elif trajectory_type == TrajectoryType.CUBIC_SPLINE:
            trajectory = self._plan_cubic_spline(waypoints, time_profile)
        elif trajectory_type == TrajectoryType.LINEAR:
            trajectory = self._plan_linear(waypoints, time_profile)
        else:
            trajectory = self._plan_minimum_jerk(waypoints, time_profile)
        
        # Apply constraints
        if constraints:
            trajectory = self._apply_constraints(trajectory, constraints)
        
        # Optimize trajectory
        trajectory = self._optimize_trajectory(trajectory, constraints)
        
        self.current_trajectory = trajectory
        self.current_trajectory_index = 0
        
        return trajectory
    
    def _generate_time_profile(self, 
                              waypoints: List[Waypoint],
                              constraints: Optional[TrajectoryConstraints] = None) -> List[float]:
        """Generate arrival times for waypoints"""
        times = []
        current_time = 0.0
        
        for i, waypoint in enumerate(waypoints):
            if waypoint.arrival_time is not None:
                current_time = waypoint.arrival_time
            else:
                # Estimate time based on distance and constraints
                if i > 0:
                    prev_waypoint = waypoints[i - 1]
                    distance = np.linalg.norm(
                        np.array(waypoint.position) - np.array(prev_waypoint.position)
                    )
                    
                    # Estimate velocity from constraints
                    max_vel = 1.0  # Default m/s
                    if constraints and constraints.max_joint_velocities:
                        # Approximate end-effector velocity from joint velocities
                        max_vel = np.mean(list(constraints.max_joint_velocities.values()))
                    
                    segment_time = distance / max_vel
                    current_time += max(segment_time, 0.1)  # Minimum 0.1s per segment
            
            times.append(current_time)
        
        return times
    
    def _plan_minimum_jerk(self, 
                          waypoints: List[Waypoint],
                          times: List[float]) -> List[TrajectoryPoint]:
        """Plan minimum jerk trajectory through waypoints"""
        trajectory = []
        
        # Solve IK for each waypoint
        waypoint_solutions = []
        for waypoint in waypoints:
            target = EndEffectorTarget(
                position=waypoint.position,
                orientation=waypoint.orientation
            )
            solution = self.ik_solver.solve(target)
            waypoint_solutions.append(solution)
        
        # Generate trajectory points
        total_time = times[-1]
        num_points = int(total_time / self.time_step)
        
        for i in range(num_points + 1):
            t = i * self.time_step
            
            # Find which segment we're in
            segment_idx = 0
            for j in range(len(times) - 1):
                if times[j] <= t <= times[j + 1]:
                    segment_idx = j
                    break
            
            # Normalize time within segment
            t_start = times[segment_idx]
            t_end = times[segment_idx + 1] if segment_idx < len(times) - 1 else times[-1]
            tau = (t - t_start) / (t_end - t_start) if t_end > t_start else 0
            
            # Get start and end joint configurations
            start_angles = waypoint_solutions[segment_idx].joint_angles
            end_angles = waypoint_solutions[segment_idx + 1].joint_angles if segment_idx + 1 < len(waypoint_solutions) else start_angles
            
            # Interpolate using minimum jerk profile
            joint_angles = {}
            joint_velocities = {}
            joint_accelerations = {}
            
            for joint_id in start_angles.keys():
                start_angle = start_angles[joint_id]
                end_angle = end_angles.get(joint_id, start_angle)
                
                # Minimum jerk profile: 10τ³ - 15τ⁴ + 6τ⁵
                profile = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                angle = start_angle + profile * (end_angle - start_angle)
                
                # Velocity: derivative of position
                velocity_profile = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
                velocity = velocity_profile * (end_angle - start_angle) / (t_end - t_start)
                
                # Acceleration: derivative of velocity
                accel_profile = 60 * tau - 180 * tau**2 + 120 * tau**3
                acceleration = accel_profile * (end_angle - start_angle) / (t_end - t_start)**2
                
                joint_angles[joint_id] = angle
                joint_velocities[joint_id] = velocity
                joint_accelerations[joint_id] = acceleration
            
            # Compute end-effector position
            end_effector_pos = self._compute_end_effector_position(joint_angles)
            
            trajectory_point = TrajectoryPoint(
                time=t,
                joint_angles=joint_angles,
                joint_velocities=joint_velocities,
                joint_accelerations=joint_accelerations,
                end_effector_position=end_effector_pos
            )
            
            trajectory.append(trajectory_point)
        
        return trajectory
    
    def _plan_minimum_acceleration(self,
                                   waypoints: List[Waypoint],
                                   times: List[float]) -> List[TrajectoryPoint]:
        """Plan minimum acceleration trajectory (quadratic)"""
        trajectory = []
        
        # Solve IK for each waypoint
        waypoint_solutions = []
        for waypoint in waypoints:
            target = EndEffectorTarget(
                position=waypoint.position,
                orientation=waypoint.orientation
            )
            solution = self.ik_solver.solve(target)
            waypoint_solutions.append(solution)
        
        # Generate trajectory points
        total_time = times[-1]
        num_points = int(total_time / self.time_step)
        
        for i in range(num_points + 1):
            t = i * self.time_step
            
            # Find segment
            segment_idx = 0
            for j in range(len(times) - 1):
                if times[j] <= t <= times[j + 1]:
                    segment_idx = j
                    break
            
            t_start = times[segment_idx]
            t_end = times[segment_idx + 1] if segment_idx + 1 < len(times) - 1 else times[-1]
            tau = (t - t_start) / (t_end - t_start) if t_end > t_start else 0
            
            start_angles = waypoint_solutions[segment_idx].joint_angles
            end_angles = waypoint_solutions[segment_idx + 1].joint_angles if segment_idx + 1 < len(waypoint_solutions) else start_angles
            
            # Quadratic profile: 3τ² - 2τ³
            profile = 3 * tau**2 - 2 * tau**3
            
            joint_angles = {}
            joint_velocities = {}
            joint_accelerations = {}
            
            for joint_id in start_angles.keys():
                start_angle = start_angles[joint_id]
                end_angle = end_angles.get(joint_id, start_angle)
                
                angle = start_angle + profile * (end_angle - start_angle)
                
                velocity_profile = 6 * tau - 6 * tau**2
                velocity = velocity_profile * (end_angle - start_angle) / (t_end - t_start)
                
                accel_profile = 6 - 12 * tau
                acceleration = accel_profile * (end_angle - start_angle) / (t_end - t_start)**2
                
                joint_angles[joint_id] = angle
                joint_velocities[joint_id] = velocity
                joint_accelerations[joint_id] = acceleration
            
            end_effector_pos = self._compute_end_effector_position(joint_angles)
            
            trajectory.append(TrajectoryPoint(
                time=t,
                joint_angles=joint_angles,
                joint_velocities=joint_velocities,
                joint_accelerations=joint_accelerations,
                end_effector_position=end_effector_pos
            ))
        
        return trajectory
    
    def _plan_cubic_spline(self,
                          waypoints: List[Waypoint],
                          times: List[float]) -> List[TrajectoryPoint]:
        """Plan trajectory using cubic splines"""
        trajectory = []
        
        # Solve IK for each waypoint
        waypoint_solutions = []
        for waypoint in waypoints:
            target = EndEffectorTarget(
                position=waypoint.position,
                orientation=waypoint.orientation
            )
            solution = self.ik_solver.solve(target)
            waypoint_solutions.append(solution)
        
        # Create spline for each joint
        joint_splines = {}
        for joint_id in waypoint_solutions[0].joint_angles.keys():
            angles = [sol.joint_angles[joint_id] for sol in waypoint_solutions]
            
            # Create cubic spline interpolation
            spline = interp1d(times, angles, kind='cubic', fill_value='extrapolate')
            joint_splines[joint_id] = spline
        
        # Generate trajectory points
        total_time = times[-1]
        num_points = int(total_time / self.time_step)
        
        for i in range(num_points + 1):
            t = i * self.time_step
            
            joint_angles = {}
            joint_velocities = {}
            joint_accelerations = {}
            
            for joint_id, spline in joint_splines.items():
                # Interpolate angle
                angle = float(spline(t))
                joint_angles[joint_id] = angle
                
                # Numerical derivatives for velocity and acceleration
                dt = 1e-6
                angle_plus = float(spline(t + dt))
                angle_minus = float(spline(t - dt))
                
                velocity = (angle_plus - angle_minus) / (2 * dt)
                acceleration = (angle_plus - 2 * angle + angle_minus) / (dt ** 2)
                
                joint_velocities[joint_id] = velocity
                joint_accelerations[joint_id] = acceleration
            
            end_effector_pos = self._compute_end_effector_position(joint_angles)
            
            trajectory.append(TrajectoryPoint(
                time=t,
                joint_angles=joint_angles,
                joint_velocities=joint_velocities,
                joint_accelerations=joint_accelerations,
                end_effector_position=end_effector_pos
            ))
        
        return trajectory
    
    def _plan_linear(self,
                    waypoints: List[Waypoint],
                    times: List[float]) -> List[TrajectoryPoint]:
        """Plan simple linear trajectory"""
        trajectory = []
        
        # Solve IK for each waypoint
        waypoint_solutions = []
        for waypoint in waypoints:
            target = EndEffectorTarget(
                position=waypoint.position,
                orientation=waypoint.orientation
            )
            solution = self.ik_solver.solve(target)
            waypoint_solutions.append(solution)
        
        # Generate trajectory points
        total_time = times[-1]
        num_points = int(total_time / self.time_step)
        
        for i in range(num_points + 1):
            t = i * self.time_step
            
            # Find segment
            segment_idx = 0
            for j in range(len(times) - 1):
                if times[j] <= t <= times[j + 1]:
                    segment_idx = j
                    break
            
            t_start = times[segment_idx]
            t_end = times[segment_idx + 1] if segment_idx + 1 < len(times) - 1 else times[-1]
            tau = (t - t_start) / (t_end - t_start) if t_end > t_start else 0
            
            start_angles = waypoint_solutions[segment_idx].joint_angles
            end_angles = waypoint_solutions[segment_idx + 1].joint_angles if segment_idx + 1 < len(waypoint_solutions) else start_angles
            
            # Linear interpolation
            joint_angles = {}
            for joint_id in start_angles.keys():
                start_angle = start_angles[joint_id]
                end_angle = end_angles.get(joint_id, start_angle)
                angle = start_angle + tau * (end_angle - start_angle)
                joint_angles[joint_id] = angle
            
            end_effector_pos = self._compute_end_effector_position(joint_angles)
            
            trajectory.append(TrajectoryPoint(
                time=t,
                joint_angles=joint_angles,
                end_effector_position=end_effector_pos
            ))
        
        return trajectory
    
    def _apply_constraints(self,
                         trajectory: List[TrajectoryPoint],
                         constraints: TrajectoryConstraints) -> List[TrajectoryPoint]:
        """Apply velocity and acceleration constraints"""
        constrained_trajectory = []
        
        for point in trajectory:
            constrained_point = TrajectoryPoint(
                time=point.time,
                joint_angles=point.joint_angles.copy(),
                joint_velocities=point.joint_velocities.copy() if point.joint_velocities else None,
                joint_accelerations=point.joint_accelerations.copy() if point.joint_accelerations else None,
                end_effector_position=point.end_effector_position
            )
            
            # Clamp velocities
            if constrained_point.joint_velocities:
                for joint_id, velocity in constrained_point.joint_velocities.items():
                    max_vel = constraints.max_joint_velocities.get(joint_id, float('inf'))
                    constrained_point.joint_velocities[joint_id] = max(min(velocity, max_vel), -max_vel)
            
            # Clamp accelerations
            if constrained_point.joint_accelerations:
                for joint_id, acceleration in constrained_point.joint_accelerations.items():
                    max_accel = constraints.max_joint_accelerations.get(joint_id, float('inf'))
                    constrained_point.joint_accelerations[joint_id] = max(min(acceleration, max_accel), -max_accel)
            
            constrained_trajectory.append(constrained_point)
        
        return constrained_trajectory
    
    def _optimize_trajectory(self,
                            trajectory: List[TrajectoryPoint],
                            constraints: Optional[TrajectoryConstraints] = None) -> List[TrajectoryPoint]:
        """Optimize trajectory for smoothness and constraint satisfaction"""
        if len(trajectory) < 3:
            return trajectory
        
        # Extract joint angles as array
        joint_ids = list(trajectory[0].joint_angles.keys())
        num_joints = len(joint_ids)
        num_points = len(trajectory)
        
        angles_array = np.zeros((num_points, num_joints))
        for i, point in enumerate(trajectory):
            for j, joint_id in enumerate(joint_ids):
                angles_array[i, j] = point.joint_angles[joint_id]
        
        # Optimize using smoothing (simple moving average)
        window_size = 5
        smoothed_angles = np.zeros_like(angles_array)
        
        for j in range(num_joints):
            for i in range(num_points):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(num_points, i + window_size // 2 + 1)
                smoothed_angles[i, j] = np.mean(angles_array[start_idx:end_idx, j])
        
        # Reconstruct trajectory
        optimized_trajectory = []
        for i, point in enumerate(trajectory):
            optimized_point = TrajectoryPoint(
                time=point.time,
                joint_angles={joint_ids[j]: smoothed_angles[i, j] for j in range(num_joints)},
                joint_velocities=point.joint_velocities,
                joint_accelerations=point.joint_accelerations,
                end_effector_position=point.end_effector_position
            )
            optimized_trajectory.append(optimized_point)
        
        return optimized_trajectory
    
    def _compute_end_effector_position(self, joint_angles: Dict[str, float]) -> Tuple[float, float, float]:
        """Compute end-effector position from joint angles"""
        position, _ = self.ik_solver.fk.compute_end_effector(joint_angles)
        return tuple(position)
    
    def execute_trajectory(self, 
                          trajectory: List[TrajectoryPoint],
                          motor_controller,
                          progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Execute trajectory on motor controller
        """
        self.is_executing = True
        self.current_trajectory = trajectory
        self.current_trajectory_index = 0
        
        try:
            for i, point in enumerate(trajectory):
                if not self.is_executing:
                    break
                
                # Move motors to target positions
                for joint_id, angle in point.joint_angles.items():
                    motor_controller.move_motor(joint_id, angle)
                
                # Wait for movement to complete
                motor_controller.wait_for_movement(timeout=1.0)
                
                # Update progress
                self.current_trajectory_index = i
                if progress_callback:
                    progress = (i + 1) / len(trajectory)
                    progress_callback(progress)
            
            self.is_executing = False
            return True
            
        except Exception as e:
            self.logger.error(f"Trajectory execution error: {e}")
            self.is_executing = False
            return False
    
    def stop_execution(self):
        """Stop current trajectory execution"""
        self.is_executing = False
    
    def get_current_point(self) -> Optional[TrajectoryPoint]:
        """Get current trajectory point during execution"""
        if self.current_trajectory_index < len(self.current_trajectory):
            return self.current_trajectory[self.current_trajectory_index]
        return None
    
    def get_trajectory_progress(self) -> float:
        """Get trajectory execution progress (0.0 to 1.0)"""
        if not self.current_trajectory:
            return 0.0
        return self.current_trajectory_index / len(self.current_trajectory)
    
    def set_trajectory_type(self, trajectory_type: TrajectoryType):
        """Set default trajectory type"""
        self.default_trajectory_type = trajectory_type
    
    def set_time_step(self, time_step: float):
        """Set trajectory time step"""
        self.time_step = max(0.001, time_step)  # Minimum 1ms


class CollisionAvoidance:
    """Collision detection and avoidance for trajectory planning"""
    
    def __init__(self, links: List[LinkConfig]):
        self.links = links
        self.logger = logging.getLogger(__name__)
    
    def check_collision(self,
                       joint_angles: Dict[str, float],
                       obstacles: List[Tuple[float, float, float, float]]) -> bool:
        """
        Check if current configuration collides with obstacles
        obstacles: list of (x, y, z, radius) tuples
        """
        # Compute link positions
        link_positions = self._compute_link_positions(joint_angles)
        
        # Check each link against each obstacle
        for link_pos, (obs_x, obs_y, obs_z, obs_radius) in zip(link_positions, obstacles):
            distance = np.linalg.norm(link_pos - np.array([obs_x, obs_y, obs_z]))
            if distance < obs_radius:
                return True
        
        return False
    
    def _compute_link_positions(self, joint_angles: Dict[str, float]) -> List[np.ndarray]:
        """Compute positions of all links"""
        positions = []
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for i, (link, joint_id) in enumerate(zip(self.links, joint_angles.keys())):
            positions.append(current_pos)
            
            # Simple forward kinematics for position
            angle = joint_angles[joint_id]
            if i == 0:
                current_pos += np.array([link.length * math.cos(angle), link.length * math.sin(angle), 0])
            elif i == 1:
                current_pos += np.array([0, link.length * math.sin(angle), link.length * math.cos(angle)])
            else:
                current_pos += np.array([0, 0, link.length])
        
        return positions
    
    def find_safe_trajectory(self,
                            trajectory: List[TrajectoryPoint],
                            obstacles: List[Tuple[float, float, float, float]]) -> List[TrajectoryPoint]:
        """Modify trajectory to avoid collisions"""
        safe_trajectory = []
        
        for point in trajectory:
            if not self.check_collision(point.joint_angles, obstacles):
                safe_trajectory.append(point)
            else:
                # Try to find nearby safe configuration
                safe_point = self._find_safe_configuration(point, obstacles)
                if safe_point:
                    safe_trajectory.append(safe_point)
        
        return safe_trajectory
    
    def _find_safe_configuration(self,
                               point: TrajectoryPoint,
                               obstacles: List[Tuple[float, float, float, float]]) -> Optional[TrajectoryPoint]:
        """Find safe configuration near current point"""
        # Simple approach: try small random perturbations
        for _ in range(10):
            perturbed_angles = point.joint_angles.copy()
            for joint_id in perturbed_angles.keys():
                perturbation = np.random.uniform(-0.1, 0.1)
                perturbed_angles[joint_id] += perturbation
            
            if not self.check_collision(perturbed_angles, obstacles):
                return TrajectoryPoint(
                    time=point.time,
                    joint_angles=perturbed_angles,
                    joint_velocities=point.joint_velocities,
                    joint_accelerations=point.joint_accelerations,
                    end_effector_position=point.end_effector_position
                )
        
        return None
