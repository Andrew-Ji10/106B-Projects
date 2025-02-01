#!/usr/bin/env python

"""
Starter script for Project 1. 
Author: Chris Correa
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from utils.utils import *

try:
    import rospy
    from moveit_msgs.msg import RobotTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except:
    pass

class MotionPath:
    def __init__(self, limb, kin, ik_solver, trajectory):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        trajectory: Trajectory object (see trajectories.py)
        total_time : float
            number of seconds you wish the trajectory to run for
        """
        self.limb = limb
        self.kin = kin
        self.ik_solver = ik_solver
        self.trajectory = trajectory
        self.previous_computed_ik = get_joint_positions(self.limb)

    def to_robot_trajectory(self, num_waypoints=300, jointspace=True):
        """
        Parameters
        ----------
        num_waypoints : float
            how many points in the :obj:`moveit_msgs.msg.RobotTrajectory`
        jointspace : bool
            What kind of trajectory.  Joint space points are 7x' and describe the
            angle of each arm.  Workspace points are 3x', and describe the x,y,z
            position of the end effector.  
        """
        traj = JointTrajectory()
        traj.joint_names = self.limb.joint_names()
        points = []
        for t in np.linspace(0, self.trajectory.total_time, num=num_waypoints):
            point = self.trajectory_point(t, jointspace)
            points.append(point)

        # We want to make a final point at the end of the trajectory so that the 
        # controller has time to converge to the final point.
        extra_point = self.trajectory_point(self.trajectory.total_time, jointspace)
        extra_point.time_from_start = rospy.Duration.from_sec(self.trajectory.total_time + 1)
        points.append(extra_point)

        traj.points = points
        traj.header.frame_id = 'base'
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = traj
        return robot_traj

    def trajectory_point(self, t, jointspace):
        """
        takes a discrete point in time, and puts the position, velocity, and
        acceleration into a ROS JointTrajectoryPoint() to be put into a 
        RobotTrajectory.  
        
        Parameters
        ----------
        t : float
        jointspace : bool
            What kind of trajectory.  Joint space points are 7x' and describe the
            angle of each arm.  Workspace points are 3x', and describe the x,y,z
            position of the end effector.  

        Returns
        -------
        :obj:`trajectory_msgs.msg.JointTrajectoryPoint`


        joint_names: [left_s0, left_s1, left_e0, left_e1, left_w0, left_w1, left_w2]
        points: 
        - 
        positions: [-0.11520713 -1.01663718 -1.13026189  1.91170776  0.5837694   1.05630898  -0.70543966]

        """
        point = JointTrajectoryPoint()
        delta_t = .01
        if jointspace:
            x_t, x_t_1, x_t_2 = None, None, None
            ik_attempts = 0
            theta_t_2 = self.get_ik(self.trajectory.target_pose(t-2*delta_t), seed=self.previous_computed_ik)
            theta_t_1 = self.get_ik(self.trajectory.target_pose(t-delta_t), seed=self.previous_computed_ik)
            theta_t   = self.get_ik(self.trajectory.target_pose(t), seed=self.previous_computed_ik)
            
            # we said you shouldn't simply take a finite difference when creating
            # the path, why do you think we're doing that here?
            point.positions = theta_t
            point.velocities = (theta_t - theta_t_1) / delta_t
            point.accelerations = (theta_t - 2*theta_t_1 + theta_t_2) / (delta_t**2)
            self.previous_computed_ik = theta_t
        else:
            point.positions = self.trajectory.target_pose(t)
            point.velocities = self.trajectory.target_velocity(t)
        point.time_from_start = rospy.Duration.from_sec(t)
        return point

    def get_ik(self, x, seed=None, ik_timeout=0.1, max_ik_attempts=10):
        """
        gets ik
        
        Parameters
        ----------
        x : 7x' :obj:`numpy.ndarray`
            pose of the end effector
        ik_timeout : float
            time in seconds after which ik solution will short circuit.

        Returns
        -------
        7x' :obj:`numpy.ndarray`
            joint values to achieve the passed in workspace position
        """
        if self.ik_solver and seed is None:
            seed = [0.0] * self.ik_solver.number_of_joints

        ik_attempts, theta = 0, None
        while theta is None and not rospy.is_shutdown():
            if self.ik_solver:
                theta = self.ik_solver.get_ik(seed,
                    x[0], x[1], x[2],      # XYZ
                    x[3], x[4], x[5], x[6] # quat
                    )
            else:
                theta = self.kin.inverse_kinematics(
                position=[x[0], x[1], x[2]],
                orientation=[x[3], x[4], x[5], x[6]])
            ik_attempts += 1
            if ik_attempts > max_ik_attempts:
                rospy.signal_shutdown(
                    'MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(x)
                )
                print('MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(x))
        return np.array(theta)