#!/usr/bin/env python
"""
Starter script for lab1. 
Author: Chris Correa
"""
import copy
import sys
import argparse
import time
import numpy as np
import signal

from paths.trajectories import LinearTrajectory, CircularTrajectory, PolygonalTrajectory
from paths.paths import MotionPath
from controllers.controllers import (
    WorkspaceVelocityController, 
    PDJointVelocityController, 
    PDJointTorqueController, 
    FeedforwardJointVelocityController
)
from utils.utils import *
from path_planner import PathPlanner

from trac_ik_python.trac_ik import IK

import rospy
import tf
import tf2_ros
import intera_interface
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics


def lookup_tag(tag_number):
    """
    Given an AR tag number, this returns the position of the AR tag in the robot's base frame.
    You can use either this function or try starting the scripts/tag_pub.py script.  More info
    about that script is in that file.  

    Parameters
    ----------
    tag_number : int

    Returns
    -------
    3x' :obj:`numpy.ndarray`
        tag position
    """

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    to_frame = 'ar_marker_{}'.format(tag_number)

    try:
        trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)
        print("Retrying ...")

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    return np.array(tag_pos)

def get_trajectory(limb, kin, ik_solver, tag_pos, args):
    """
    Returns an appropriate robot trajectory for the specified task.  You should 
    be implementing the path functions in paths.py and call them here
    
    Parameters
    ----------
    task : string
        name of the task.  Options: line, circle, square
    tag_pos : 3x' :obj:`numpy.ndarray`
        
    Returns
    -------
    :obj:`moveit_msgs.msg.RobotTrajectory`
    """
    num_way = args.num_way
    controller_name = args.controller_name
    task = args.task

    # target_position = tag_pos[0]
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'right_gripper_base', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    print("Current Position:", current_position)

    if task == 'line':
        target_pos = tag_pos[0]
        target_pos[2] += 0.4 #linear path moves to a Z position above AR Tag.
        print("TARGET POSITION:", target_pos)
        trajectory = LinearTrajectory(start_position=current_position, goal_position=target_pos, total_time=4)
    elif task == 'circle':
        target_pos = tag_pos[0]
        target_pos[2] += 0.5
        print("TARGET POSITION:", target_pos)
        trajectory = CircularTrajectory(center_position=target_pos, radius=0.1, total_time=6)    
    elif task == 'polygon':
        print("tagpos", tag_pos)
        points = np.array([point + np.array([0,0,0.5]) for point in tag_pos])
        print("points", points)
        trajectory = PolygonalTrajectory(start_position=current_position, points=points, total_time=3)
    else:
        raise ValueError('task {} not recognized'.format(task))
    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, controller_name!='workspace')

def get_controller(controller_name, limb, kin):
    """
    Gets the correct controller from controllers.py

    Parameters
    ----------
    controller_name : string

    Returns
    -------
    :obj:`Controller`
    """
    if controller_name == 'workspace':
        # YOUR CODE HERE
        Kp = None
        Kv = None
        controller = WorkspaceVelocityController(limb, kin, Kp, Kv)
    elif controller_name == 'jointspace':
        Kp = 3 * np.array([0.4, 2, 1.7, 1.7, 2, 2, 3])
        Kd = 0.3 * np.array([2, 1, 2, 0.5, 5, 0.8, 0.8])
        Ki = 0 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    elif controller_name == 'torque':
        # YOUR CODE HERE
        Kp = None
        Kv = None
        controller = PDJointTorqueController(limb, kin, Kp, Kv)
    elif controller_name == 'open_loop':
        controller = FeedforwardJointVelocityController(limb, kin)
    else:
        raise ValueError('Controller {} not recognized'.format(controller_name))
    return controller


def main():
    """
    Examples of how to run me:
    python scripts/main.py --help <------This prints out all the help messages
    and describes what each parameter is
    python scripts/main.py -t line -ar 1 -c workspace --log
    python scripts/main.py -t circle -ar 2 -c jointspace --logget_tra
    python scripts/main.py -t polygon -ar 3 -c torque --log

    You can also change the rate, timeout if you want
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('-ar_marker', '-ar', nargs='+', help=
        'Which AR marker to use.  Default: 1'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='moveit', 
        help='Options: moveit, open_loop, jointspace, workspace, or torque.  Default: moveit'
    )
    parser.add_argument('-rate', type=int, default=200, help="""
        This specifies how many ms between loops.  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200"""
    )
    parser.add_argument('-timeout', type=int, default=None, help=
        """after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None"""
    )
    parser.add_argument('-num_way', type=int, default=300, help=
        'How many waypoints for the :obj:`moveit_msgs.msg.RobotTrajectory`.  Default: 300'
    )
    parser.add_argument('--log', action='store_true', help='plots controller performance')
    args = parser.parse_args()

    rospy.init_node('moveit_node')
    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_hand")
    limb = intera_interface.Limb('right')
    # this is used to get the dynamics (inertia matrix, manipulator jacobian, etc) from the robot
    # in the current position, UNLESS you specify other joint angles.  see the source code
    # https://github.com/ucb-ee106/baxter_pykdl/blob/master/src/sawyer_pykdl/sawyer_pykdl.py
    # for info on how to use each method
    kin = sawyer_kinematics('right')

    tag_pos = [lookup_tag(marker) for marker in args.ar_marker]
    # Get an appropriate RobotTrajectory for the task (circular, linear, or square)
    # If the controller is a workspace controller, this should return a trajectory where the
    # positions and velocities are workspace positions and velocities.  If the controller
    # is a jointspace or torque controller, it should return a trajectory where the positions
    # and velocities are the positions and velocities of each joint.
    robot_trajectory = get_trajectory(limb, kin, ik_solver, tag_pos, args)

    # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
    # of the trajectory
    planner = PathPlanner('right_arm')
    if args.controller_name == "workspace":
        pose = create_pose_stamped_from_pos_quat(
            robot_trajectory.joint_trajectory.points[0].positions,
            [0, 1, 0, 0],
            'base'
        )
        plan = planner.plan_to_pose(pose)
        planner.execute_plan(plan[1])
    else:
        start = robot_trajectory.joint_trajectory.points[0].positions
        print("START:", robot_trajectory.joint_trajectory.points[0].positions)
        
        while not rospy.is_shutdown():
            try:
                # limb.move_to_joint_positions(joint_array_to_dict(start, limb), timeout=7.0, threshold=0.0001) # ONLY FOR EMERGENCIES!!!
                plan = planner.plan_to_joint_pos(start)
                planner.execute_plan(plan[1])
                break
            except moveit_commander.exception.MoveItCommanderException as e:
                print(e)
                print("Failed planning, retrying...")

    if args.controller_name == "moveit":
        # by publishing the trajectory to the move_group/display_planned_path topic, you should 
        # be able to view it in RViz.  You will have to click the "loop animation" setting in 
        # the planned path section of MoveIt! in the menu on the left side of the screen.
        pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        # disp_traj.trajectory_start = planner._group.get_current_joint_values()
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)

        try:
            input('Press <Enter> to execute the trajectory using MOVEIT')
        except KeyboardInterrupt:
            sys.exit()
        # uses MoveIt! to execute the trajectory.  make sure to view it in RViz before running this.
        # the lines above will display the trajectory in RViz
        planner.execute_plan(robot_trajectory)
    else:
        # Project 1 PART B
        controller = get_controller(args.controller_name, limb, kin)
        try:
            input('Press <Enter> to execute the trajectory using YOUR OWN controller')
        except KeyboardInterrupt:
            sys.exit()
        # execute the path using your own controller.
        done = controller.execute_path(
            robot_trajectory, 
            rate=args.rate, 
            timeout=args.timeout, 
            log=args.log
        )
        if not done:
            print('Failed to move to position')
            sys.exit(0)


if __name__ == "__main__":
    main()
