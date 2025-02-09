#!/usr/bin/env python
"""
Following AR Tag Script for Project 1B
Author: Chris Correa
"""
import copy
import sys
import argparse
import time
import numpy as np
import signal

from controllers.controllers import (
    WorkspaceVelocityController, 
    PDJointVelocityController, 
    PDJointTorqueController, 
    FeedforwardJointVelocityController
)
from utils.utils import *

from trac_ik_python.trac_ik import IK

import rospy
import tf
import intera_interface
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
    done = True

    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), rospy.Duration(10.0))
            break
        except Exception as e:
            print("Retrying ...")

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    return np.array(tag_pos)

def get_traj(limb, kin, ik_solver, tag_pos, rate, num_way=10):
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

    # target_position = tag_pos[0]
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    try:
        trans = tfBuffer.lookup_transform('base', 'stp_022412TP99883_tip', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    #print("Current Position:", current_position)

    target_pos = tag_pos[0]
    target_pos[2] += 0.4 #linear path moves to a Z position above AR Tag.
    #print("TARGET POSITION:", target_pos)
    trajectory = ConstTrajectory(start_position=current_position, goal_position=target_pos, total_time=rate/1000)
    
    trajectory.display_trajectory()

    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_way, True)

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
        Kp = 0.6 * np.array([1.8, 2.4, 1.7, 2, 2, 2.4, 2.5])
        Kd = 0.005 * np.array([2, 1, 3, 0.5, 150, 1.2, 0.8])
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

if __name__ == "__main__":
    """
    Examples of how to run me:
    python scripts/follow_ar.py -ar 1 -c workspace -a left --log
    python scripts/follow_ar.py -ar 2 -c velocity -a left --log
    python scripts/follow_ar.py -ar 3 -c torque -a right --log

    You can also change the rate, timeout if you want
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ar_marker', '-ar', type=float, default=1, help=
        'Which AR marker to use.  Default: 1'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='workspace', 
        help='Options: workspace, jointspace, or torque.  Default: workspace'
    )
    parser.add_argument('-arm', '-a', type=str, default='right', help=
        'Options: left, right.  Default: right'
    )
    parser.add_argument('-rate', type=int, default=25, help="""
        This specifies how many ms between loops (input is HZ).  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200"""
    )
    parser.add_argument('-timeout', type=int, default=500, help=
        """after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None"""
    )
    parser.add_argument('--log', action='store_true', help='plots controller performance')
    args = parser.parse_args()

    rospy.init_node('moveit_node')
    
    ik_solver = IK("base", "right_hand")
    limb = intera_interface.Limb(args.arm)
    kin = sawyer_kinematics(args.arm)

    controller = get_controller(args.controller_name, limb, kin)
    try:
        input('Press <Enter> to execute the follow ar trajectory using YOUR OWN controller')
    except KeyboardInterrupt:
        sys.exit()
    controller.follow_ar_tag(
        int(args.ar_marker), 
        rate=args.rate, 
        timeout=args.timeout, 
        log=args.log
    )
    