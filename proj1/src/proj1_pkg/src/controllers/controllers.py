#!/usr/bin/env python

"""
Starter script for Project 1B. 
Author: Chris Correa, Valmik Prabhu
"""

# Python imports
import sys
import numpy as np
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Lab imports
from utils.utils import *

# ROS imports
try:
    import tf
    import tf2_ros
    import rospy
    import baxter_interface
    import intera_interface
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.msg import RobotTrajectory
    from trac_ik_python.trac_ik import IK
    from paths.trajectories import LinearTrajectory
    from paths.trajectories import ConstTrajectory
    from paths.paths import MotionPath
except:
    pass

NUM_JOINTS = 7

def get_traj(limb, kin, trans, ik_solver, tag_pos, rate=1000, num_way=40):
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
    start_t = rospy.Time.now()

    # # target_position = tag_pos[0]
    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)
    # print("pos calc -1: ", (rospy.Time.now() - start_t).to_sec())

    # try:
    #     trans = tfBuffer.lookup_transform('base', 'stp_022412TP99883_tip', rospy.Time(0), rospy.Duration(10.0))
    # except Exception as e:
    #     print(e)

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])

    print("pos calc: ", (rospy.Time.now() - start_t).to_sec())


    target_pos = tag_pos
    target_pos[2] += 0.6 #linear path moves to a Z position above AR Tag.
    trajectory = ConstTrajectory(start_position=current_position, goal_position=target_pos, total_time=rate/1000)
    
    print("pt2 calc: ", (rospy.Time.now() - start_t).to_sec())


    #trajectory.display_trajectory()

    path = MotionPath(limb, kin, ik_solver, trajectory)
    
    ShouldMove = np.linalg.norm(target_pos-current_position) >= 0.01
    
    # if (ShouldMove):
    #     #print("Current Position:", current_position)
    #     #print("TARGET POSITION:", target_pos)
  


    return path.to_robot_trajectory(num_way, True), ShouldMove

class Controller:

    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor
    self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`baxter_pykdl.baxter_kinematics` or :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin

        # Set this attribute to True if the present controller is a jointspace controller.
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly.  

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        """
        pass

    def interpolate_path(self, path, t, current_index = 0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray` 
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray` 
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray` 
            desired accelerations
        current_index : int
            waypoint index at which search was terminated 
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points)-1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if (path.joint_trajectory.points[current_index].time_from_start.to_sec() > t):
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown() and \
            current_index < max_index and \
            path.joint_trajectory.points[current_index+1].time_from_start.to_sec() < t+epsilon
        ):
            current_index = current_index+1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[current_index].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[current_index+1].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index+1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index+1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index+1].accelerations
            )

            target_position = target_position_low + \
                (t - time_low)/(time_high - time_low)*(target_position_high - target_position_low)
            target_velocity = target_velocity_low + \
                (t - time_low)/(time_high - time_low)*(target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + \
                (t - time_low)/(time_high - time_low)*(target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(path.joint_trajectory.points[current_index].positions)
            target_velocity = np.array(path.joint_trajectory.points[current_index].velocities)
            target_acceleration = np.array(path.joint_trajectory.points[current_index].velocities)

        return (target_position, target_velocity, target_acceleration, current_index)


    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)

    def plot_results(
        self,
        times,
        actual_positions, 
        actual_velocities, 
        target_positions, 
        target_velocities
    ):
        """
        Plots results.
        If the path is in joint space, it will plot both workspace and jointspace plots.
        Otherwise it'll plot only workspace

        Inputs:
        times : nx' :obj:`numpy.ndarray`
        actual_positions : nx7 or nx6 :obj:`numpy.ndarray`
            actual joint positions for each time in times
        actual_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            actual joint velocities for each time in times
        target_positions: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace positions for each time in times
        target_velocities: nx7 or nx6 :obj:`numpy.ndarray`
            target joint or workspace velocities for each time in times
        """

        # Make everything an ndarray
        times = np.array(times)
        actual_positions = np.array(actual_positions)
        actual_velocities = np.array(actual_velocities)
        target_positions = np.array(target_positions)
        target_velocities = np.array(target_velocities)

        # Find the actual workspace positions and velocities
        actual_workspace_positions = np.zeros((len(times), 3))
        actual_workspace_velocities = np.zeros((len(times), 3))
        actual_workspace_quaternions = np.zeros((len(times), 4))

        for i in range(len(times)):
            positions_dict = joint_array_to_dict(actual_positions[i], self._limb)
            fk = self._kin.forward_position_kinematics(joint_values=positions_dict)
            
            actual_workspace_positions[i, :] = fk[:3]
            actual_workspace_velocities[i, :] = \
                self._kin.jacobian(joint_values=positions_dict)[:3].dot(actual_velocities[i])
            actual_workspace_quaternions[i, :] = fk[3:]
        # check if joint space
        if self.is_jointspace_controller:
            # it's joint space

            target_workspace_positions = np.zeros((len(times), 3))
            target_workspace_velocities = np.zeros((len(times), 3))
            target_workspace_quaternions = np.zeros((len(times), 4))

            for i in range(len(times)):
                positions_dict = joint_array_to_dict(target_positions[i], self._limb)
                target_workspace_positions[i, :] = \
                    self._kin.forward_position_kinematics(joint_values=positions_dict)[:3]
                target_workspace_velocities[i, :] = \
                    self._kin.jacobian(joint_values=positions_dict)[:3].dot(target_velocities[i])
                target_workspace_quaternions[i, :] = np.array([0, 1, 0, 0])

            # Plot joint space
            plt.figure()
            joint_num = len(self._limb.joint_names())
            for joint in range(joint_num):
                plt.subplot(joint_num,2,2*joint+1)
                plt.plot(times, actual_positions[:,joint], label='Actual')
                plt.plot(times, target_positions[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Position Error")
                plt.legend()

                plt.subplot(joint_num,2,2*joint+2)
                plt.plot(times, actual_velocities[:,joint], label='Actual')
                plt.plot(times, target_velocities[:,joint], label='Desired')
                plt.xlabel("Time (t)")
                plt.ylabel("Joint " + str(joint) + " Velocity Error")
                plt.legend()
            print("Close the plot window to continue")
            plt.show()

        else:
            # it's workspace
            target_workspace_positions = target_positions
            target_workspace_velocities = target_velocities
            target_workspace_quaternions = np.zeros((len(times), 4))
            target_workspace_quaternions[:, 1] = 1

        plt.figure()
        workspace_joints = ('X', 'Y', 'Z')
        joint_num = len(workspace_joints)
        for joint in range(joint_num):
            plt.subplot(joint_num,2,2*joint+1)
            plt.plot(times, actual_workspace_positions[:,joint], label='Actual')
            plt.plot(times, target_workspace_positions[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Position Error")
            plt.legend()

            plt.subplot(joint_num,2,2*joint+2)
            plt.plot(times, actual_velocities[:,joint], label='Actual')
            plt.plot(times, target_velocities[:,joint], label='Desired')
            plt.xlabel("Time (t)")
            plt.ylabel(workspace_joints[joint] + " Velocity Error")
            plt.legend()

        print("Close the plot window to continue")
        plt.show()

        # Plot orientation error. This is measured by considering the
        # axis angle representation of the rotation matrix mapping
        # the desired orientation to the actual orientation. We use
        # the corresponding angle as our metric. Note that perfect tracking
        # would mean that this "angle error" is always zero.
        angles = []
        for t in range(len(times)):
            quat1 = target_workspace_quaternions[t]
            quat2 = actual_workspace_quaternions[t]
            theta = axis_angle(quat1, quat2)
            angles.append(theta)

        plt.figure()
        plt.plot(times, angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Error Angle of End Effector (rad)")
        print("Close the plot window to continue")
        plt.show()
        

    def execute_path(self, path, rate=200, timeout=None, log=False):
        """
        takes in a path and moves the baxter in order to follow the path.  

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For plotting
        if log:
            times = list()
            actual_positions = list()
            actual_velocities = list()
            target_positions = list()
            target_velocities = list()

        # For interpolation
        max_index = len(path.joint_trajectory.points)-1
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zerogripper
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            (
                target_position, 
                target_velocity, 
                target_acceleration, 
                current_index
            ) = self.interpolate_path(path, t, current_index)

            #print("target pos", target_position)
            #print("actual pos", current_position)

            # For plotting
            if log:
                times.append(t)
                actual_positions.append(current_position)
                actual_velocities.append(current_velocity)
                target_positions.append(target_position)
                target_velocities.append(target_velocity)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                self.stop_moving()
                break
            
            print("execute loop time: ", ((rospy.Time.now() - start_t).to_sec() - t))

        if log:
            self.plot_results(
                times,
                actual_positions, 
                actual_velocities, 
                target_positions, 
                target_velocities
            )
        return True

    def follow_ar_tag(self, tag, rate, timeout=1500, log=False):
        """
        takes in an AR tag number and follows it with the baxter's arm.  You 
        should look at execute_path() for inspiration on how to write this. 

        Parameters
        ----------
        tag : int
            which AR tag to use
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller pgrippererformance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)
        ik_solver = IK("base", "right_hand")

        # tag looking
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)

        to_frame = 'ar_marker_12'
        done = True

        while not rospy.is_shutdown():
            t = (rospy.Time.now() - start_t).to_sec()
            if timeout is not None and t >= timeout:
                # Set velocities to zerogripper
                self.stop_moving()
                return False

            while not rospy.is_shutdown():
                try:
                    trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), rospy.Duration(10.0))
                    trans2 = tfBuffer.lookup_transform('base', 'stp_022412TP99883_tip', rospy.Time(0), rospy.Duration(10.0))
                    break
                except Exception as e:
                    print("Retrying ...")

            
            current_position = np.array([getattr(trans2.transform.translation, dim) for dim in ('x', 'y', 'z')])
            current_joint_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)
            
            updatedTagPos = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
            #target_pos[2] += 0.5
            updatedTagPos[2] += 0.6
            desiredPosition = updatedTagPos
            dist = updatedTagPos - current_position
            workspaceVel = (updatedTagPos - current_position)/(1/rate)
            print("workspaceVel: ", workspaceVel)

            numInterSteps = 50
            avg = 1 # avg 5 IK solves to ensure accuracy
            innerR = rospy.Rate(rate * numInterSteps) # rate for intermediate step calc & execution
            innerT = 1 / (rate * numInterSteps)

            print("tolerated?", np.linalg.norm(dist) > 0.01)
            if (np.linalg.norm(dist) > 0.01):
                for i in range(numInterSteps):
                    delta_t = 0.001
                    intermediatePos = desiredPosition + i * innerT * workspaceVel
                    intermediatePosbefore = intermediatePos - workspaceVel * delta_t

                    intermediateJointPositions, unsuccfirstIk, succIKs1 = 0, 0, 0
                    while succIKs1 < avg:
                        IKcalc = np.array(ik_solver.get_ik(current_joint_position, intermediatePos[0], intermediatePos[1], intermediatePos[2], 0, 1, 0, 0))
                        
                        if IKcalc is None:
                            unsuccfirstIk += 1
                            if unsuccfirstIk > 10:
                                rospy.signal_shutdown('MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(i))
                                print('MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(i))
                                return None
                            continue
                        else:
                            succIKs1 += 1
                            intermediateJointPositions += IKcalc
                        
                    intermediateJointPosition = intermediateJointPositions/succIKs1
                    
                    intermediateJointPositionbefores, unsuccsecondIK, succIKs2 = 0, 0, 0 
                    while succIKs2 < avg:
                        IKcalc = np.array(ik_solver.get_ik(current_joint_position, intermediatePosbefore[0], intermediatePosbefore[1], intermediatePosbefore[2], 0, 1, 0, 0))

                        if IKcalc is None:
                            unsuccsecondIK += 1
                            if unsuccsecondIK > 10:
                                rospy.signal_shutdown('MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(i))
                                print('MAX IK ATTEMPTS EXCEEDED AT x(t)={}'.format(i))
                                return None
                            continue
                        else:
                            succIKs2 += 1
                            intermediateJointPositionbefores += IKcalc
                        
                    intermediateJointPositionbefore = intermediateJointPositionbefores/succIKs2
                    
                    jointVel = (intermediateJointPositionbefore - intermediateJointPosition) / delta_t

                    # theta = ik_solver.get_ik(seed,
                    #         x[0], x[1], x[2],      # XYZ
                    #         x[3], x[4], x[5], x[6] # quat
                    #         )
                    # current_joint_position = intermediateJointPosition
                    current_joint_position = get_joint_positions(self._limb)
                    #print("CURRENT Joint POSITION:", current_joint_position)

                    #print("Target Joint Vel num:", i,  jointVel)
                    
                    self.step_control(intermediateJointPosition, jointVel, None)
                    
                    innerR.sleep()
            
            r.sleep()

            print("loop time: ", ((rospy.Time.now() - start_t).to_sec() - t))
        
        return True

class FeedforwardJointVelocityController(Controller):
    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        self._limb.set_joint_velocities(joint_array_to_dict(targgripperet_velocity, self._limb))

class WorkspaceVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  The difference between this controller and the
    PDJointVelocityController is that this controller compares the baxter's current WORKSPACE position and
    velocity desired WORKSPACE position and velocity to come up with a WORKSPACE velocity command to be sent
    to the baxter.  Then this controller should convert that WORKSPACE velocity command into a joint velocity
    command and sends that to the baxter.  Notice the shape of Kp and Kv
    """
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 6x' :obj:`numpy.ndarray`
        Kv : 6x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time.
        target_position will be a 7 vector describing the desired SE(3) configuration where the first
        3 entries are the desired position vector and the next 4 entries are the desired orientation as
        a quaternion, all written in spatial coordinates.
        target_velocity is the body-frame se(3) velocity of the desired SE(3) trajectory gd(t). This velocity
        is given as a 6D Twist (vx, vy, vz, wx, wy, wz).
        This method should call self._kin.forward_position_kinematics() to get the current workspace 
        configuration and self._limb.set_joint_velocities() to set the joint velocity to something.  
        Remember that we want to track a trajectory in SE(3), and implement the controller described in the
        project document PDF.
        Parameters
        ----------
        target_position: (7,) ndarray of desired SE(3) position (px, py, pz, qx, qy, qz, qw) (position + quaternion).
        target_velocity: (6,) ndarray of desired body-frame se(3) velocity (vx, vy, vz, wx, wy, wz).
        target_acceleration: ndarray of desired accelerations (should you need this?).
        """
        raise NotImplementedError
        control_input = None        
        self._limb.set_joint_velocities(joint_array_to_dict(control_input, self._limb))


class PDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  This controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the sawyer's 
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the sawyer.
    """
    def __init__(self, limb, kin, Kp, Ki, Kd, Kw):
        """
        Parameters
        ----------
        limb : :obj:`sawyer_interface.Limb`
        kin : :obj:`sawyerKinematics`
        Kp : 7x' :obj:`numpy.ndarray` of proportional constants
        Ki: 7x' :obj:`numpy.ndarray` of integral constants
        Kd : 7x' :obj:`numpy.ndarray` of derivative constants
        Kw : 7x' :obj:`numpy.ndarray` of anti-windup constants
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Ki = np.diag(Ki)
        self.Kd = np.diag(Kd)
        self.Kw = Kw
        self.integ_error = np.zeros(7)
        self.kin = kin
        self.limb = limb
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position 
        according to the input path and the current time. Each Controller below extends this 
        class, and implements this accordingly. This method should call
        self._limb.joint_angle and self._limb.joint_velocity to get the current joint position and velocity
        and self._limb.set_joint_velocities() to set the joint velocity to something.  You may find
        joint_array_to_dict() in utils.py useful

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)
        
        # TODO: implement PID control to set the joint velocities. 

        error = target_position - current_position
        proportional = np.dot(self.Kp, error)

        self.integ_error = (self.Kw * self.integ_error) + error
        integral = np.dot(self.Ki, self.integ_error)

        err_d = target_velocity - current_velocity
        derivative = np.dot(self.Kd, err_d)

        controller_velocity = proportional + integral + derivative

        self._limb.set_joint_velocities(joint_array_to_dict(controller_velocity, self._limb))
    
    # def follow_ar_tag(self, tag, rate, timeout=1500, log=False):
    #     """
    #     takes in an AR tag number and follows it with the baxter's arm.  You 
    #     should look at execute_path() for inspiration on how to write this. 

    #     Parameters
    #     ----------
    #     tag : int
    #         which AR tag to use
    #     rate : int
    #         This specifies how many ms between loops.  It is important to
    #         use a rate and not a regular while loop because you want the
    #         loop to refresh at a constant rate, otherwise you would have to
    #         tune your PD parameters if the loop runs slower / faster
    #     timeout : int
    #         If you want the controller to terminate after a certain number
    #         of seconds, specify a timeout in seconds.
    #     log : bool
    #         whether or not to display a plot of the controller pgrippererformance

    #     Returns
    #     -------
    #     bool
    #         whether the controller completes the path or not
    #     """

    #     # For timing
    #     start_t = rospy.Time.now()
    #     r = rospy.Rate(rate)
    #     ik_solver = IK("base", "right_hand")

    #     # tag looking
    #     tfBuffer = tf2_ros.Buffer()
    #     listener = tf2_ros.TransformListener(tfBuffer)

    #     to_frame = 'ar_marker_11'
    #     done = True

    #     while not rospy.is_shutdown():
    #         t = (rospy.Time.now() - start_t).to_sec()
    #         if timeout is not None and t >= timeout:
    #             # Set velocities to zerogripper
    #             self.stop_moving()
    #             return False

    #         while not rospy.is_shutdown():
    #             try:
    #                 trans = tfBuffer.lookup_transform('base', to_frame, rospy.Time(0), rospy.Duration(10.0))
    #                 trans2 = tfBuffer.lookup_transform('base', 'stp_022412TP99883_tip', rospy.Time(0), rospy.Duration(10.0))
    #                 break
    #             except Exception as e:
    #                 print("Retrying ...")

    #         updatedTagPos = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])


    #         print("mid loop: ", ((rospy.Time.now() - start_t).to_sec() - t))

    #         updated_traj, ShouldMove = get_traj(self._limb, self._kin, trans2, ik_solver, updatedTagPos, rate=rate, num_way=1) # in this case rate is how much time given to controller for path
    #         print("traj gen: ",  ((rospy.Time.now() - start_t).to_sec() - t))
    #         if (ShouldMove):
    #             self.execute_path(updated_traj, rate=rate/10, timeout=rate/1000, log=False)
    #         print("upper loop: ", ((rospy.Time.now() - start_t).to_sec() - t))
    #         r.sleep()

    #         print("loop time: ", ((rospy.Time.now() - start_t).to_sec() - t))
        
    #     return True



class PDJointTorqueController(Controller):
    def __init__(self, limb, kin, Kp, Kv):
        """
        Parameters
        ----------
        limb : :obj:`baxter_interface.Limb`
        kin : :obj:`BaxterKinematics`
        Kp : 7x' :obj:`numpy.ndarray`
        Kv : 7x' :obj:`numpy.ndarray`
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Kv = np.diag(Kv)
        self.is_jointspace_controller = True

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Makes a call to the robot to move according to its current position and the desired position 
        according to the input path and the current time. This method should call
        get_joint_positions and get_joint_velocities from the utils package to get the current joint 
        position and velocity and self._limb.set_joint_torques() to set the joint torques to something. 
        You may find joint_array_to_dict() in utils.py useful as well.
        Recall that in order to implement a torque based controller you will need access to the 
        dynamics matrices M, C, G such that
        M ddq + C dq + G = u
        For this project, you will access the inertia matrix and gravity vector as follows:
        Inertia matrix: self._kin.inertia(positions_dict)
        Coriolis matrix: self._kin.coriolis(positions_dict, velocity_dict)
        Gravity matrix: self._kin.gravity(positions_dict) (You might want to scale this matrix by 0.01 or another scalar)
        These matrices were computed by a library and the coriolis matrix is approximate, 
        so you should think about what this means for the kinds of trajectories this 
        controller will be able to successfully track.
        Look in section 4.5 of MLS.
        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        raise NotImplementedError
        control_input = None
        self._limb.set_joint_torques(joint_array_to_dict(control_input, self._limb))
