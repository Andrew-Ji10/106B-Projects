from time import sleep
import SimpleArm_dynamics
import numpy as np
import matplotlib.pyplot as plt
import pyglet
from pyglet import shapes
from trajectories import LinearTrajectory, CircularTrajectory, PolygonalTrajectory
from controllers import JointVelocityController, JointTorqueController, WorkspaceVelocityController
import colorsys
import argparse

"""
2DOF manipulator simulation. Used for testing controllers and paths
C106B Project 1A
"""

trajectory = None
controller = None

class SimpleArmSim(pyglet.window.Window):

    def __init__(self, width, height):
        """
        Parameters
        ----------
        width: width of display window in pixels
        height: height of display window in pixels
        trajectory: desired trajectory for manipulator to follow. Will remove orientation and z-axis information
        controller: controller used to execute desired trajectory
        """

        # Define constants for for the manipulator
        self.l1 = 8
        self.l2 = 6
        m1 = 2
        m2 = 1
        I1 = (1/12) * m1 * self.l1**2
        I2 = (1/12) * m2 * self.l2**2
        g = 9.8
        constants = [self.l1, self.l2, m1, m2, I1, I2, g]

        # Setup condensed functions for getting important matrices
        self.M_func = lambda q, q_dot: SimpleArm_dynamics.M_func(*(constants + [q, q_dot]))
        self.C_func = lambda q, q_dot: SimpleArm_dynamics.C_func(*(constants + [q, q_dot]))
        self.G_func = lambda q, q_dot: SimpleArm_dynamics.G_func(*(constants + [q, q_dot]))
        self.J_body_func = lambda q, q_dot: SimpleArm_dynamics.J_body_func(*(constants + [q, q_dot]))

        # Other simulation constants
        self.dt = 1/60
        super().__init__(width, height, "2DOF Manipulator Simulation")
        self.time = 0
        self.batch = pyglet.graphics.Batch()
        self.pixel_origin = [width/2, 50]

        # Constants for visualization
        link_w = 0.25
        board_w = link_w/2
        self.pm = 25 # pixels per meter
        self.deg2rad = 180/np.pi

        # Manipulator Links
        self.link1 = shapes.Rectangle(
            x=self.pixel_origin[0],
            y=self.pixel_origin[1],
            width=link_w*self.pm,
            height=self.l1*self.pm,
            color=(87, 74, 226),
            batch=self.batch)
        self.link1.anchor_x = (link_w/2)*self.pm
        self.link1.anchor_y = 0

        self.link2 = shapes.Rectangle(
            x=self.pixel_origin[0],
            y=self.pixel_origin[1] + self.l1*self.pm,
            width=link_w*self.pm,
            height=self.l2*self.pm,
            color=(226, 173, 242),
            batch=self.batch)
        self.link2.anchor_x = (link_w/2)*self.pm
        self.link2.anchor_y = 0

        # Manipulator states
        self.q = np.array([
            0,
            0
        ], dtype=np.float32)
        self.q_dot = np.array([
            0,
            0
        ], dtype=np.float32)

        if trajectory:
            # Set initial configuration to trajectory start
            sol = self.ik(trajectory.target_pose(0)[:2]) # We only need xy
            self.q = sol

            # Compute IK over entire trajectory
            self.ik_sols = [sol]
            self.joint_velocity_sols = []
            self.joint_acceleration_sols = []

            # Add circles to visualize trajectory
            self.circles = []
            count = 0
            for t in np.arange(0, trajectory.total_time, self.dt):
                pos = trajectory.target_pose(t)[:2] # We only need xy
                self.ik_sols.append(self.ik(pos, self.ik_sols[-1]))
                self.joint_velocity_sols.append((self.ik_sols[-1] - self.ik_sols[-2])/self.dt)
                if len(self.joint_velocity_sols) > 1:
                    self.joint_acceleration_sols.append((self.joint_velocity_sols[-1] - self.joint_velocity_sols[-2])/self.dt)
                if count % 20 == 0:
                    rgb = colorsys.hsv_to_rgb(t/trajectory.total_time, 1, 1)
                    rgb_norm = [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)]
                    circle = shapes.Circle(
                        x=pos[0]*self.pm + self.pixel_origin[0],
                        y=pos[1]*self.pm + self.pixel_origin[1],
                        radius=3,
                        color=rgb_norm,
                        batch=self.batch)
                    self.circles.append(circle)
                count += 1
    
    def ang_diff(self, a1, a2):
        """ Helper function for IK """

        diff = np.mod(np.abs(a1 - a2), 2 * np.pi)
        if diff < np.pi:
            return diff
        else:
            return 2 * np.pi - diff

    def ik(self, pos, seed=np.zeros(2)):
        """
        Return IK solution for desired xy position using MLS 3.1

        Parameters
        ----------
        pos - xy position of end effector
        seed - prevous IK solution

        Returns
        -------
        sol - joint angles corresponding to xy position of end effector
        """

        r = np.linalg.norm(pos)
        cos_alpha = (self.l1**2 + self.l2**2 - r**2)/(2*self.l1*self.l2)
        if cos_alpha > 1 or cos_alpha < -1:
            raise Exception("IK Solver Failed")
        alpha = np.arccos(cos_alpha)
        theta2_sol1 = np.pi + alpha
        theta2_sol2 = np.pi - alpha

        cos_beta = (r**2 + self.l1**2 - self.l2**2)/(2*self.l1*r)
        beta = np.arccos(cos_beta)
        if cos_beta > 1 or cos_beta < -1:
            raise Exception("IK Solver Failed")
        theta1_sol1 = np.arctan2(-pos[0], pos[1]) + beta
        theta1_sol2 = np.arctan2(-pos[0], pos[1]) - beta
        diff1 = self.ang_diff(seed[0], theta1_sol1) + self.ang_diff(seed[1], theta2_sol1)
        diff2 = self.ang_diff(seed[0], theta1_sol2) + self.ang_diff(seed[1], theta2_sol2)

        # Return solution closest to previous solution
        if diff1 < diff2:
            return np.array([theta1_sol1, theta2_sol1])
        else:
            return np.array([theta1_sol2, theta2_sol2])

    def xy(self):
        """Return positions of junction and end effector"""

        first_tip = np.array([
            -self.l1*np.sin(self.q[0]),
            self.l1*np.cos(self.q[0])
        ])
        second_tip = np.array([
            -self.l2*np.sin(self.q[0] + self.q[1]),
            self.l2*np.cos(self.q[0] + self.q[1])
        ])
        return first_tip, first_tip + second_tip

    def on_draw(self):
        """Clear the screen and draw shapes"""

        self.clear()
        self.batch.draw()
    
    def update_kinematic(self):
        """For controllers that require a kinematic (instantaneous velocity change) model"""

        if isinstance(controller, JointVelocityController):
            # Get current desired joint angle velocity
            index = int(self.time // self.dt)
            safe_index = lambda i: max(min(i, len(self.joint_velocity_sols) - 1), 0)
            target_velocity = self.joint_velocity_sols[safe_index(index)]
            joint_velocity = controller.step_control(None, target_velocity, None)
            self.q_dot = joint_velocity
        
        elif isinstance(controller, WorkspaceVelocityController):
            # Get current desired end effector velocity
            target_velocity = trajectory.target_velocity(self.time)[:2] # We only need xy
            joint_velocity = controller.step_control(None, target_velocity, None)
            self.q_dot = joint_velocity

        # Integrate state
        self.q = self.q + self.q_dot * self.dt
        self.update_frame()

    def update_dynamic(self):
        """ For controllers that require a dynamic model """

        M = self.M_func(self.q, self.q_dot) 
        C = self.C_func(self.q, self.q_dot) 
        G = self.G_func(self.q, self.q_dot) 

        joint_torque = np.zeros((4, 1))
        if isinstance(controller, JointTorqueController):
            # Get current desired joint torque
            index = int(self.time // self.dt)
            safe_index = lambda i: max(min(i, len(self.joint_acceleration_sols) - 1), 0)
            target_acceleration = self.joint_acceleration_sols[safe_index(index)]
            joint_torque = controller.step_control(None, None, target_acceleration)

        # Simulate dynamics
        q_ddot = np.matmul(np.linalg.inv(M), -(np.matmul(C, self.q_dot.reshape(-1, 1)) + G) + joint_torque).reshape(-1)
        self.q_dot = self.q_dot + q_ddot * self.dt
        self.q = self.q + self.q_dot * self.dt
        self.update_frame()
    
    def update_frame(self):
        """Animate the shapes"""

        self.time += self.dt

        self.link1.rotation = -self.q[0] * self.deg2rad
        self.link2.rotation = -(self.q[0] + self.q[1]) * self.deg2rad

        junction, end = self.xy()
        self.link2.x = junction[0]*self.pm + self.pixel_origin[0]
        self.link2.y = junction[1]*self.pm + self.pixel_origin[1]

def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory()
    elif args.task == 'circle':
        trajectory = CircularTrajectory()
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory()
    return trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('-controller_name', '-c', type=str, default='jointspace', 
        help='Options: jointspace, workspace, or torque.  Default: jointspace'
    )
    args = parser.parse_args()

    # Width and height in pixels of simulator. Adjust as needed
    width = 500
    height = 500

    trajectory = define_trajectories(args)

    sim = SimpleArmSim(width, height)

    if args.controller_name == 'jointspace':
        controller = JointVelocityController(sim)
    if args.controller_name == 'workspace':
        controller = WorkspaceVelocityController(sim)
    elif args.controller_name == 'torque':
        controller = JointTorqueController(sim)

    if trajectory and controller:
        if args.controller_name == 'jointspace':
            update_func = sim.update_kinematic
        if args.controller_name == 'workspace':
            update_func = sim.update_kinematic
        elif args.controller_name == 'torque':
            update_func = sim.update_dynamic

        while(sim.time < trajectory.total_time):
            # Integrate the dynamics
            update_func()

            # Render scene
            sim.dispatch_events()
            sim.dispatch_event('on_draw')
            sim.flip()

            # Wait until finshed with dt if needed
            delta_time = pyglet.clock.tick()
            time_left = max(sim.dt - delta_time, 0)
            sleep(time_left)
