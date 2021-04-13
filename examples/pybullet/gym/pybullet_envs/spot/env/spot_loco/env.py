"""
Locomotion controller for Spot
"""

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# TODO: double check what all these packages do...
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
import pybullet_data
from pybullet_utils import bullet_client as bc
from pkg_resources import parse_version

from ruamel.yaml import YAML, dump, RoundTripDumper
from env.spot_loco import __SPOT_LOCO_RESOURCE_DIRECTORY__ as __RSCDIR__

root = os.path.dirname(os.path.abspath(__file__)) + '/../'
log_path = root + '/data/SpotLoco'

logger = logging.getLogger(__name__)

class SpotBulletEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self, renders=False, discrete_actions=False):
        
        self.cfg = YAML().load(open(__RSCDIR__ + "/config.yaml", 'r'))

        self._p = bc.BulletClient(connection_mode=p2.GUI)
        self._p.setTimeStep(self.cfg["environment"]["simulation_dt"])
        
        self._renders = self.cfg['environment']['render']
        self._discrete_actions = discrete_actions
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1

        self.spot = ArticulatedSystem(self._p, self.cfg)

        self.seed()
        # self.reset()
        self.viewer = None
        self._configure()

        self.gait_freq_ = 0.
        self.action_history = np.zeros((self.spot.ACTION_DIM*3,))
        self.err_prev = np.zeros((self.spot.N_JOINTS,))
        self.err_curr = np.zeros((self.spot.N_JOINTS,))
        self.body_linear_vel_ = np.zeros((3,))
        self.body_angular_vel_ = np.zeros((3,))
        self.target_vel_ = np.zeros((3,))

        self.gait_params_ = {
            "stride" : 0.6,
            "max_foot_height" : 0.08,
            "phase" : 0.,
            "swing_start" : [0., 0.5, 0.5, 0.],
            "swing_duration" : [0.5, 0.5, 0.5, 0.5],
            "phase_time_left" : [1., 1., 1., 1.],
            "foot_target" : [0.08, 0.08, 0.08, 0.08],
            "foot_position" : [0., 0., 0., 0.],
            "is_stance_gait" : False,
            "desired_contact_states" : [True, True, True, True]
        }

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        p = self._p

        unscaled_action = self.spot.unscale_action(action)
        self.gait_freq_ = unscaled_action[-1]
        joint_targets = unscaled_action[:-1]

        # update the action history buffer
        temp = self.action_history[-self.spot.ACTION_DIM*2:].copy()
        self.action_history[-self.spot.ACTION_DIM:] = joint_targets
        self.action_history[:2*self.spot.ACTION_DIM] = temp

        loop_count = self.cfg["environment"]["control_dt"] // self.cfg["environment"]["simulation_dt"]
        for _ in range(loop_count):

            joint_velocities = []
            joint_positions = []
            self.err_prev = self.err_curr
            for i in range(self.spot.N_JOINTS):
                joint_velocity, joint_position = p.getJointState(self.spot.spot, i)
                joint_velocities.append(joint_velocity)
                joint_positions.append(joint_position)

            self.err_curr = joint_targets - np.array(joint_positions)

            effort = self.spot.P_GAIN * self.err_curr + self.spot.D_GAIN * (self.err_curr - self.err_prev) / self.cfg["environment"]["simulation_dt"]

            self.err_prev = self.err_curr

            for i in range(self.spot.N_JOINTS):
                p.setJointMotorControl2(self.spot.spot, i, p.TORQUE_CONTROL, force=effort[i])

            p.stepSimulation()

        obs_scaled = self.update_observation()
        reward = self.compute_reward()
        done = self.is_terminal_state()

        self.update_gait_parameters()
        self.sample_velocity_target()

        return obs_scaled, reward, done

    def update_observation(self):
        gc, gv = self.spot.get_state()
        foot_contacts = self.spot.get_foot_contacts()

        obs_unscaled = np.zeros((self.spot.OB_DIM,)); pos = 0

        # body orientation
        rot_mat = self._p.getMatrixFromQuaternion(gc[3:7])
        obs_unscaled[pos:pos+3] = rot_mat[-1]; pos += 3
        rot_mat = np.array(rot_mat).reshape((3, -1))

        # joint angles
        obs_unscaled[pos:pos+self.spot.N_JOINTS] = gc[-self.spot.N_JOINTS:]; pos += self.spot.N_JOINTS

        # body velocities
        self.body_linear_vel_ = np.matmul(rot_mat.transpose(), gv[:3, np.newaxis])[:]
        self.body_angular_vel_ = np.matmul(rot_mat.transpose(), gv[3:6, np.newaxis])[:]
        obs_unscaled[pos:pos+3] = self.body_linear_vel_; pos += 3
        obs_unscaled[pos:pos+3] = self.body_angular_vel_; pos += 3

        # joint velocities
        obs_unscaled[pos:pos+self.spot.N_JOINTS] = gv[-self.spot.N_JOINTS:]; pos += self.spot.N_JOINTS

        # target velocity
        obs_unscaled[pos:pos+3] = self.target_vel_; pos += 3

        # action history
        obs_unscaled

        return

    def compute_reward(self):
        return

    def is_terminal_state(self):
        return
    
class ArticulatedSystem:
    def __init__(self, p, cfg):
        
        self.p = p
        self.spot = p.loadURDF(cfg["urdf"], [0., 0., 0.])
        self.N_JOINTS = p2.getNumJoints(self._spot)
        self.GC_DIM = self.N_JOINTS + 6 # number of joints + base x, y, z, r, p, y
        self.GV_DIM = self.N_JOINTS + 7 # number of joints + base x, y, z, body quat
        self.ACTION_DIM = self.N_JOINTS + 1 # joint angle targets + gait frequency

        BUFFER_LEN = cfg["environment"]["bufferLength"]
        self.OB_DIM = 1 + 3 + 12 + 3 + 3 + 12 + 3 + 3*self.ACTION_DIM + 1 + 4*5 + BUFFER_LEN * (12 + 12) - 4 + 3 * BUFFER_LEN - 1
        self.BUFFER_STRIDE = cfg["environment"]["bufferStride"]
        
        obs_high = np.array([np.inf] * self.OB_DIM)
        action_high = np.array([np.inf] * self.ACTION_DIM)
        
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.P_GAIN = 100.
        self.D_GAIN = 2.

        self.TARGET_VEL_RANGE = {
            "x" : cfg["environment"]["x"],
            "y" : cfg["environment"]["y"],
            "yaw" : cfg["environment"]["yaw"]
        }

        self.GC_INIT = [0., 0., 0., 0.54, 1., 0., 0., 0., -0.1, 1.1, -1.9, 0.1, 1.1, -1.9, -0.1, 1.1, -1.9, 0.1, 1.1, -1.9]
        self.ACTION_MEAN = np.array(self.GC_INIT.append(0.))
        self.ACTION_STD = np.array([0.6] * self.ACTION_DIM)
        self.ACTION_STD[-1] = 0.1

        self.OB_MEAN = np.zeros((self.OB_DIM,)); pos = 0
        self.OB_MEAN[pos:pos+3] = 0.; pos += 3 # gravity axis
        self.OB_MEAN[pos:pos+self.N_JOINTS] = np.array(self.GC_INIT[-self.N_JOINTS:]); pos += self.N_JOINTS # joint position
        self.OB_MEAN[pos:pos+6] = 0.; pos += 6 # body lin/ang velocity
        self.OB_MEAN[pos:pos+self.N_JOINTS] = 0.; pos += self.N_JOINTS # joint velocities
        self.OB_MEAN[pos:pos+3] = 0.; pos += 3 # target velocity
        self.OB_MEAN[pos:pos+3*self.ACTION_DIM] = 0.; pos += 3*self.ACTION_DIM
        self.OB_MEAN[pos] = 0.; pos += 1 # stride
        self.OB_MEAN[pos:pos+4] = 0.; pos += 4 # swing start
        self.OB_MEAN[pos:pos+4] = 0.; pos += 4 # swing duration
        self.OB_MEAN[pos:pos+4] = 0.; pos += 4 # phase time left
        self.OB_MEAN[pos:pos+4] = 0.; pos += 4 # desired contact states
        self.OB_MEAN[pos:pos+self.N_JOINTS*BUFFER_LEN] = 0.; pos += self.N_JOINTS*BUFFER_LEN # joint position history
        self.OB_MEAN[pos:pos+self.N_JOINTS*BUFFER_LEN] = 0.; pos += self.N_JOINTS*BUFFER_LEN # joint velocity history
        self.OB_MEAN[pos:pos+3*BUFFER_LEN] = 0. # body velocity history

        self.OB_STD = np.zeros((self.OB_DIM,)); pos = 0
        self.OB_STD[pos:pos+3] = 0.7; pos += 3 # gravity axis
        self.OB_STD[pos:pos+self.N_JOINTS] = 1.; pos += self.N_JOINTS # joint angles
        self.OB_STD[pos:pos+3] = 2.; pos += 3 # linear velocity
        self.OB_STD[pos:pos+3] = 4.; pos += 3 # angular velocity
        self.OB_STD[pos:pos+self.N_JOINTS] = 10.; pos += self.N_JOINTS # joint velocities
        self.OB_STD[pos] = self.TARGET_VEL_RANGE["x"][-1]; pos += 1 # target velocity x
        self.OB_STD[pos] = self.TARGET_VEL_RANGE["y"][-1]; pos += 1 # target velocity y
        self.OB_STD[pos] = self.TARGET_VEL_RANGE["yaw"][-1]; pos += 1 # target velocity yaw
        self.OB_STD[pos:pos+3*self.ACTION_DIM] = 1.; pos += 3*self.ACTION_DIM # action history
        self.OB_STD[pos] = 1.; pos += 1 # stride
        self.OB_STD[pos:pos+4] = 1.; pos += 4 # swing start
        self.OB_STD[pos:pos+4] = 1.; pos += 4 # swing duration
        self.OB_STD[pos:pos+4] = 1.; pos += 4 # phase time left
        self.OB_STD[pos:pos+4] = 1.; pos += 4 # desired contact states
        self.OB_STD[pos:pos+self.N_JOINTS*BUFFER_LEN] = 1.; pos += self.N_JOINTS*BUFFER_LEN # joint position history
        self.OB_STD[pos:pos+self.N_JOINTS*BUFFER_LEN] = 1.; pos += self.N_JOINTS*BUFFER_LEN # joint velocity history
        self.OB_STD[pos:pos+3*BUFFER_LEN] = 1. # body velocity error

        self.stance_gait_params = {
            "is_stance_gait": True,
            "swing_start": [0., 0., 0., 0.],
            "swing_duration": [0., 0., 0., 0.],
            "foot_target_height": [0., 0., 0., 0.],
        }

        self.REWARD_COEFFS = {
            "torque" : cfg["environment"]["torqueRewardCoeff"],
            "target_following" : cfg["environment"]["targetFollowingRewardCoeff"],
            "action_smoothness" : cfg["environment"]["actionSmoothnessRewardCoeff"],
            "body_orientation" : cfg["environment"]["bodyOrientationRewardCoeff"],
            "symmetry" : cfg["environment"]["symmetryRewardCoeff"],
            "foot_clearance" : cfg["environment"]["footClearanceRewardCoeff"],
            "gait_following" : cfg["environment"]["gaitFollowingRewardCoeff"],
            "vertical_velocity" : cfg["environment"]["verticalVelocityRewardCoeff"],
            "body_rates" : cfg["environment"]["bodyRatesRewardCoeff"],
            "body_height" : cfg["environment"]["bodyHeightRewardCoeff"],
            "internal_contact" : cfg["environment"]["internalContactRewardCoeff"],
            "joint_velocity" : cfg["environment"]["jointVelocityRewardCoeff"],
        }

        self.FOOT_INDICES = [2, 5, 8, 11]

        self.gc_ = np.zeros((self.GC_DIM,))
        self.gv_ = np.zeros((self.GV_DIM,))
        self.gf_ = np.zeros((self.GV_DIM,))
        self.foot_contacts = np.array([True] * 4, dtype=bool)

        
    def unscale_action(self, scaled_action):

        return np.multiply(self.ACTION_STD, scaled_action) + self.ACTION_MEAN

    def scale_observation(self, unscaled_observation):

        return np.divide(unscaled_observation - self.OB_MEAN, self.OB_STD)

    def get_state(self):
        """
        Get the generalized coordinates, generalized velocities, generalized forces for Spot
        """

        for i in range(self.spot.N_JOINTS):
            self.gv_[6+i], self.gc_[7+i] = self.p.getJointState(self.spot, i)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.spot)
        self.gv_[:3] = np.array(lin_vel)
        self.gv_[3:6] = np.array(ang_vel)

        base_pos, base_orientation = self.p.getBasePositionAndOrientation(self.quadruped)
        self.gc_[:3] = np.array(base_pos)
        self.gc_[3:7] = np.array(base_orientation)

        return self.gc_.copy(), self.gv_.copy()

    def get_foot_contacts(self):

        for i in range(4):
            if self.p.getContactPoints(self.spot, -1, self.FOOT_INDICES[i]):
                self.foot_contacts[i] = True

        return self.foot_contacts.copy()

