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

        self._p = bc.BulletClient(connection_mode=p2.GUI)
        
        cfg = YAML().load(open(__RSCDIR__ + "/config.yaml", 'r'))
        
        self._renders = cfg['environment']['render']
        self._discrete_actions = discrete_actions
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1

        self._spot = ArticulatedSystem("spot.urdf", self._p, cfg)

        self.seed()
        # self.reset()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class ArticulatedSystem:
    def __init__(self, urdf_dir, p, cfg):

        self._spot = p.loadURDF(urdf_dir, [0., 0., 0.])
        self.nJoints = p2.getNumJoints(self._spot)
        self.gcDim = self.nJoints + 6 # number of joints + base x, y, z, r, p, y
        self.gvDim = self.nJoints + 7 # number of joints + base x, y, z, body quat
        self.actionDim = self.nJoints + 1

        buffer_len = cfg["environment"]["bufferLength"]
        self._obDim = 1 + 3 + 12 + 3 + 3 + 12 + 3 + 3*self.actionDim + 1 + 4*5 + buffer_len * (12 + 12) - 4 + 3 * buffer_len - 1
        
        obs_high = np.array([np.inf] * self._obDim)
        action_high = np.array([np.inf] * self.actionDim)
        
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)




