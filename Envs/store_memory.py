# ================== DQNAgent Env ======================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

from absl import logging

from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import circular_replay_buffer
import gin.tf
import numpy as np
import tensorflow as tf

# DQNAgent
import DQNAgent

# ================== Carla Env =========================
from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

# carla env
import gym
import gym_carla
import carla

# ================ Replay Buffer ===================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os
import pickle

from absl import logging
import gin.tf
import numpy as np
import tensorflow as tf

# replay_buffer
from replay_memory import circular_replay_buffer.WrappedReplayBuffer


# parameters for the gym_carla environment
  params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
  }

# ====================== Global Variants =============================
sess=None
num_actions=8
summary_writer=None
observation_shape=(256,256)
observation_dtype=np.float32
stack_size=4
network=atari_lib.NatureDQNNetwork
gamma=0.99
update_horizon=1
min_replay_history=20000
update_period=4
target_update_period=8000

""" seems not useful

summary_writing_frequency=500
allow_partial_reload=False
"""

replay_capacity=1000000
num_iterations=100 # num_iteartions * max_time_episode <= replay_capacity

# _create_env
def _create_carla_env(params):
  env = gym.make('carla-v0', params=params)
  return env

# _create_agenr
def _create_agent(  sess=None, 
                    num_actions,  # =environment.action_space.n
                    summary_writer=summary_writer
                 ):
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)

def run():
  env = _create_carla_env(params)
  dqn = _create_agent(sess=sess, num_actions=num_actions, summary_writer=summary_writer)
  buffer = WrappedReplayBuffer(self,
               observation_shape,
               stack_size,
               use_staging=False,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=1000,
               extra_storage_types=None,
               observation_dtype=np.uint8,
               terminal_dtype=np.uint8,
               action_shape=(),
               action_dtype=np.int32,
               reward_shape=(),
               reward_dtype=np.float32)
  
  for iteration in range(num_iterations):
    obs = env.reset()
    terminal = False
    replay_buffer = dqn.get_replay()
    for i in range(params['max_time_episode']):
      # agent interact with environment, choose action and then get feedbacks of environment
      action = dqn._select_action() # randomly choice, return int
      next_obs, reward, terminal, info = env.step(action)
      # store transition into buffer
      replay_buffer.add(observatin=obs,   # np.array with shape observation_shape
                        action=action,    # int, the action in the transition
                        reward=reward,    # float, the reward received in the transition
                        terminal=int(terminal),  # np.dtype, acts as a boolean indicating whether the transition was terminal (1) or not (0)
                        *args=info,       # extra contents with shapes and dtypes according to extra_storage_types
                        priority=None,    # unused in the circular replay buffer
                        episode_end=False) 
      obs = next_obs
      
      # terminal condition
      if terminal:
        break
      
  # save replay_buffer into file
  replay_buffer.save(checkpoint_dir='./checkpoint_dir',     # str, the directory where numpy checkpoint files should be saved.
                     iteration_number=11                    # int, iteration_number to use as a suffix in naming numpy checkpoint files
                    )
      
if __name__ == '__main__':
  run()
