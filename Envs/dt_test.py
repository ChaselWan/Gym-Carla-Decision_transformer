#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


# ============= Carla Envs ==============
import gym
import gym_carla
import carla

# softly make a memory_buffer with randomly step
from replay_memory.circular_replay_buffer import WrappedReplayBuffer


def main():
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
    'task_mode': 'roudabout',  # mode of the task, [random, roundabout (only for Town03)]
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

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)
  buffer = WrappedReplayBuffer(self,
               observation_shape=(256,256),  # tuple uses () while list uses []
               stack_size=4,
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
""" Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: int, the discount factor.
      wrapped_memory: The 'inner' memory data structure. If None,
        it creates the standard DQN replay memory.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.
"""        

  for episode in range(1,11): # a trajectory in trajectories
    obs = env.reset()
    terminal = False
    while not terminal:
        action = random.randint(0,8)
        obs, r, terminal, info = env.step(action)
        buffer.add(obs, action, r, terminal)
        if terminal:
          break               

if __name__ == '__main__':
  main()
