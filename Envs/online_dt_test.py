#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


#gym-carla envs needs
import gym
import gym_carla
import carla

#online-dt needs
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym
import d4rl
import torch
import numpy as np

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger



Class Experiment:
  def __init__(self, variant, params):
    
    self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
    self.params = params # def params in __init__ so that I can call params in _get_env_spec.
    self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(variant["env_name"])
    
    # initialize by offline trajs
    self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
    
    self.aug_trajs = []
    
    self.device = variant.get("device", "cuda")
    self.target_entropy = -self.act_dim
    self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,  # focus on how action_range works ***
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=True,
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)
    
    self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
    )
    
    self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
    )

    self.log_temperature_optimizer = torch.optim.Adam(
        [self.model.log_temperature],
        lr=1e-4,
        betas=[0.9, 0.999],
    )
    
    # track the training progress and 
    # training/evaluatiom/online performance in all the iterations
    self.pretrain_iter = 0
    self.online_iter = 0
    self.total_transitions_sampled = 0
    self.variant = variant
    self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
    self.logger = Logger(variant)

    
  def _get_env_spec(self, params):
    # env = gym.make('carla-v0', variant["env"]),but here I would like to transfer params(dict data).
    env = gym.make('carla-v0', self.params)
    # self.observation_space = spaces.Dict(observation_space_dict)
    """
      observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
      }
    """
    state_dim = env.observation_space['lidar'].shape[0] # Lidar photos are set as states temporarily
    # Assumed that self.discrete is True
    """
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    """
    act_dim = env.action_space.shape[0]
    action_range =  # Don't know how action_range works temporarily  ********
    # env.close()
    env._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])
    # or "env.reset()"? Don't know which can work well.
    return state_dim, act_dim, action_range
  
  def _load_dataset(self, env_name):
    
    dataset_path = f" ./data/{env_name}.pkl" # dataset should be zipped as pkl file
    # Need to see how dataset set **********
    with open(dataset_path, "rb") as f: # python的打开文件操作
      # rb: 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。
      trajectories = pickle.load(f)  # pkl file 'f' would be reconstructed as python object
      
    states, traj_lens, returns = [], [], []
    for path in trajectories:
      states.append(path["observations"])
      traj_lens.append(len(path["observations"]))
      returns.append(path["rewards"].sum()) # returns here should be rtgs
    
    # used for input normalization
    states = np.concatenate(states, axis=0) # 数组拼接
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6 # 计算沿指定轴的均值和标准差 # but 1e-6 ?
    num_timesteps = sum(traj_lens)
    
    print("="*50)
    print(f"Starting new experiment: {env_name}")
    print(f"{len(traj_lens)}  trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
    print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
    print("=" * 50)
    
    sorted_inds = np.argsort(returns)  # lowest to highest, return index rather than list[index]
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]  # list[-1] returns the last(highest here) data
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds] < num_timesteps:
      timesteps += traj_lens[sorted_inds[ind]]
      num_trajectories += 1
      ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]
    trajectories = [trajectories[ii]] for ii in sorted_inds]
  
    

def main():
  # ======================== Parameters for the gym_carla environment ========================
  params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
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

  
  #======================== Set variants parser ========================
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=10)
  parser.add_argument("--env_name", type=str, default="carla-v0")

  # model options
  parser.add_argument("--K", type=int, default=20)
  parser.add_argument("--embed_dim", type=int, default=512)
  parser.add_argument("--n_layer", type=int, default=4)
  parser.add_argument("--n_head", type=int, default=4)
  parser.add_argument("--activation_function", type=str, default="relu")
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--eval_context_length", type=int, default=5)
  # 0: no pos embedding others: absolute ordering
  parser.add_argument("--ordering", type=int, default=0)

  # shared evaluation options
  parser.add_argument("--eval_rtg", type=int, default=3600)
  parser.add_argument("--num_eval_episodes", type=int, default=10)

  # shared training options
  parser.add_argument("--init_temperature", type=float, default=0.1)
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
  parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
  parser.add_argument("--warmup_steps", type=int, default=10000)

  # pretraining options
  parser.add_argument("--max_pretrain_iters", type=int, default=1)
  parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

  # finetuning options
  parser.add_argument("--max_online_iters", type=int, default=1500)
  parser.add_argument("--online_rtg", type=int, default=7200)
  parser.add_argument("--num_online_rollouts", type=int, default=1)
  parser.add_argument("--replay_size", type=int, default=1000)
  parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
  parser.add_argument("--eval_interval", type=int, default=10)

  # environment options
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
  parser.add_argument("--save_dir", type=str, default="./exp")
  parser.add_argument("--exp_name", type=str, default="default")

  args = parser.parse_args()
  

  # ======================== Set gym-carla environment ========================
  experiment = Experiment(vars(args), params)  # transfer parameters

  
  
  obs = env.reset()

  while True:
    action = [2.0, 0.0]
    obs,r,done,info = env.step(action)

    if done:
      obs = env.reset()


if __name__ == '__main__':
  main()
