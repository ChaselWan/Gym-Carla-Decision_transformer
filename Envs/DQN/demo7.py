import pygame

import abc
import glob
import os
import sys
from types import LambdaType
from collections import deque
from collections import namedtuple

import random 
import time
import numpy as np
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
 
import torch.optim as optim
 
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, ByteTensor
Tensor = FloatTensor



import gym
import gym_carla
import carla



# Easy implement of replay_buffer and DQNNet

class ReplayBuffer(object):
	# buffer just needs to store transition and save them
	def __init__(self, observation_shape,  observation_dtype,  action_shape,  action_dtype, reward_shape, reward_dtype, terminal_dtype, replay_capacity, stack_size):
		# parameters in
		self.observation_shape = observation_shape
		self.observation_dtype = observation_dtype
		self.action_shape = action_shape
		self.action_dtype = action_dtype
		self.reward_shape = reward_shape
		self.reward_dype = reward_dtype
		self.terminal_dtype = terminal_dtype
		self.next_experience_is_episode_start = True
		self.stack_size = stack_size
		self.storage_elements = [
			ReplayElement('observation', self.observation_shape,
				      self.observation_dtype),
			ReplayElement('action', self.action_shape, self.action_dtype),
			ReplayElement('reward', self.reward_shape, self.reward_dtype),
			ReplayElement('terminal', (), self.terminal_dtype)]
		self.replay_capacity = replay_capacity  
		self.memory = self._create_storage
		self.add_count = np.array(0)
		self.episode_and_indices = set()
		self.zero_transition = self._make_zero_transition  # can't be directly functioned outside the class
	
	def _create_storage(self):
		# Create the numpy arrays used to store transitions  
		memory_store = {} # dict
		for storage_element in self.storage_elements:
			array_shape = [self.replay_capacity] + list(storage_element.shape)
			memory_store[storage_element.name] = np.empty(array_shape, dtype=storage_element.type)     # np.empty: 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化
		return memory_store
		
	def _make_zero_transition(self):
		"""Creates a zero transition,used in episode beginnings"""
		_zero_transition = []
		for element_type in self.storage_elements:
			_zero_transition.append(np.zeros(element_type.shape, dtype=element_type.type))
			self.episode_end_indices.discard(self.cursor())
			zero_transition = {e.name: _zero_transition[idx] for idx, e in enumerate(self.storage_elements)}
			
		return zero_transition   # 感觉可以作为一个常量
		
	def make_transition(self, observation, action, reward, terminal):
		# even can be comlemented outside the class
		transition = {"observation": observation, "action": action,  "reward": reward,  "terminal": terminal}
		return transition
		
	def add(self, transition):
		# main function is function the self.zero_transition
		if self.next_experience_is_episode_start:
			for i in range(self.stack_size - 1):
				self._add(self.zero_transition)  # first sight will be the stack_size-1 zero_transition and 1 transition
		if terminal:
			self.episode_end_indices.add(self.add_count % self.replay_capacity)
			self.next_experience_is_episode_start = True
		else:
			self.episode_end_indices.discard(self.add_count % self.replay_capacity)
		self._add(transition)
		
	def _add(self, transition):
		cursor = self.add_count % self.replay_capacity
		for arg_name in transitions:
			self.memory[arg_name][cursor] = transition[arg_name]
		self.add_count += 1
	
	def save(self, checkpoint_dir, iteration_number):
		"""Save Replay Buffer attributes into a file.
  			This method will save all the replay buffer's state in a single file.
    
    			Args:
      			checkpoint_dir: str, the directory where numpy checkpoint files should be saved.
      			iteration_number: int, this to be used as a suffix in naming numpy checkpoint files.
   		 """
		if not tf.io.gfile.exists(checkpoint_dir):
			print(" The checkpoint_dir does not exist !")
			return
		
		checkpointable_elements = self.storage_elements
		for attr in checkpoint_elements:
			filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
			with tf.io.gfile.GFile(filename, 'wb') as f:
				with gzip.GzipFile(fileobj=f, mode='wb') as outfile:
					np.save(outfile, self.memory[attr], allow_pickle=False)


# DQN global parameters
IM_WIDTH = 80
IM_HEIGHT = 60
SHOW_PREVIEW = False
 
SECOND_PER_EPISODE = 10

EPSILON = 0.9       # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 100       # How frequently target netowrk updates
MEMORY_CAPACITY = 100
BATCH_SIZE = 32
LR = 0.01           # learning rate
MODEL_NAME = "nnModule"

def select_action(action_number):
	if action_number == 0:
		real_action = [1, -0.2]
	elif action_number == 1:
		real_action = [1, 0]
	elif action_number == 2:
		real_action = [1, 0.2]
	elif action_number == 3:
		real_action = [2, -0.2]
	elif action_number == 4:
		real_action = [2, 0]
	elif action_number == 5:
		real_action = [2, 0.2]
	elif action_number == 6:
		real_action = [3.0, -0.2]
	elif action_number == 7:
		real_action = [3.0, 0]
	elif action_number == 8:
		real_action = [3.0, 0.2]
	return real_action

class Net(nn.Module):
	def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(26912,5)  #self.head = nn.Linear(896,5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 一层卷积
        x = F.relu(self.bn2(self.conv2(x)))  # 两层卷积
        x = F.relu(self.bn3(self.conv3(x)))  # 三层卷积
        return self.head(x.view(x.size(0),-1)) # 全连接层 
        
class DQN(object):
	def __init__(self):
		self.eval_net,self.target_net = Net(),Net()
		self.learn_step_counter = 0 # count the steps of learning process        
		self.memory = []
		self.position = 0 # counter used for experience replay buff        
		self.capacity = 200
		
		self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
		self.loss_func = nn.MSELoss()
		
	def  choose_action(self, x):
		# This function is used to make decision based upon epsilon greedy
		x = torch.unsqueeze(torch.FloatTensor(x), 0) # add 1 dimension to input state x
		x = x.permute(0,3,2,1)  #把图片维度从[batch, height, width, channel] 转为[batch, channel, height, width]
		if np.random.uniform() < EPSILON:   # greedy
			actions_value = self.eval_net.forward(x)
			action = torch.max(actions_value, 1)[1].data.numpy()
			action = action[0]
		else: 
			action = np.random.randint(0, 5)
		return action
	
	def push_memory(self, obs, a, r, obs_):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
			self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(obs), 0),torch.unsqueeze(torch.FloatTensor(obs_), 0),\
								torch.from_numpy(np.array([a])),torch.from_numpy(np.array([r],dtype='int64')))
			self.position = (self.position + 1) % self.capacity
			
	def get_sample(self,batch_size):
		return random.sample(self.memory, batch_size)
		
	def learn(self):
		if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		self.learn_step_counter += 1
		transitions = self.get_sample(BATCH_SIZE)  # 抽样
		print(transitions)
		batch = Transition(*zip(*transitions))
		b_s = Variable(torch.cat(batch.state))
		b_a = Variable(torch.cat(batch.action))
		b_r = Variable(torch.cat(batch.reward))
		b_s_ = Variable(torch.cat(batch.next_state))
		
		
		b_s = b_s.permute(0,3,2,1)  
		b_s_ = b_s_.permute(0,3,2,1)        
		q_eval = self.eval_net(b_s).gather(1,b_a.unsqueeze(1)) # (batch_size, 1)
		
		
		q_next = self.target_net(b_s_).detach() # detach from computational graph, don't back propagate
		q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # (batch_size, 1)
		loss = self.loss_func(q_eval, q_target)
		self.optimizer.zero_grad() # reset the gradient to zero
		loss.backward()
		self.optimizer.step() # execute back propagation for one step
        
Transition = namedtuple('Transition',('state', 'next_state','action', 'reward'))


def main():
	# parameters for the gym_carla environment
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
		'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
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
	env = gym.make('carla-v0', params=params)
	dqn = DQN()
	buffer = ReplayBuffer(observation_shape=(256,256), 
			      observation_dtype=np.float32, 
			      action_shape=(), 
			      action_dtype=np.uint8,
			      reward_shape=(),
			      reward_dtype=np.float32,
			      terminal_dtype=(),
			      replay_capacity=1000,
			      stack_size=4)
	dqn.eval_net.load_state_dict(torch.load('EvalNet.pt'))
	dqn.eval_net.eval()
	dqn.target_net.load_state_dict(torch.load('TargetNet.pt'))
	dqn.target_net.eval()
	
	count = 0
	max_reward = float("-inf")
	reward_list = []
	for iteration in range(1,11):
		print("# Iteration{} start!".format(iteration))
		reward = 0
		env.reset()
		obs = env.reset()
		for episode in range(100):
			a=dqn.choose_action(obs)
			action = select_action(a)
			print("# Episode{} start!".format(episode))
			print("choose_action:", action)
			obs_,r,done,info = env.step(action)
			print(obs.shape[0])
			dqn.push_memory(obs, a, r, obs_)
			transition = buffer.make_transition(obs, a, r, done)
			buffer.add(transition)
			reward += r
			obs=obs_
			if  (dqn.position % (MEMORY_CAPACITY-1) )== 0: #or done:
				dqn.learn()
				count+=1
				print('learned times:',count)
			if done:
				print("Done!")
				break
		print("# the reward of this iteration is", reward)
		if reward > max_reward:
			max_reward = reward
			torch.save(dqn.eval_net.state_dict(),'EvalNet.pt')
			torch.save(dqn.target_net.state_dict(),'TargetNet.pt')
			buffer.save(Dopamin_DQN,1)

if __name__ == '__main__':
	main()
