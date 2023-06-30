import pygame

import abc
import glob
import os
import gzip
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
import collections
from dopamine.replay_memory.circular_replay_buffer import OutOfGraphReplayBuffer

import gym
import gym_carla
import carla
import tensorflow as tf


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
# *******************parameters for the gym_carla environment*******************
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
		'if_reload_world': False, # iteration = 1 not to reload
	}

# *******************hyparameters for the DQN model-and-dataset  save-and-load*******************
	model_file_dir = time.strftime("%Y-%m-%d",time.gmtime())
	if not tf.io.gfile.exists('./dopamine_model/{}'.format(model_file_dir)):
		os.mkdir('./dopamine_model/{}'.format(model_file_dir))

	count = 11
	gained_reward = 351.99231270847946

# *******************step, train and save*******************
	env = gym.make('carla-v0', params=params)
	dqn = DQN()
	buffer = OutOfGraphReplayBuffer(observation_shape=(256,256,3),
                              stack_size=3,
                              replay_capacity=1000,
                              batch_size=32,
                              update_horizon=1,
                              gamma=0.99,
                              max_sample_attempts=100,
                              extra_storage_types=None,
                              observation_dtype=np.float32,
                              terminal_dtype=np.uint8,
                              action_shape=(),
                              action_dtype=np.int32,
                              reward_shape=(),
                              reward_dtype=np.float32,
                              checkpoint_duration=4,
                              keep_every=None)

	dqn.eval_net.load_state_dict(torch.load('./dopamine_model/'+model_file_dir+'/EvalNet-learned_times={}-reward={}.pt'.format(count, gained_reward)))
	dqn.eval_net.eval()
	dqn.target_net.load_state_dict(torch.load('./dopamine_model/'+model_file_dir+'/TargetNet-learned_times={}-reward={}.pt'.format(count, gained_reward)))
	dqn.target_net.eval()
	

	max_reward = gained_reward
	buffer_saved_times = 0
	for iteration in range(1,100):  # try to reload world per iteration
		if buffer_saved_times == 5:
			break
		print("# Iteration{} start!".format(iteration))
		reward = 0
		obs = env.reset()
		for episode in range(200):
			a=dqn.choose_action(obs)
			action = select_action(a)
			print("# Episode{} start!".format(episode))
			obs_,r,terminal,info = env.step(action)
			dqn.push_memory(obs, a, r, obs_)
			buffer.add(obs, a, r, int(terminal))  # directly add
			reward += r
			obs=obs_
			if  (dqn.position % (MEMORY_CAPACITY-1) )== 0 and reward > max_reward: #or done:
				dqn.learn()
				torch.save(dqn.eval_net.state_dict(),'./dopamine_model/'+model_file_dir+'/EvalNet-learned_times={}-reward={}.pt'.format(count, max_reward))
				torch.save(dqn.target_net.state_dict(),'./dopamine_model/'+model_file_dir+'/TargetNet-learned_times={}-reward={}.pt'.format(count, max_reward))
				max_reward = reward
				count+=1
				print('learned times:',count)
			if terminal:
				print("Done!")
				break
			if buffer.cursor() % 200 == 0:
				print("###################buffer has added 200 transition##################")
		print("# the reward of this iteration is", reward)
		if buffer.is_full():
			buffer.save("/home/chaselwan/公共的/gym-carla-master/dopamine_dataset",buffer_saved_times)
			print('*******************buffer has been saved successfully!*******************')
			buffer_saved_times += 1



if __name__ == '__main__':
	main()
# remember add a function which shows whether buffer is full or not
# Save when buffer is full

