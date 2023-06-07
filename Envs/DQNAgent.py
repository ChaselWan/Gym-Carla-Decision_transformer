"""Chinese babies' own DQNAgent("""

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


# These are aliases which are used by other classes.
NATURE_DQN_OBSERVATION_SHAPE = (256,256)
NATURE_DQN_DTYPE = tf.float32
NATURE_DQN_STACK_SIZE = 4
nature_dqn_network = atari_lib.NatureDQNNetwork


@gin.configurable
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
  """Returns the current epsilon for the agent's epsilon-greedy policy.

  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.

  Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.

  Returns:
    A float, the current epsilon value computed according to the schedule.
  """
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = np.clip(bonus, 0., 1. - epsilon)
  return epsilon + bonus


@gin.configurable
def identity_epsilon(unused_decay_period, unused_step, unused_warmup_steps,
                     epsilon):
  return epsilon


@gin.configurable
class DQNAgent(object):
  """An implementation of the DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=atari_lib.NatureDQNNetwork,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               eval_mode=False,
               use_staging=False,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.compat.v1.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500,
               allow_partial_reload=False):
    """Initializes the agent and constructs the components of its graph."""
    assert isinstance(observation_shape, tuple) 
	# assert语句是一种插入调试断点到程序的一种便捷的方式。
	# 判断一个变量是否是某个类型可以用isinstance()判断
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t use_staging: %s', use_staging)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)

    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.optimizer = optimizer
    tf.compat.v1.disable_v2_behavior()

    if isinstance(summary_writer, str):  # If we're passing in directory name.
      self.summary_writer = tf.compat.v1.summary.FileWriter(summary_writer)
    else:
      self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self.allow_partial_reload = allow_partial_reload

    if sess is None:
      config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
      # Allocate only subset of the GPU memory as needed which allows for
      # running multiple agents/workers on the same GPU.
      config.gpu_options.allow_growth = True
      self._sess = tf.compat.v1.Session('', config=config)
    else:
      self._sess = sess

    with tf.device(tf_device):
      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)   # *********** from this save 
      self._replay.save(checkpoint_dir, iteration_number)  # iteration_number also goes suffix
      """Save the underlying replay buffer's contents in a file.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      iteration_number: int, the iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """

