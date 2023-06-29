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
from dopamine.replay_memory.circular_replay_buffer import WrappedReplayBuffer

buffer = WrappedReplayBuffer(observation_shape=(256,256,3),
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

print(buffer)
