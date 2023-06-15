# Chinese babies' own DQNAgent
# The most important thing is replay buffer

import tensorflow as tf
import numpy as np
import collections

STORE_FILENAME_PREFIX = '$store$_'

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

class NatureDQNNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's Q-values."""

  def __init__(self, num_actions, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(NatureDQNNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                        name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense1(x)

    return DQNNetworkType(self.dense2(x))


class ReplayBuffer(object):
  """
  A buffer specially for storing memory which doesn't function for anything else.
  """
  def __init__(self, 
               observation_shape, 
               observation_dtype,
               action_shape, 
               action_dtype, 
               reward_shape, 
               reward_dtype, 
               terminal_dtype,              
               stack_size, 
               replay_capacity, 
               batch_size, 
               update_horizon, 
               # gamma: for sample, 
               # checkpoint_duration: for iteration
               ):
    """Initializes Replay Buffer"""
    self.observation_shape = observation_shape
    self.observation_dtype = observation_dtype
    self.action_shape = action_shape
    self.action_dtype = action_dtype
    self.reward_shape = reward_shape
    self.reward_dtype = reward_dtype
    self.terminal_dtype = terminal_dtype
    self.stack_size = stack_size
    self.replay_capacity = replay_capacity
    self.batch_size = batch_size  # for sample but unuseful temprorarily
    self.storage_elements = [
        ReplayElement('observation', self.observation_shape,
                      self.observation_dtype),
        ReplayElement('action', self.action_shape, self.action_dtype),
        ReplayElement('reward', self.reward_shape, self.reward_dtype),
        ReplayElement('terminal', (), self.terminal_dtype)
    ]
    """print(self.storage_elements:
    [shape_type(name='observation', shape=(256, 256), type=<class 'numpy.float32'>),
    shape_type(name='action', shape=(), type=<class 'numpy.uint8'>), shape_type(name='reward', shape=(), type=<class 'numpy.uint8'>), 
    shape_type(name='terminal', shape=(), type=<class 'numpy.uint8'>)]
    """
    
    # create buffer store
    self.memory = self._create_storage()
    self.add_count = np.array(0)
    # self.invalid_range 
    # self._cumulative_discount_vector: for sample and caculate ?
    self._next_experience_is_episode_start = True # initial before start
    self.episode_end_indices = set()  # store indices of episode_end
    self.zero_transition = self._create_zero_transition()
    
    
  def _create_storage(self):
    # Create the numpy arrays used to store transitions    
    memory_store = {} # dict
    for storage_element in self.storage_elements:
      array_shape = [self.replay_capacity] + list(storage_element.shape)
      memory_store[storage_element.name] = np.empty(
        array_shape, dtype=storage_element.type)     # np.empty: 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
    
    return memory_store
  
  def get_episode_and_indices(self):
    return self.episode_end_indices
                   
               
  def add(self, observation, action, reward, terminal):
    # Adds a transition to the replay memory
    """
     Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
                was terminal (1) or not (0).
    """
    if self._next_experience_is_episode_start:   # 只在开头用，在开头之前加
      for _ in range(self.stack_size - 1):  # zero_transition变成常量之后，直接调用add
        self._add(self.zero_transition)
      self._next_experience_is_episode_start = False
      
    if terminal:
      self.episode_end_indices.add(self.cursor())
      self._next_experience_is_episode_start = True
    else:
      self.episode_end_indices.discard(self.cursor())
    
    transition = {"observation": observation,
                  "action": action, 
                  "reward": reward, 
                  "terminal": terminal}
    self._add(transition)
  
  def _create_zero_transition(self):
    """Creates a zero transition,used in episode beginnings"""
    _zero_transition = []
    for element_type in self.storage_elements:
      _zero_transition.append(np.zeros(element_type.shape, dtype=element_type.type))
    self.episode_end_indices.discard(self.cursor())
    # self._add(*zero_transition)
    zero_transition = {e.name: _zero_transition[idx]
                  for idx, e in enumerate(self.storage_elements)}
    return zero_transition   # 感觉可以作为一个常量

  def cursor(self):
    """Index to the location where the next transition will be written."""
    return self.add_count % self.replay_capacity
  
  def _add(self,transition):
    # Add Function Finally
    cursor = self.cursor()
    for arg_name in transitions:
      self.memory[arg_name][cursor] = transition[arg_name]
    
    self.add_count += 1
    # self.invalid_range = invalid_range(self.cursor(), self._replay_capacity, self._stack_size, self._update_horizon)
  
  def is_empty(self):
    """Is the Replay Buffer empty?"""
    return self.add_count == 0
  
  def is_full(self):
    """Is the Replay Buffer full?"""
    return self.add_count >= self.replay_capacity
  
# stack state function are temprorily not written
  
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
    checkpointable_elements = self._return_checkpointable_elements()
    
    for attr in checkpoint_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      # pirnt(filename): E:\Users\Admin\Desktop\日程表\202306\空间天气预报作业\name_ckpt.33.gz
      with tf.io.gfile.GFile(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as outfile:
          if attr.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]  # 就是把STORE_FILENAME_PREFIX去掉了
            np.save(outfile, self.memory[array_name], allow_pickle=False)
      
      # Do not need to garbage other checkpoint_files
      
  def _return_checkpointable_elements(self):
    checkpoint_elements = {}
    for member_name, member in self.__dict__.items():
      if member_name == 'memory':
        for array_name, array in self.memory.items():
          checkpointable_elements[STORE_FILENAME_PREFIX + array_name] = array
          # 把memory以内和以外的member分开.
          # STORE_FILENAME_PREFIX indicates that the variable is contained in
    return checkpointable_elements
  
  def _generate_filename(self, checkpoint_dir, name, suffix):
    return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))
  
  def load(self, checkpoint_dir, suffix):
    """Restores the object from bundle_dictionary and numpy checkpoints"""
    save_elements = self._return_checkpointable_elements()
    skip_episode_end_indices = False
    # We will first make sure we have all the necessary files available to avoid
    # loading a partially-specified (i.e. corrupted) replay buffer.
    
    for attr in save_elements:
      filename = self._generate_filename(checkpoint_dir, attr, suffix)
      
      if not tf.io.gfile.exists(filename):
        if attr == 'episode_end_indices':
          logging.warning('Unable to find episode_end_indives.This is '
                          'expected for old checkpoints.')
          skip_episode_end_indices = True
          continue
        
        raise tf.errors.NotFoundError(None, None, 
                                      'Missing file:{}'.format(filename))
        
    # If we've reached this point then we have verified that all expected files
    # are available.
    for attr in save_elements:
      if attr == 'episode_end_indices' and skip_episode_end_indices:
        continue

    filename = self._generate_filename(checkpoint_dir, attr, suffix)
    with tf.io.gfile.GFile(filename, 'rb') as f:
      with gzip.GzipFile(fileobj=f) as infile:
        if attr.startswith(STORE_FILENAME_PREFIX):
          array_name = attr[len(STORE_FILENAME_PREFIX):]
          self._store[array_name] = np.load(infile, allow_pickle=False)
        elif isinstance(self.__dict__[attr], np.ndarray):
          self.__dict__[attr] = np.load(infile, allow_pickle=False)
        else:
          self.__dict__[attr] = pickle.load(infile)
  
