

ReplayElement = (
  collections.namedtuple('shape_type', ['name', 'shape', 'type']))

STORE_FILENAME_PREFIX = '$store$_'

class Replay_Buffer(object):
  """ A simple out-of-graph Replay Buffer.
  Paraphrased from Dopamine.OutOfGraphReplayBuffer.
  
  Updates:
    skipped logging.info
    skipped process of learn
  
  Function:
    add:add transition into self._store
    save:save self._store into a file
    load:load file of self._store
  """
  def __init__(self,
              observation_shape=(256,256)
              stack_size=4,
              replay_capacity=1000,
              batch_size=32,
              update_horizon=1,
              gamma=0.99,
              max_sample_attempts=1000,
              extra_storage_types=None,
              observation_dtype=np.uint8,  # whether revise
              terminal_dtype=np.uint8,
              action_shape=9, 
              action_dtype=np.int32,
              reward_shape=(),   # Empty tuple means the action is a scalar.
              reward_dtype=np.float32,
              checkpoint_duration=4,
              keep_every=None):
    assert isinstance(observation_shape, tuple)  # for debug
    if replay_capacity < update_horizon + stack_size:
      raise ValueError('There is not enough capacity to cover '
                       'update_horizon and stack_size.')
      
    # skip logging.info
    # import parameters as follows:
    self._action_shape = action_shape
    self._action_dtype = action_dtype
    self._reward_shape = reward_shape
    self._reward_dtyoe = reward_dtype
    self._observation_shape = observation_shape
    self._stack_size = stack_size
    self._stack_shape = self._observation_shape + (self._stack_size,)  # 
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._update_horizon = update_horizon
    self._gamma = gamma
    self._observation_dtype = observation_dtype
    self._terminal_dtype = terminal_dtype
    self._max_sample_attempts = max_sample_attempts
    
    self._create_storage()  # create self._store to store memory
    self.add_count = np.array(0)  # int,counter of how many transitions have been added (including the blank ones at the beginning of an episode).
    self.episode_end_indices = set()
    
    
  def _create_storage(self):
    """creates the numpy arrays used to store transitions."""
    self._store = {}  # for a trajectory?
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_capacity] + list(storage_element.shape)
      self._store[storeage_element.name] = np.empty(array_shape, dtyoe=storage_element.type)    

  def get_storage_signature(self):
    """The signature of the add function."""
    # Returns: list of ReplayElements defining the type of the contents stored.
    storage_elements = [
        ReplayElement('observation', self._observation_shape,
                      self._observation_dtype),
        ReplayElement('action', self._action_shape, self._action_dtype),
        ReplayElement('reward', self._reward_shape, self._reward_dtype),
        ReplayElement('terminal', (), self._terminal_dtype)
    ]
    """
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                        element.type))  # seems to be skipped
    """
    return storage_elements
  
  def add(self, observation, action, reward, terminal,
         episode_end=False):
    """make transitions for step, and make zero_transition when terminal
    """
    # self._check_add_types(observation, action, reward, terminal)
    
    # if next_experience is start of a episode, add a zero_transition
    # but this function can be complished in step episode, therefore skipped
    
    transition = {
      'observation': observation,
      'action':action,
      'reward':reward,
      'terminal':terminal}
    
    self._add_transition(self, transition)
    if terminal:
      self.episode_end_indices.add(self.cursor())
      zero_transition = {}
      for element_type in self.get_storage_element():
        zero_transition[element_type] = np.zeros(element_type.shape, dtype=element_type.type)  # add element_type.key and element_type.value to zero_transition
      self._add_transition(self, zero_transition)
      
  def _add_transition(self, transition):
    """
    Args:
      transition: The dictionary of names and values of the transition
                  to add to the storage.
    """
    cursor = self.cursor()
    for argname in transition:
      self._store[arg_name][cursor] = transition[arg_name]
    
    self.add_count += 1
    
    """ can be skipped
    self.invalid_range = invalid_range(
        self.cursor(), self._replay_capacity, self._stack_size,
        self._update_horizon)    
    """
  def cursor(self):
    """Index to the location where the next transition will be written."""
    return self.add_count % self._replay_capacity
  
  def save(self, checkpoint_dir, iteration_number):
    """
    This method will save all the replay buffer's state in a single file.
    
    Args:
      checkpoint_dir: str, the directory where numpy checkpoint files should be
      iteration_number: int, iteration_number to use as a suffix in naming
        numpy checkpoint files
    """
    if not tf.io.gfile.exists(checkpoint_dir):  # Determines whether a path exists or not.
      print("folder or path does not exist")
      return
    
    checkpointable_elements = self._return_checkpointable_elements()
    
    for attr in checkpointable_elements:
      filename = self._generate_filename(checkpoint_dir, attr, iteration_number)
      with tf.io.gfile.GFile(filename, 'wb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb') as outfile:
          if attar.startswith(STORE_FILENAME_PREFIX):
            array_name = attr[len(STORE_FILENAME_PREFIX):]
            np.save(outfile, self._store[array_name], allow_pickle=False)
            
            # Some numpy arrays might not be part of storage
          elif isinstance(self.__dict__[attr], np.ndarray):
            np.save(outfile, self.__dict__[attr], allow_pickle=False)
          else:
            pickle.dump(self.__dict__[attr], outfile)
    
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
