# Easy implement of replay_buffer and DQNNet

class ReplayBuffer(observation_shape, 
                   observation_dtype, 
                   action_shape, 
                   action_dtype,
                   reward_shape,
                   reward_dtype,
                   terminal_dtype,
                   replay_capacity,
                   stack_size):
  # buffer just needs to store transition and save them

  def __init__(self):
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
    
    # construct a storage
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
      memory_store[storage_element.name] = np.empty(
        array_shape, dtype=storage_element.type)     # np.empty: 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
      
    return memory_store

  def _make_zero_transition(self):
    """Creates a zero transition,used in episode beginnings"""
    _zero_transition = []
    for element_type in self.storage_elements:
      _zero_transition.append(np.zeros(element_type.shape, dtype=element_type.type))
    self.episode_end_indices.discard(self.cursor())
    # self._add(*zero_transition)
    zero_transition = {e.name: _zero_transition[idx]
                  for idx, e in enumerate(self.storage_elements)}
    return zero_transition   # 感觉可以作为一个常量


  def make_transition(self, observation, action, reward, terminal):
    # even can be comlemented outside the class
    transition = {"observation": observation,
                  "action": action, 
                  "reward": reward, 
                  "terminal": terminal}
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


  def sample(self):
    pass

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

  def load(self)
