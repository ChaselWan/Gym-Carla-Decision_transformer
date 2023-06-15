from DQNComponent import ReplayBuffer,NatureDQNNetwork



class DQNAgent(object):
  # An agent for inputing observations, outputing action and storing memory to buffer
  
  def __init__(self, 
               num_actions, 
               observation_shape, 
               observation_dtype, 
               stack_size,
               network, 
               gamma, 
               update_horizon, 
               cumulative_gama = math.pow(gamma, update_horizon), 
               min_replay_history,  # 
               target_update_period, # 
               epsilon_fn, 
               epsilon_train, 
               epsilon_eval, 
               epsilon_decay_period, 
               update_period, 
               #eval_mode, 
               training_steps, 
               optimizer):
    # initialize DQNagent and replay_buffer
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
    # self.eval_mode = eval_mode 是否是evaluation模式
    self.training_steps = 0
    self.optimizer = optimizer
    # tf.compat.v1.disable_v2_behavior():ban tensorflow-version2.x function
    
    
    with tf.device(tf_device):  # tf_device = cpu
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(
          self.observation_dtype, state_shape, name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)
      
      self._build_networks()
      
      self._train_op = self._build_train_op()
      
    
  def _create_network(self, name):
    network = self.network(self.num_actions, name=name)
    return network
  def _build_replay_buffer(self):
    return ReplayBuffer(observation_shape=self.observation_shape, 
                        observation_dtype=self.observation_dtype,
                        action_shape=self.action_shape, 
                        action_dtype=self.action_dtype, 
                        reward_shape=self.reward_shape, 
                        reward_dtype=self.reward_dtype, 
                        terminal_dtype=self.terminal_dtype,              
                        stack_size=self.stack_size, 
                        replay_capacity=100000, 
                        batch_size=self.batch_size, 
                        update_horizon=1)
  
  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.
    
    self.online_convnet: For computing the current state's Q-values.
    self.target_convnet: For computing the next state's target Q-values.
    self._net_outputs: The actual Q-values.
    self._q_argmax: The action maximizing the current state's Q-values.
    self._replay_net_outputs: The replayed states' Q-values.
    self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)
  
  def _build_train_op(self):
    """Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
      self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
      self._replay_net_outputs.q_values * replay_action_one_hot, 
      axis=1, 
      name='replay_chosen_q')
    
    target = tf.stop_gradient(self._build_target_q_op())
  
  def _build_target_q_op(self):
    """ Returns:
      target_q_op: An op calculating the Q-value.
    """
    replay_next_qt_max = tf.reduce_max(
      self._replay_next_target_net_outputs.q_values, 1)
    
    return self._replay.rewards + self.cumulative_gamma * replay_next_qt_max * (
      1. - tf.cast(self._replay.terminals, tf.float32))
  
  def step(self, reward, observation):
    
  
  def choose_action(self, observation):
    """Returns: int, the selected action
    """
    if random.randint() <= self.epsilon_fn:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      return self._sess.run(
