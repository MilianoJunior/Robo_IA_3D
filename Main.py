# importação das bibliotecas
import numpy as np
from dados.Dados import Data
from Agente.ActorCritic import ActorCritic
import tensorflow as tf
from typing import Any, List, Sequence, Tuple
from Ambiente.Ambiente import ambiente
import tqdm


# Hiperparametros
num_actions = 3
num_hidden_units = 128
seed = 42
# env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
num_days = 1120
# cudart64_110.dll
# print('rr: ',rr)
max_episodes = 1000
max_steps_per_episode = 550000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 10000000
running_reward = 120

# Discount factor for future rewards
gamma = 0.99

# Importação e tratamento de dados
data = Data(num_days)
entrada_rnn,entrada_trader = data.import_data()

# Criação da rede neural
# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

model = ActorCritic(num_actions, num_hidden_units)
env = ambiente(entrada_rnn,entrada_trader)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""  
  # print('passo 5: ',passo)
  state, reward, done =  env.trade(action) # env.step(action)
  # print(state,reward,done,action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  # print('passo 4: ',passo)
  return tf.numpy_function(env_step, [action], 
                            [tf.float32, tf.int32, tf.int32])


def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state
  
  for t in tf.range(max_steps):
    # print('passo 3: ',passo,' : max_steps: ',t)
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    # print('-----------------------')
    # print('state',state,reward.stack(),done)
    # print('-----------------------')
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()


  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
    # print('   ')
    # print('***********************')
    # print(i,discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
                (tf.math.reduce_std(returns) + eps))

  return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
optimizer = tf.keras.optimizers.RMSprop(0.001)


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    # print('passo 2',passo)
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)
  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward


# self.dados2.values[self.cont]




with tqdm.trange(max_episodes) as t:
    for i in range(max_episodes):
      # teste.reset()
      # trader.reset()
      initial_state = tf.constant(entrada_rnn.values[0], dtype=tf.float32)
      episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))
    
      running_reward = episode_reward*0.01 + running_reward*.99
    
      t.set_description(f'Episode {i}')
      t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
    
      # Show average episode reward every 10 episodes
      if i % 10 == 0:
        pass # print(f'Episode {i}: average reward: {avg_reward}')
    
      if running_reward > reward_threshold:  
          break
    
    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')



