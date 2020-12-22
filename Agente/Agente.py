import tensorflow as tf
from typing import Any, List, Sequence, Tuple
import numpy as np
import tqdm
seed = 42

tf.random.set_seed(seed)
np.random.seed(seed)
class Agente():
    def __init__(self,model,
                 env,
                 gamma,
                 max_episodes,
                 max_steps_per_episode,
                 optimizer=tf.keras.optimizers.RMSprop(0.001)):
        self.model = model
        self.env = env
        self.eps = np.finfo(np.float32).eps.item()
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = optimizer
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.running_reward = 0 
        self.reward_threshold = 50000
    
    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""  
        # print('passo 5: ',passo)
        state, reward, done =  self.env.trade(action) # env.step(action)
        # print(state,reward,done,action)
        return (state.astype(np.float32), 
                np.array(reward, np.int32), 
                np.array(done, np.int32))
    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        # print('passo 4: ',passo)
        return tf.numpy_function(self.env_step, [action], 
                                 [tf.float32, tf.int32, tf.int32])
    
    # excuta previsões até o fim de um episodio
    def run_episode(self,initial_state: tf.Tensor,  
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
            state, reward, done = self.tf_env_step(action)
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
    
    """Compute expected returns per timestep."""
    def get_expected_return(self,rewards: tf.Tensor, 
                            gamma: float, 
                            standardize: bool = True) -> tf.Tensor:
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
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                        (tf.math.reduce_std(returns) + self.eps))

        return returns
    """Computes the combined actor-critic loss."""
    def compute_loss(self,action_probs: tf.Tensor,  
                    values: tf.Tensor,  
                    returns: tf.Tensor) -> tf.Tensor:
        advantage = returns - values
          
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
          
        critic_loss = self.huber_loss(values, returns)
    
        return actor_loss + critic_loss
    """Runs a model training step."""
    @tf.function
    def train_step(self,initial_state: tf.Tensor, 
                    model: tf.keras.Model, 
                    optimizer: tf.keras.optimizers.Optimizer, 
                    gamma: float, 
                    max_steps_per_episode: int) -> tf.Tensor:
      

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            # print('passo 2',passo)
            action_probs, values, rewards = self.run_episode(
                initial_state, model, max_steps_per_episode) 

            # Calculate expected returns
            returns = self.get_expected_return(rewards, gamma)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)
        
        return episode_reward
    def initialize(self):
        entrada = np.array([0.6136174899707383, -0.2495323667129071, -0.025853157127760887,
                       -1.0962950381503433, -0.0002503568956511874, 0.527837309399546,
                       0.5770627403157165, -0.6079863229644101, -0.7517983187218091,
                       -1.2609361411198572, -0.06103149876206698, 0.08414481233777048,
                       -0.955093995394538, -0.880256498539612, 0.1572224903597151,
                       -0.1331124051696158, 0.2173943589760951, -0.08262033035611288,
                       -1.127094097106907, -0.5935006806908938, -0.356243152115261,
                       1.529681015979107, 1.366448524306046, 1.5917846995970126],
                        dtype=object)
        with tqdm.trange(self.max_episodes) as t:
            for i in range(self.max_episodes):
              # teste.reset()
              # trader.reset()
              initial_state = tf.constant(entrada, dtype=tf.float32)
              episode_reward = int(self.train_step(initial_state, self.model, self.optimizer, self.gamma, self.max_steps_per_episode))
            
              running_reward = episode_reward*0.01 + self.running_reward*.99
            
              t.set_description(f'Episode {i}')
              t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
            
              # Show average episode reward every 10 episodes
              if i % 10 == 0:
                  pass # print(f'Episode {i}: average reward: {avg_reward}')
            
              if running_reward > self.reward_threshold:  
                  break
            
            print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')