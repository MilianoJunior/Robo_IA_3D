from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from typing import Any, List, Sequence, Tuple

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
      # funcao_ativacao,
      # camadas,
      # saidas):
    """Initialize."""
    super().__init__()
    model = keras.Sequential([
        layers.Dense(num_hidden_units, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
    ])
    model1 = keras.Sequential([
        keras.layers.Dense(num_actions,activation="linear")
    ])
    model2 = keras.Sequential([
        keras.layers.Dense(1,activation="linear")
    ])
    self.common = model
    self.actor = model1
    self.critic = model2

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)