# importação das bibliotecas
import numpy as np
from dados.Dados import Data
from Agente.ActorCritic import ActorCritic
import tensorflow as tf
from typing import Any, List, Sequence, Tuple
from Ambiente.Ambiente import ambiente
from Agente.Agente import Agente
import tqdm

# Hiperparametros
num_actions = 3
num_hidden_units = 128

num_days = 540

max_episodes = 20
max_steps_per_episode = 550000

reward_threshold = 10000000
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Importação e tratamento de dados
data = Data(num_days)
entrada_rnn,entrada_trader,base = data.import_data()


model = ActorCritic(num_actions, num_hidden_units)
env = ambiente(entrada_rnn,entrada_trader)

agente = Agente(model,
                env,
                gamma,
                max_episodes,
                max_steps_per_episode)
agente.initialize()
# model.save('Save_models/modelo_01')  



