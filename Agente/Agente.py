import tensorflow as tf
from typing import Any, List, Sequence, Tuple
import numpy as np
class Agente():
    def __init__(self,model,env):
        self.model = model
        self.env = env
    
    def env_step(self,action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""  
        # print('passo 5: ',passo)
        state, reward, done =  self.env.trade(action) # env.step(action)
        # print(state,reward,done,action)
        return (state.astype(np.float32), 
                np.array(reward, np.int32), 
                np.array(done, np.int32))