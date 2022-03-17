import random

import torch

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
from rl_agent.critic_net import CriticNeuralNet
from rl_agent.util import NeuralNetworkConfig


def generate_batch(data_list,
                   batch_size):
    batch = []
    while len(batch) < batch_size:
        batch.append(random.choice(data_list))
    return batch


class Critic:

    def __init__(self,
                 nn_config: NeuralNetworkConfig,
                 environment: BoardGameEnvironment,
                 input_size: int,
                 ):
        self.environment = environment
        self.input_size = input_size
        self.nn_config = nn_config

        self.loss_hist = []
        self.train_buffer = []
        self.latest_set_loss_list = []

        self.model = CriticNeuralNet(
            nn_config=self.nn_config,
            environment=self.environment,
            input_size=self.input_size,
        )

    def train_from_buffer(self,
                          buffer):
        loss = self.model.train_from_buffer(buffer)

        # self.loss_hist.append(np.mean(loss))
        self.loss_hist.extend(loss)

    def get_state_value(self,
                        state: BoardGameBaseState):
        # if state.current_player_turn() == 0:
        state_vec = state.get_as_vec()
        # else:
        #     state_vec = state.get_as_inverted_vec()

        state_val = self.model.forward(torch.tensor([state_vec], dtype=torch.float))[0].tolist()
        return state_val

    def get_states_value(self,
                         state_list):
        x = [s.get_as_vec() for s in state_list]
        # x = [s.get_as_vec() if s.current_player_turn() == 0 else s.get_as_inverted_vec() for s in state_list]
        state_val = self.model.forward(torch.tensor(x, dtype=torch.float))[0].tolist()
        return state_val
