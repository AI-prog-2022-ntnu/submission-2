import math
import random

import numpy as np
import torch
from torch import nn

from enviorments.base_state import BaseState, BoardGameBaseState
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
                 input_size: int,
                 ):
        self.input_size = input_size
        self.nn_config = nn_config

        self.loss_hist = []
        self.train_buffer = []
        self.latest_set_loss_list = []

        self.model = CriticNeuralNet(
            nn_config=self.nn_config,
            input_size=self.input_size,
        )

    def train_network(self,
                      values):
        if len(values) == 0:
            print("No values to train critic")
            return

        loss_fn = torch.nn.L1Loss()

        x_list, y_list = [], []
        for x, y in values:
            x_list.append(x)
            y_list.append([y])

        loss = self.model.train_network(
            inp_x=x_list,
            inp_y=y_list
        )
        # self.loss_hist.append(np.mean(loss))
        self.loss_hist.extend(loss)

        full_pred = self.model.forward(torch.tensor(x_list, dtype=torch.float))
        full_y_tensor = torch.tensor(y_list, dtype=torch.float)
        tot_set_loss = loss_fn(full_pred.to(device="cpu"), full_y_tensor)
        self.latest_set_loss_list.append(tot_set_loss)
        print(f"\nCritic_loss: {tot_set_loss}")

    def get_state_value(self,
                        state: BoardGameBaseState,
                        debug=False):

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
