import random

import numpy as np
import torch
from torch import nn

from enviorments.base_state import BaseState, GameBaseState


class CriticNeuralNet(nn.Module):
    def __init__(self,
                 inp_s):
        super(CriticNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inp_s, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 1),
            nn.Tanh(),
        )

    def forward(self,
                x):
        return self.network(x)

    def save_model(self,
                   fp):
        torch.save(self.state_dict(), fp)

    @staticmethod
    def load_model(fp,
                   *args):
        model = CriticNeuralNet(*args)
        model.load_state_dict(torch.load(fp))
        model.eval()
        return model


def generate_batch(data_list,
                   batch_size):
    batch = []
    while len(batch) < batch_size:
        batch.append(random.choice(data_list))
    return batch


class Critic:

    def __init__(self,
                 # critic_lr: float,
                 input_space):
        self.input_space = input_space

        self.loss_hist = []

        self._gen_model()

    def _gen_model(self):
        self.model = CriticNeuralNet(self.input_space)

    def train_network(self,
                      values):

        self.model.train(True)
        torch.set_printoptions(profile="full", linewidth=1000)
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters())
        train_itrs = 50
        batch_size = 5

        for _ in range(train_itrs):
            x_list, y_list = [], []
            for x, y in generate_batch(values, batch_size):
                x_list.append(x)
                y_list.append(y)
            x = torch.tensor(x_list, dtype=torch.float)
            y = torch.tensor(y_list, dtype=torch.float)
            pred = self.model.forward(x)
            # print("y:    ", y)
            # print("pred: ", pred)
            # print(pred)

            loss: torch.Tensor = loss_fn(pred, y)
            self.loss_hist.append(loss.item())

            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.model.train(False)

    def get_state_value(self,
                        state: GameBaseState,
                        debug=False):
        state_vec = state.get_as_vec()
        state_val = self.model.forward(torch.tensor([state_vec], dtype=torch.float))[0].tolist()
        return state_val

    def get_states_value(self,
                         state_list):
        x = [s.get_as_vec() for s in state_list]
        state_val = self.model.forward(torch.tensor(x, dtype=torch.float))[0].tolist()
        return state_val
