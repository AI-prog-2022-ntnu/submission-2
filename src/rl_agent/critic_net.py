import math
import random
import time

import numpy as np
import torch
from torch import nn

from enviorments.base_state import BaseState, BoardGameBaseState
from rl_agent.util import NeuralNetworkConfig


class CriticNeuralNet(nn.Module):
    def __init__(self,
                 nn_config: NeuralNetworkConfig,
                 input_size: int):
        super(CriticNeuralNet, self).__init__()

        self.input_size = input_size
        self.b_size = math.floor(math.sqrt(input_size))  # TODO: FIX IMPORTANT!
        self.nn_config = nn_config

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=50,
                kernel_size=(5, 5),
                padding=2,
                # stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=50,
                out_channels=30,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                padding=1,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=30,
                out_channels=20,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=20,
                out_channels=30,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear((input_size) * 30, 1),
            nn.Tanh()
        )

        self.loss_fn = torch.nn.L1Loss()

        # self.opt = torch.optim.Adam(self.parameters(), lr=0.00005)
        self.opt = torch.optim.Adam(self.network.parameters())

        # self.opt = torch.optim.RMSprop(self.parameters(), lr=0.001)
        # self.opt = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self,
                x):
        p1_inp = torch.where(x == 1, 1.0, 0.0)
        p2_inp = torch.where(x == -1, 1.0, 0.0)

        p_stack = torch.stack([p1_inp, p2_inp], dim=1)

        x = p_stack.view((-1, 2, self.b_size, self.b_size))
        out = self.network(x)

        # x = p_stack.view((-1, self.inp_s * 2))

        # x = x.view((-1, 1, self.b_size, self.b_size))
        return out

    def train_network(self,
                      inp_x,
                      inp_y):
        self.train(True)

        x = torch.tensor(inp_x, dtype=torch.float)
        y = torch.tensor(inp_y, dtype=torch.float)

        loss_values = []

        wait_milli_sec = self.nn_config.episode_train_time_ms
        c_time = time.monotonic_ns()
        stop_t = c_time + (wait_milli_sec * 1000000)
        rnds = 0
        # for i in range(num_rollouts):

        while time.monotonic_ns() < stop_t:
            rnds += 1
            rand_idx = torch.randint(len(x), (self.nn_config.batch_size,))
            x = x[rand_idx]
            y = y[rand_idx]

            self.opt.zero_grad()

            pred = self.forward(x)

            loss: torch.Tensor = self.loss_fn(pred, y)
            loss_values.append(loss.tolist())

            loss.backward()
            self.opt.step()

        print(f"CRITIC: completed {rnds} training epochs with batch size {self.nn_config.batch_size} in the {wait_milli_sec}ms limit")
        self.train(False)
        return loss_values

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
