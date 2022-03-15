import math
import random
import time

import numpy as np
import torch
from torch import nn

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState
from rl_agent.util import NeuralNetworkConfig


class CriticNeuralNet(nn.Module):
    def __init__(self,
                 nn_config: NeuralNetworkConfig,
                 environment: BoardGameEnvironment,
                 input_size: int,
                 invert_p2=True):
        super(CriticNeuralNet, self).__init__()

        self.environment = environment
        self.invert_p2 = invert_p2
        self.input_size = input_size
        self.b_size = math.floor(math.sqrt(input_size))  # TODO: FIX IMPORTANT!
        self.nn_config = nn_config

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=100,
                kernel_size=(5, 5),
                padding=2,
                # stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=100,
                out_channels=40,
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
                in_channels=40,
                out_channels=20,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=20,
                out_channels=20,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.Flatten(),
            nn.ELU(),
            nn.Linear((input_size) * 20, 1),
        )

        self.loss_fn = torch.nn.L1Loss()

        # self.opt = torch.optim.Adam(self.parameters(), lr=0.00005)
        # self.opt = torch.optim.Adam(self.network.parameters())

        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.0005)
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

    def _train_network(self,
                       inp_x,
                       inp_y):
        self.train(True)

        x_inp = torch.tensor(inp_x, dtype=torch.float)
        y_inp = torch.tensor(inp_y, dtype=torch.float)

        loss_values = []

        wait_milli_sec = self.nn_config.episode_train_time_ms
        c_time = time.monotonic_ns()
        stop_t = c_time + (wait_milli_sec * 1000000)
        rnds = 0
        stop = False

        bs = 5
        set_len = len(x_inp)
        rest = set_len % bs
        currs = 0

        while not stop:
            # while currs < set_len:
            #     if (currs + bs) > set_len:
            #         x = x_inp[:-rest]
            #         y = y_inp[:-rest]
            #     else:
            #         x = x_inp[currs:currs + bs]
            #         y = y_inp[currs:currs + bs]
            #     currs += bs

            rnds += 1
            rand_idx = torch.randint(set_len, (self.nn_config.batch_size,))
            x = x_inp[rand_idx]
            y = y_inp[rand_idx]

            self.opt.zero_grad()

            pred = self.forward(x)

            loss: torch.Tensor = self.loss_fn(pred, y)
            loss_values.append(loss.tolist())

            loss.backward()
            self.opt.step()
            if self.nn_config.use_iter:
                if self.nn_config.data_passes is not None:
                    stop = (rnds / self.nn_config.batch_size) > self.nn_config.data_passes
                else:
                    stop = rnds > self.nn_config.train_iterations
            else:
                stop = time.monotonic_ns() > stop_t

        print(f"CRITIC: completed {rnds} training epochs with batch size {self.nn_config.batch_size} in the {wait_milli_sec}ms limit")
        self.train(False)
        return loss_values

    def _mabye_invert_input(self,
                            inp):
        x = []
        inverted_idx_list = []
        # invert if used
        for n, state in enumerate(inp):
            state: BoardGameBaseState = state
            state_vec = state.get_as_vec()
            if self.invert_p2:
                if state.current_player_turn() == -1:
                    inverted_state_vec = self.environment.invert_observation_space_vec(state_vec)
                    inverted_idx_list.append(n)
                    x.append(inverted_state_vec)
                else:
                    x.append(state_vec)
            else:
                x.append([*state_vec, state.current_player_turn()])

        return x, inverted_idx_list

    def train_from_buffer(self,
                          state_buffer):
        if len(state_buffer) == 0:
            print("No values to train critic")
            return

        x_list, y = [], []
        for state, value in state_buffer:
            x_list.append(state)
            y.append([value])

        x, inverted_idx_list, = self._mabye_invert_input(x_list)

        for inverted_idx in inverted_idx_list:
            y[inverted_idx][0] = y[inverted_idx][0] * -1

        loss = self._train_network(
            inp_x=x,
            inp_y=y
        )

        return loss

    def get_state_value(self,
                        state_list: [BoardGameBaseState]):

        x, inverted_idx_list, = self._mabye_invert_input(state_list)

        out = self.forward(x)

        for inverted_idx in inverted_idx_list:
            out[inverted_idx] = out[inverted_idx] * -1

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
