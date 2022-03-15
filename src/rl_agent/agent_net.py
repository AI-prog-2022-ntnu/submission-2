import math
import random
import time

import numpy as np
import torch
from torch import nn

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
from rl_agent.util import NeuralNetworkConfig, get_action_visit_map_as_target_vec


class BoardGameActorNeuralNetwork(nn.Module):
    def __init__(self,
                 nn_config: NeuralNetworkConfig,
                 environment: BoardGameEnvironment,
                 input_size,
                 output_size,
                 invert_p2=True):
        super(BoardGameActorNeuralNetwork, self).__init__()

        self.invert_p2 = invert_p2
        self.environment = environment
        self.output_size = output_size
        self.nn_config = nn_config

        self.board_size = math.floor(math.sqrt(input_size))  # TODO: this is bad

        if self.invert_p2:
            self.input_size = input_size
        else:
            self.input_size = input_size + 1

        # self.network = nn.Sequential(
        #     nn.Linear(self.input_size * 2, 200),
        #     nn.ELU(),
        #     nn.Linear(200, self.output_size),
        #     # nn.Sigmoid(),
        #     # nn.ReLU()  # <- DONT CHANGE
        #     # nn.Tanh()
        # )
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
            nn.Linear((input_size) * 30, self.output_size),
        )

        self.soft_max = torch.nn.Softmax(dim=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.opt = torch.optim.Adam(self.parameters())
        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.001)
        # self.opt = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self,
                inp_x,
                ):
        x = inp_x

        if self.invert_p2:
            keep_filter = x == 0
        else:
            keep_filter = x[:, :-1] == 0
        # keep_filter_inv = torch.logical_not(keep_filter)

        p1_inp = torch.where(x == 1, 1.0, 0.0)
        p2_inp = torch.where(x == -1, 1.0, 0.0)

        p_stack = torch.stack([p1_inp, p2_inp], dim=1)

        # x_ys = math.floor(math.sqrt(self.input_size))
        x = p_stack.view((-1, 2, self.board_size, self.board_size))

        # x = p_stack.view((-1, self.input_size * 2))

        out: torch.Tensor = self.network(x)

        out_soft_masked = torch.zeros_like(out)
        for n in range(len(x)):
            valid_moves = out[n, keep_filter[n, :]]
            soft_m_moves = self.soft_max(valid_moves)
            out_soft_masked[n, keep_filter[n, :]] = soft_m_moves

        return out_soft_masked

    def _train_network(self,
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

        print(f"AGENT: completed {rnds} training epochs with batch size {self.nn_config.batch_size} in the {wait_milli_sec}ms limit")
        self.train(False)
        return loss_values

    def get_action_visit_map_as_target_vec(self,
                                           action_visit_map: {}):
        possible_actions = self.environment.get_action_space()
        visit_sum = sum(action_visit_map.values())
        ret = []
        for action in possible_actions:
            value = action_visit_map.get(action)
            if value is None or visit_sum == 0:
                ret.append(0)
            else:
                ret.append(value / visit_sum)
        return ret

    def train_from_state_buffer(self,
                                state_buffer):
        torch.set_printoptions(profile="full", linewidth=1000)

        raw_x, y = [], []

        for state, v_count_map in state_buffer:
            raw_x.append(state)
            y.append(self.get_action_visit_map_as_target_vec(v_count_map))

        x, inverted_idx_list = self._mabye_invert_input(raw_x)

        for inverted_idx in inverted_idx_list:
            y[inverted_idx] = self.environment.invert_action_space_vec(y[inverted_idx])

        loss = self._train_network(x, y)
        return loss

    def _mabye_invert_input(self,
                            inp):
        x = []
        inverted_idx_list = []
        # invert if used
        for n, state in enumerate(inp):
            state: BoardGameBaseState = state
            state_vec = state.get_as_vec()
            if self.invert_p2:
                if state.current_player_turn() != 0:
                    inverted_state_vec = self.environment.invert_observation_space_vec(state_vec)
                    inverted_idx_list.append(n)
                    x.append(inverted_state_vec)
                else:
                    x.append(state_vec)
            else:
                x.append([*state_vec, state.current_player_turn()])

        return x, inverted_idx_list

    def get_probability_distribution(self,
                                     state_list: [BoardGameBaseState]):

        x, inverted_idx_list = self._mabye_invert_input(state_list)
        # predict
        prob_dist = self.forward(torch.tensor(x, dtype=torch.float))
        prob_dist = prob_dist.tolist()

        # de invert if necessary
        for inverted_idx in inverted_idx_list:
            prob_dist[inverted_idx] = self.environment.invert_action_space_vec(prob_dist[inverted_idx])

        return prob_dist

    def save_model(self,
                   fp):
        torch.save(self.state_dict(), fp)

    @staticmethod
    def load_model(fp,
                   *args):
        model = BoardGameActorNeuralNetwork(*args)
        model.load_state_dict(torch.load(fp))
        model.eval()
        return model
