import copy
import math
import time

import torch
from torch import nn

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
from rl_agent.util import NeuralNetworkConfig


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
        self.nn_config = copy.deepcopy(nn_config)

        self.board_size = math.floor(math.sqrt(input_size))  # TODO: this is bad

        if self.invert_p2:
            self.input_size = input_size
        else:
            self.input_size = input_size + 1

        self.network = self._get_conv_network()
        self.soft_max = torch.nn.Softmax(dim=-1)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = self.nn_config.optimizer(self.network.parameters(), lr=self.nn_config.lr)
        self.opt = torch.optim.RMSprop(self.parameters())

    def _get_conv_network(self):
        """
        Creates the conv network model.
        """

        network = nn.Sequential(*self.nn_config.network_layout)

        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        network.apply(init_weights)
        return network

    def forward(self,
                inp_x,
                ):
        x = inp_x

        if self.invert_p2:
            keep_filter = x == 0
        else:
            pass

        p1_inp = torch.where(x == 1, 1.0, 0.0)
        p2_inp = torch.where(x == -1, 1.0, 0.0)

        p_stack = torch.stack([p1_inp, p2_inp], dim=1)
        x = p_stack.view((-1, 2, self.board_size, self.board_size))
        out: torch.Tensor = self.network(x)
        # print(out.shape)
        # exit()

        out_soft_masked = torch.zeros_like(out)
        for n in range(len(x)):
            valid_moves = out[n, keep_filter[n, :]]
            soft_m_moves = valid_moves  # self.soft_max(valid_moves)
            soft_m_moves = torch.nan_to_num(soft_m_moves)
            out_soft_masked[n, keep_filter[n, :]] = soft_m_moves

        return out_soft_masked

    def _train_network(self,
                       inp_x,
                       inp_y):
        """
        Trains the network.
        """
        self.train(True)
        loss_values = []

        x_inp = torch.tensor(inp_x, dtype=torch.float)
        y_inp = torch.tensor(inp_y, dtype=torch.float)

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
            rnds += 1
            rand_idx = torch.randint(set_len, (self.nn_config.batch_size,))
            x = x_inp[rand_idx]
            y = y_inp[rand_idx]

            # Sets gradients of all model parameters to zero
            self.network.zero_grad()
            self.opt.zero_grad()

            # Creates a prediction
            pred = self.forward(x)

            _, labels = torch.max(y, dim=1)

            # Gets the cross entropy loss from the prediction.
            loss: torch.Tensor = self.loss_fn(pred, labels)
            loss_values.append(loss.tolist())

            # Computes the gradient of current tensor w.r.t. graph leaves
            loss.backward()

            # Performs a single optimization step
            self.opt.step()
            if self.nn_config.data_passes is not None:
                stop = (rnds / self.nn_config.batch_size) > self.nn_config.data_passes
            else:
                stop = rnds > self.nn_config.train_iterations
            if stop_t is not None and not stop:
                stop = time.monotonic_ns() > stop_t

        self.train(False)
        return loss_values

    def get_action_visit_map_as_target_vec(self,
                                           action_visit_map: {}):
        """
        Returns the action visit map as a target vector.
        """
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
        """
        Trains the network and returns the loss.
        """
        torch.set_printoptions(profile="full", linewidth=1000)

        raw_x, y = [], []

        for state, v_count_map in state_buffer:
            raw_x.append(state)
            y.append(self.get_action_visit_map_as_target_vec(v_count_map))

        x, inverted_idx_list = self._maybe_invert_input(raw_x)

        for inverted_idx in inverted_idx_list:
            y[inverted_idx] = self.environment.invert_action_space_vec(y[inverted_idx])

        loss = self._train_network(x, y)
        return loss

    def _maybe_invert_input(self,
                            inp):
        """
        Inverts the input if the current player is not the player with id 0.
        """
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
        """
        Returns the probability distribution.
        """

        x, inverted_idx_list = self._maybe_invert_input(state_list)
        # predict
        prob_dist = self.forward(torch.tensor(x, dtype=torch.float))
        prob_dist[prob_dist != 0] = self.soft_max(prob_dist[prob_dist != 0])
        prob_dist = prob_dist.tolist()

        # de invert if necessary
        for inverted_idx in inverted_idx_list:
            prob_dist[inverted_idx] = self.environment.invert_action_space_vec(prob_dist[inverted_idx])

        return prob_dist

    def save_model(self,
                   fp):
        """
        Saves the model.
        """
        torch.save(self.state_dict(), fp)

    @staticmethod
    def load_model(fp,
                   *args):
        """
        Loads the model.
        """
        model = BoardGameActorNeuralNetwork(*args)
        model.load_state_dict(torch.load(fp))
        model.eval()
        return model
