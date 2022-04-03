import random
from enum import Enum

import torch
from torch import nn


class NeuralNetworkConfig:
    def __init__(self,
                 episode_train_time_ms,
                 train_iterations,
                 batch_size,
                 lr,
                 activation_function="relu",
                 optimizer="adam",
                 data_passes=None,
                 use_iter=True,
                 nr_layers=None,

                 ):
        self.data_passes = data_passes
        self.use_iter = use_iter
        self.train_iterations = train_iterations
        self.lr = lr
        self.batch_size = batch_size
        self.episode_train_time_ms = episode_train_time_ms
        self.nr_layers = nr_layers
        self.activation_function = self.set_activation_function(activation_function)
        self.optimizer = self.set_optimizer(optimizer)

    def set_activation_function(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "linear":
            # TODO: find out if this applies to cnn?
            # return nn.Linear()
            pass
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise Exception("invalid activation function")

    def set_optimizer(self, name):
        if name == "adagrad":
            return torch.optim.Adagrad
        elif name == "sgd":
            return torch.optim.SGD
        elif name == "rms":
            return torch.optim.RMSprop
        elif name == "adam":
            return torch.optim.Adam
        else:
            raise Exception("invalid optimizer")


class EGreedy:
    def __init__(self,
                 init_val,
                 min_val,
                 rounds_to_min):
        self._init_epsilon = init_val
        self.epsilon = init_val
        self._epsilon_decay = (self.epsilon - min_val) / rounds_to_min
        self._epsilon_lb = min_val

    def round(self):
        self.epsilon = max(self.epsilon - self._epsilon_decay, self._epsilon_lb)

    def reset(self):
        self.epsilon = self._init_epsilon

    def should_pick_greedy(self,
                           increment_round=False):
        ret = False
        if random.random() > self.epsilon:
            ret = True

        if increment_round:
            self.round()
        return ret


def generate_batch(x,
                   y,
                   batch_size):
    batch_x = []
    batch_y = []
    while len(batch_x) < batch_size:
        idx = random.choice(range(len(x)))
        batch_x.append(x[idx])
        batch_y.append(y[idx])
    return batch_x, batch_y


def get_action_visit_map_as_target_vec(environment,
                                       action_visit_map: {},
                                       invert=False):
    possible_actions = environment.get_action_space()
    visit_sum = sum(action_visit_map.values())

    ret = []
    for action in possible_actions:
        if invert:
            value = action_visit_map.get((action[1], action[0]))
        else:
            value = action_visit_map.get(action)

        if value is None or visit_sum == 0:
            ret.append(0)
        else:
            ret.append(value / visit_sum)
    return ret


class ActivationFuncs(Enum):
    RELU = "relu"
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class Optimizers(Enum):
    ADAGRAD = "adagrad"
    SGD = "sgd"
    RMS = "rms"
    ADAM = "adam"
