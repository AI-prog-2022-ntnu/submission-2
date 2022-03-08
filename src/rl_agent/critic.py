import math
import random

import numpy as np
import torch
from torch import nn

from enviorments.base_state import BaseState, GameBaseState


class CriticNeuralNet(nn.Module):
    def __init__(self,
                 inp_s,
                 b_size):
        super(CriticNeuralNet, self).__init__()

        self.device_used = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.network = nn.Sequential(
        #     nn.Linear(inp_s, 200),
        #     nn.ELU(),
        #     nn.Linear(200, 200),
        #     nn.ELU(),
        #     nn.Linear(200, 1),
        #     nn.Tanh(),
        # )

        self.loss_fn = torch.nn.L1Loss().to(device=self.device_used)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=10,
                kernel_size=(5, 5),
                padding=2,
                # stride=2
            ),
            # nn.Hardtanh(),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=10,
                out_channels=30,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((b_size * b_size) * 30, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            # nn.ReLU()  # <- DONT CHANGE
            # nn.Tanh(),
            # nn.Softmax()
        ).to(device=self.device_used)

        self.opt = torch.optim.Adam(self.parameters())
        self.b_size = b_size

    def forward(self,
                inp_x):
        x = inp_x.to(device=self.device_used)

        p1_inp = torch.where(x == 1, 1.0, 0.0)
        p2_inp = torch.where(x == -1, 1.0, 0.0)

        p_stack = torch.stack([p1_inp, p2_inp], dim=1)

        x = p_stack.view((-1, 2, self.b_size, self.b_size))

        # x = x.view((-1, 1, self.b_size, self.b_size))
        return self.network(x)

    def train_network(self,
                      inp_x,
                      inp_y,
                      train_itrs,
                      batch_s):
        # x, y = inp_x.to(device=self.device_used), inp_y.to(device=self.device_used)
        # self.train(True)
        # pred = self.forward(x)
        # # print("y:    ", y)
        # # print("pred: ", pred)
        # # print(pred)
        # # print("pred ", pred.get_device())
        # # print("y ", y.get_device())
        #
        # loss: torch.Tensor = self.loss_fn(pred, y)
        #
        # # print(loss)
        # self.opt.zero_grad()
        # loss.backward()
        # self.opt.step()
        # return loss

        x = torch.tensor(inp_x, dtype=torch.float, device=self.device_used)
        y = torch.tensor(inp_y, dtype=torch.float, device=self.device_used)

        # x, y = inp_x.to(device=self.device_used), inp_y.to(device=self.device_used)

        self.to(device=self.device_used)
        self.loss_fn.to(device=self.device_used)
        self.train(True)
        loss_values = torch.zeros(train_itrs)

        print(f"begin critic training for {train_itrs} iterations")

        for n in range(train_itrs):
            print("{:>5}/{:<5}".format(n, train_itrs), end="\r")
            rand_idx = torch.randint(len(x), (batch_s,))
            x = x[rand_idx]
            y = y[rand_idx]
            pred = self.forward(x)
            # print(y)
            # print(pred)
            # print("y:    ", y)
            # print("pred: ", pred)
            # print(pred)

            loss: torch.Tensor = self.loss_fn(pred, y)
            loss_values[n] = loss

            # print(loss)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        print()

        self.train(False)
        self.to(device="cpu")
        return loss_values.tolist()

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
                 input_space,
                 b_size):
        self.b_size = b_size
        self.input_space = input_space

        self.loss_hist = []
        self.train_buffer = []

        self._gen_model()

    def _gen_model(self):
        self.model = CriticNeuralNet(self.input_space, b_size=self.b_size)

    def train_network(self,
                      values):
        if len(values) == 0:
            print("No values to train critic")
            return
        device_used = "cuda:0"
        # model = self.model.to(device=device_used)
        # model.train(True)
        # torch.set_printoptions(profile="full", linewidth=1000)
        loss_fn = torch.nn.L1Loss()  # .cuda(device_used)
        # opt = torch.optim.Adam(model.parameters())
        passes_over_data = 50
        batch_size = 20
        train_itrs = math.ceil((len(values) * passes_over_data) / batch_size)

        x_list, y_list = [], []
        for x, y in generate_batch(values, batch_size):
            x_list.append(x)
            y_list.append([y])

        # for n in range(train_itrs):
        #     print("{:>5}/{:<5}".format(n, train_itrs), end="\r")

        # x = torch.tensor(x_list, dtype=torch.float, device=device_used)
        # y = torch.tensor(y_list, dtype=torch.float, device=device_used)

        # x = torch.tensor(x_list, dtype=torch.float)
        # y = torch.tensor(y_list, dtype=torch.float)
        loss = self.model.train_network(x_list, y_list, train_itrs, batch_size)
        self.loss_hist.extend(loss)
        # pred = model.forward(x).cuda()
        # # print("y:    ", y)
        # # print("pred: ", pred)
        # # print(pred)
        # print("pred ", pred.get_device())
        # print("y ", y.get_device())
        #
        # loss: torch.Tensor = loss_fn(pred, y)
        # self.loss_hist.append(loss.item())
        #
        # # print(loss)
        # opt.zero_grad()
        # loss.backward()
        # opt.step()
        print()

        full_x, full_y = [], []
        for x, y in values:
            full_x.append(x)
            full_y.append([y])

        full_pred = self.model.forward(torch.tensor(full_x, dtype=torch.float))
        full_y_tensor = torch.tensor(full_y, dtype=torch.float)
        tot_set_loss = loss_fn(full_pred.to(device="cpu"), full_y_tensor)
        print(f"\nCritic_loss: {tot_set_loss}")

    def get_state_value(self,
                        state: GameBaseState,
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
