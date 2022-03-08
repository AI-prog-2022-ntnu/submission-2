import math
import random

import numpy as np
import torch
from torch import nn


class ActorNeuralNetwork(nn.Module):
    def __init__(self,
                 inp_s,
                 out_s,
                 b_size):
        super(ActorNeuralNetwork, self).__init__()

        self.train_device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.network = nn.Sequential(
        #     nn.Linear(inp_s, 100),
        #     nn.ELU(),
        #     nn.Linear(100, 100),
        #     nn.ELU(),
        #     nn.Linear(100, out_s),
        #     # nn.ReLU()  # <- DONT CHANGE
        #     nn.Tanh()
        #     # nn.Softmax()
        # )

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=30,
                kernel_size=(5, 5),
                padding=2,
                # stride=2
            ),
            # nn.Hardtanh(),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=30,
                out_channels=30,
                kernel_size=(3, 3),
                padding=1
                # stride=2
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((b_size * b_size) * 30, out_s),
            nn.ReLU()  # <- DONT CHANGE
            # nn.Tanh(),
            # nn.Softmax()
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters())
        self.b_size = b_size

    def forward(self,
                inp_x,
                use_cuda=False):
        if use_cuda:
            x = inp_x.to(device=self.train_device)
        else:
            x = inp_x
        zero_inp = torch.where(x == 0, 1, 0)

        p1_inp = torch.where(x == 1, 1.0, 0.0)
        p2_inp = torch.where(x == -1, 1.0, 0.0)

        p_stack = torch.stack([p1_inp, p2_inp], dim=1)

        x = p_stack.view((-1, 2, self.b_size, self.b_size))
        # print(x.dtype)
        # x.type(torch.LongTensor)
        # x = x.long()
        # print(x.dtype)
        # print(x)
        # print(x.size())

        # batch_s * channels * witdth * height
        # x = x.view((-1, 1, self.b_size, self.b_size))

        out: torch.Tensor = self.network(x)
        # print(out)
        # print(out.size())

        # e_pov_v = torch.exp(out)
        # soft_m_sum = torch.sum(input=torch.multiply(e_pov_v, zero_inp), dim=1)
        #
        # s = e_pov_v.size()
        # s1, s2 = s[0], s[1]
        #
        # soft_m = torch.div(e_pov_v, soft_m_sum.unsqueeze(1).expand(s1, s2)).float()
        # soft_m_filtered = torch.multiply(soft_m, zero_inp)

        used_vals = torch.multiply(out, zero_inp)
        soft_m_sum = torch.sum(input=used_vals, dim=1)
        # if soft_m_sum.float().sum() == 0:
        #     soft_m = used_vals
        # else:
        s = out.size()
        s1, s2 = s[0], s[1]
        soft_m = torch.div(used_vals, soft_m_sum.unsqueeze(1).expand(s1, s2)).float()
        soft_m = torch.nan_to_num(soft_m, 0)  # TODO: this is bad shold find better solution

        thisisjustpoorsoftwaredesign = torch.tensor([0], dtype=torch.float)
        soft_m_filtered = torch.where(zero_inp == 1, soft_m, thisisjustpoorsoftwaredesign)

        if torch.any(torch.isnan(soft_m_filtered)):
            print("#############")
            print("x", x)
            print("out vec", out)
            print("zero inp mask", zero_inp)
            print("soft m filtered ", soft_m_filtered)
            print("soft m sum ", soft_m_sum)
            # print(soft_m_sum.unsqueeze(1).expand(s1, s2))
            print(soft_m)
            print("#############")
            exit()
        return soft_m_filtered

        # if torch.any(torch.isnan(soft_m_filtered)):
        # print("#############")
        # print(x)
        # print(u)
        # print(soft_m)
        # print(out)
        # print("#############")
        # return soft_m_filtered

    def train_network(self,
                      inp_x,
                      inp_y,
                      train_itrs,
                      batch_s):

        x = torch.tensor(inp_x, dtype=torch.float, device=self.train_device)
        y = torch.tensor(inp_y, dtype=torch.float, device=self.train_device)

        # x, y = inp_x.to(device=self.device_used), inp_y.to(device=self.device_used)

        self.to(device=self.train_device)
        self.loss_fn.to(device=self.train_device)
        self.train(True)
        loss_values = torch.zeros(train_itrs)

        print(f"begin actor training for {train_itrs} iterations")

        for n in range(train_itrs):
            print("{:>5}/{:<5}".format(n, train_itrs), end="\r")
            rand_idx = torch.randint(len(x), (batch_s,))
            x = x[rand_idx]
            y = y[rand_idx]
            pred = self.forward(x, use_cuda=True)
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
        model = ActorNeuralNetwork(*args)
        model.load_state_dict(torch.load(fp))
        model.eval()
        return model
