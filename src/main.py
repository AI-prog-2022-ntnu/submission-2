import copy
import math
import random
import time
from multiprocessing import freeze_support

import numpy as np
import torch.cuda
import seaborn as sns
from matplotlib import pyplot as plt

from enviorments.hex.hex_game import HexGameEnvironment, make_random_hex_board, invert
from rl_agent.rl_agent import MonteCarloTreeSearchAgent

# import multiprocessing
import torch


def main():
    torch.set_num_threads(1)
    import os

    # from concurrent.futures import ProcessPoolExecutor
    # from multiprocessing import JoinableQueue, Queue, Process
    #
    # executor = ProcessPoolExecutor()
    # a = [0 for _ in range(10)]
    #
    #
    # def abc(tr,
    #         que: Queue):
    #     while True:
    #         val = que.get(block=True)
    #         time.sleep(0.5)
    #         print(tr, val)
    #
    #
    # futures = []
    # que = Queue()
    # for n in range(3):
    #     # f = executor.submit(abc, n, que)
    #     p = Process(target=abc, args=(n, que))
    #     p.start()
    #     # futures.append(f)
    #
    # for i in range(10):
    #     que.put(i)
    #     time.sleep(0.3)
    #
    # time.sleep(10)
    # for future in futures:
    #     r = future.cancel()
    #     print(r)

    # print(a)

    env = HexGameEnvironment(5)

    agent = MonteCarloTreeSearchAgent(
        num_rollouts=3000,
        environment=env,
        worker_thread_count=10,
        exploration_c=math.sqrt(2)
    )

    agent.run_topp(500)
    # agent.debug = True

    # s = env.get_initial_state()
    # board = make_random_hex_board(5)
    # s.hex_board = board
    # env.display_state(s)
    #
    # inverted_board = copy.deepcopy(board)
    #
    # axis_size = len(board)
    # for x in range(axis_size):
    #     for y in range(axis_size):
    #         inverted_board[x][y] = invert(board[y][x])
    # s2 = copy.deepcopy(s)
    # s2.hex_board = inverted_board
    # env.display_state(s2)
    #

    #
    # pd = agent.get_prob_dists([s, s2])
    # s_mod = copy.deepcopy(pd[1])
    #
    # for x in range(axis_size):
    #     for y in range(axis_size):
    #         s_mod[(x * axis_size) + y] = pd[1][(y * axis_size) + x]
    #
    # print("pd 0:     ", pd[0])
    # print("pd 1:     ", pd[1])
    # print("pd 1 inv: ", s_mod)
    #
    # exit()

    # print(torch.cuda.is_available())

    model_fp = "saved_models/model_5x5_test"
    # model_fp = None
    agent.load_model_from_fp(model_fp)

    agent.display = True
    agent.debug = True
    # agent.run_episode()
    # agent.display = False
    # agent.debug = False
    # exit()

    # agent.train_n_episodes(10)
    # agent.display = True
    # agent.train_n_episodes(500, model_fp)

    # agent.train_n_episodes(10, model_fp)

    if False:
        plot = sns.lineplot(x=list(range(len(agent.loss_hist))), y=agent.loss_hist)
        plot.get_figure().savefig("out_actor.png")

        plt.clf()

        plot = sns.lineplot(x=list(range(len(agent.critic.loss_hist))), y=agent.critic.loss_hist)
        plot.get_figure().savefig("out_critic.png")

    agent.debug = True
    # agent.display = True
    # agent.train_n_episodes(5)
    agent.play_against_human()

    # init_s = env.get_initial_state()
    # init_s.hex_board = [[1, 1, -1], [1, 1, -1], [-1, -1, 1]]

    # init_s.hex_board = [[1, -1, -1, 1], [-1, 1, 1, -1], [1, -1, 0, -1], [-1, 1, 1, 1]]
    # init_s.get_as_inverted_vec()
    # env.display_state(init_s)

    # inverted
    # init_s.hex_board = [[-1, 1, -1, 1], [1, -1, 1, -1], [1, -1, 0, -1], [-1, 1, 1, -1]]
    # env.display_state(init_s)

    # print(init_s.get_as_vec())
    # print(init_s.get_as_inverted_vec())
    # env.display_state(init_s)

    # init_s = env.get_initial_state()
    # env.display_state(init_s)

    # num = 1000
    # b_size = 16
    # tb = []
    # for _ in range(num):
    #     x = [random.randint(0,1) for n in range(b_size)]
    #     y = [random.random() for n in range(b_size)]
    #     tb.append((x,y))
    # agent = MonteCarloTreeSearchAgent(env)
    #
    # x = [random.randint(0, 1) for n in range(b_size)]
    # init_s.hex_board[0][0] = 1
    # a = agent.pick_action(init_s, env)
    # print(a)
    # agent.train_buffer = tb
    # agent.train()
    # env.display_state(init_s)
    # print(env.is_state_won(init_s))


if __name__ == '__main__':
    freeze_support()
    main()
