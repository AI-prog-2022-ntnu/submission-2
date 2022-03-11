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
from rl_agent.mc_tree_search import min_max_search, probe_n
from rl_agent.rl_agent import MonteCarloTreeSearchAgent

# import multiprocessing
import torch


def main():
    torch.set_num_threads(1)

    env = HexGameEnvironment(5)

    agent = MonteCarloTreeSearchAgent(
        num_rollouts=1000,
        ms_tree_search_time=1000,
        topp_saves=10,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=10,
    )

    model_fp = "saved_models/model_5x5_vers_5"
    # model_fp = None
    agent.load_model_from_fp(model_fp)

    agent.display = True
    agent.debug = True

    agent.train_n_episodes(1000, model_fp)
    agent.run_topp(2000, num_games=1000)
    exit()

    if False:
        plot = sns.lineplot(x=list(range(len(agent.loss_hist))), y=agent.loss_hist)
        plot.get_figure().savefig("out_actor.png")

        plt.clf()

        plot = sns.lineplot(x=list(range(len(agent.critic.loss_hist))), y=agent.critic.loss_hist)
        plot.get_figure().savefig("out_critic.png")

    agent.debug = True
    # agent.display = True
    # agent.train_n_episodes(5)
    # agent.play_against_human()

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
