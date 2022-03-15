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

from rl_agent.util import NeuralNetworkConfig


def main():
    torch.set_num_threads(1)

    env = HexGameEnvironment(
        board_size=4,
        internal_board_size=10
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=2000,
        batch_size=5,
        lr=None,
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=2000,
        batch_size=5,
        lr=None,
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=500,
        topp_saves=10,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=10,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    # init = env.get_initial_state()
    # env.display_state(init)
    # exit()

    model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = None
    agent.load_model_from_fp(model_fp)

    agent.display = True
    agent.debug = True

    agent.train_n_episodes(100, model_fp)
    agent.run_topp(100, num_games=100)
    exit()


if __name__ == '__main__':
    freeze_support()
    main()
