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
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.rl_agent import MonteCarloTreeSearchAgent

# import multiprocessing
import torch

from rl_agent.util import NeuralNetworkConfig


def main():
    torch.set_num_threads(1)

    env = HexGameEnvironment(
        board_size=7,
        internal_board_size=7
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=5,
        lr=None,
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=5,
        lr=None,
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=1500,
        topp_saves=100,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=43,
        worker_fork_number=6,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    # mcts = MontecarloTreeSearch(
    #     exploration_c=agent.exploration_c,
    #     environment=agent.environment,
    #     agent=agent,
    #     worker_thread_count=agent.worker_thread_count,
    #     worker_fork_number=agent.worker_fork_number
    # )
    # init = env.get_initial_state()
    # # # init._board_vec = [0, 1, -1, -1, 1, 0, 0, 1, 0, -1, -1, 1, 0, 0, 1, 0, -1, -1, -1, 0, 0, -1, 1, 0, -1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    # # init._board_vec = [1, 1, 1, 0, -1, -1, 1, 1, 1, -1, 1, 1, 1, 0, 1, 0, 0, -1, 0, -1, 1, 1, 1, 1, -1, 1, 0, 0, -1, 1, 1, 1, 1, -1, 1, 0, 0, 0, -1, -1, 1, 1, 1, 0, 1, 0, 0, 1, 0, -1, 1, 1, 1, -1, 1,
    # #                    1, 0, 0, 0, -1, 1,
    # #                    1, 1, -1, 1, 1, 0, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # init._board_vec = [1, 1, 1, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 1, 1, 1, 0, -1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, -1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
    #                    0, 0, 0, 1, 1, 1, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    # env.display_state(init)
    # ns, r, done = env.act(state=init, action=(3, 6), inplace=True)
    # print(ns, r, done)
    # env.display_state(init)
    # # init.change_turn()
    # # print(env.get_state_winning_move(state=init))
    # exit()

    # model_fp = "self_play/test_1"
    model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = None

    # agent.save_actor_critic_to_fp(model_fp)
    # exit()

    agent.load_model_from_fp(model_fp)

    # agent.run_self_training(
    #     num_games=10000,
    #     discount=0.9,
    #     lr=1,
    #     topp_saves=10,
    # )

    # agent.display = True
    # agent.debug = True
    #
    # agent.play_against_human()
    agent.train_n_episodes(
        n=10000,
        fp=model_fp,
        games_in_topp_matches=200
    )
    # agent.run_topp(100, num_games=500)
    exit()


if __name__ == '__main__':
    freeze_support()
    main()
