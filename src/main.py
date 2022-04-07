import math
from multiprocessing import freeze_support

# import multiprocessing
import torch
from torch import nn
import torch.cuda

from enviorments.hex.hex_game import HexGameEnvironment
from rl_agent.rl_agent import MonteCarloTreeSearchAgent
from rl_agent.util import ActivationFuncs
from rl_agent.util import NeuralNetworkConfig
from rl_agent.util import Optimizers


def main():
    bs = 5
    env = HexGameEnvironment(
        board_size=bs,
        internal_board_size=bs
    )

    k = 10
    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=10000,
        train_iterations=None,
        data_passes=1,
        batch_size=80,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        network_layout=[
            nn.Conv2d(in_channels=2, out_channels=k, kernel_size=(5, 5), padding=2, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(3, 3), padding=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(k * (bs - 1) ** 2, out_features=300),
            nn.Linear(300, bs ** 2),
        ]

    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=1000,
        topp_saves=20,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=11,
        worker_fork_number=3,
        actor_nn_config=actor_nn_config,
    )

    # model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = "saved_models/model_5x5_vers_1"
    model_fp = "saved_models/model_7x7_vers_1"
    # model_fp = None

    agent.load_model_from_fp(model_fp)

    agent.debug = True
    agent.display = True
    agent.train_n_episodes(
        n=100,
        fp=model_fp
    )

    agent.run_topp(
        n=100,
        num_games=500
    )


if __name__ == '__main__':
    torch.set_num_threads(1)
    freeze_support()
    main()
