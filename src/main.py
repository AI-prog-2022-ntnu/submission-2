from multiprocessing import freeze_support

# import multiprocessing
import torch
import torch.cuda

from enviorments.hex.hex_game import HexGameEnvironment
from rl_agent.rl_agent import MonteCarloTreeSearchAgent
from rl_agent.util import ActivationFuncs
from rl_agent.util import NeuralNetworkConfig
from rl_agent.util import Optimizers


def play_against_human(board_size=7,
                       internal_board_size=7):
    env = HexGameEnvironment(
        board_size=board_size,
        internal_board_size=internal_board_size
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.002,
        nr_layers=2,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.0005,
        nr_layers=10,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=2000,
        topp_saves=10,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=11,
        worker_fork_number=9,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    # model_fp = None
    model_fp = "saved_models/model_10x10_vers_1"
    agent.load_model_from_fp(model_fp)

    agent.display = True
    agent.debug = True
    agent.play_against_human()
    exit()


def play_TOPP(board_size=7,
              internal_board_size=7):
    env = HexGameEnvironment(
        board_size=board_size,
        internal_board_size=internal_board_size
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.002,
        nr_layers=2,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.0005,
        nr_layers=10,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=1500,
        topp_saves=100,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=11,
        worker_fork_number=5,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = None

    agent.load_model_from_fp(model_fp)

    agent.display = True
    agent.debug = True
    agent.train_n_episodes(
        n=200,
        fp=model_fp,
        games_in_topp_matches=100
    )
    agent.run_topp(100, num_games=500)
    exit()


def train_TOPP(board_size=7,
               internal_board_size=7):
    env = HexGameEnvironment(
        board_size=board_size,
        internal_board_size=internal_board_size
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.002,
        nr_layers=2,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=10,
        batch_size=10,
        lr=0.0005,
        nr_layers=10,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=1500,
        topp_saves=100,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=43,
        worker_fork_number=1,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = None

    agent.load_model_from_fp(model_fp)

    agent.run_self_training(
        num_games=10000,
        discount=0.9,
        lr=1,
        topp_saves=10,
    )

    exit()


def train(board_size=7,
          internal_board_size=7):
    env = HexGameEnvironment(
        board_size=board_size,
        internal_board_size=internal_board_size
    )

    actor_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=1,
        batch_size=50,
        lr=0.002,
        nr_layers=2,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    critic_nn_config = NeuralNetworkConfig(
        episode_train_time_ms=1000,
        train_iterations=0,
        data_passes=1,
        batch_size=50,
        lr=0.0005,
        nr_layers=10,
        activation_function=ActivationFuncs.RELU.value,
        optimizer=Optimizers.ADAM.value
    )

    agent = MonteCarloTreeSearchAgent(
        ms_tree_search_time=800,
        topp_saves=100,
        environment=env,
        # exploration_c=math.sqrt(2),
        exploration_c=1,
        worker_thread_count=43,
        worker_fork_number=1,
        actor_nn_config=actor_nn_config,
        critic_nn_config=critic_nn_config,
    )

    model_fp = "saved_models/model_10x10_vers_1"
    # model_fp = None

    agent.load_model_from_fp(model_fp)

    # agent.debug = True
    # agent.display = True
    agent.train_n_episodes(
        n=1000,
        fp=model_fp
    )

    exit()


def main():
    torch.set_num_threads(1)
    train(board_size=7, internal_board_size=7)
    # play_against_human()
    # play_TOPP()
    # train_TOPP()


if __name__ == '__main__':
    freeze_support()
    main()
