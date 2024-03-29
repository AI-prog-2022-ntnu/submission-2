import copy
import math
import time

import torch

from ActorClient import ActorClient
from enviorments.hex.hex_game import HexGameEnvironment, HexBoardGameState
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.rl_agent import MonteCarloTreeSearchAgent
from rl_agent.util import NeuralNetworkConfig


def map_board(list,
              internal_s,
              b_size,
              init_state: HexBoardGameState,
              turn):
    y_shift = internal_s - b_size
    for x in range(b_size):
        for y in range(b_size):
            val = list[(y * b_size) + x]

            if True:  # turn == 1:
                if val == 2:
                    val = -1
                elif val == 1:
                    val = 1
            # else:
            #     if val == 2:
            #         val = -1
            #     elif val == 1:
            #         val = 1

            init_state.set_board_val(x, y + y_shift, val)
    return init_state


class OHTClient(ActorClient):
    def __init__(self,
                 qualify,
                 auth):
        super().__init__(qualify=qualify, auth=auth)

        torch.set_num_threads(1)
        self.env = HexGameEnvironment(
            board_size=7,
            internal_board_size=7
        )

        actor_nn_config = NeuralNetworkConfig(
            episode_train_time_ms=1000,
            train_iterations=0,
            data_passes=200,
            batch_size=10,
            lr=0.02,
            nr_layers=2,
        )

        critic_nn_config = NeuralNetworkConfig(
            episode_train_time_ms=1000,
            train_iterations=0,
            data_passes=200,
            batch_size=10,
            lr=0.02,
            nr_layers=10,
        )

        self.agent = MonteCarloTreeSearchAgent(
            ms_tree_search_time=900,
            topp_saves=10,
            environment=self.env,
            # exploration_c=math.sqrt(2),
            exploration_c=1,
            worker_thread_count=11,
            worker_fork_number=3,
            actor_nn_config=actor_nn_config,
            critic_nn_config=critic_nn_config,
        )
        self.mcts = MontecarloTreeSearch(
            is_training=False,
            exploration_c=self.agent.exploration_c,
            environment=self.agent.environment,
            agent=self.agent,
            worker_thread_count=self.agent.worker_thread_count,
            worker_fork_number=self.agent.worker_fork_number
        )
        model_fp = "saved_models/model_10x10_vers_1"
        # model_fp = None
        self.agent.load_model_from_fp(model_fp)
        self.b_size = 0
        self.series_p_name = None
        self.p_1_wins = 0
        self.p_2_wins = 0
        self.current_game_num = 0
        self.tot_games = 0
        self.is_player_1 = True
        pass

    def handle_series_start(
            self,
            unique_id,
            series_id,
            player_map,
            num_games,
            game_params
    ):
        self.series_p_name = series_id
        self.env.board_size = game_params[0]
        print(f"\nis playing as {self.series_p_name}\n")

        self.p_1_wins = 0
        self.p_2_wins = 0
        self.current_game_num = 0
        self.tot_games = num_games

        self.logger.info(
            'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
            ', game_params=%s',
            unique_id, series_id, player_map, num_games, game_params,
        )

    def handle_get_action(self,
                          state):

        print()
        print("NEW MOVE:")
        # start_t = time.monotonic_ns()
        player = state[0] % 2
        used_state = self.env.get_initial_state()
        used_state.players_turn = player - 1

        print("us pt ", used_state.players_turn)
        print("pt ", player)
        # print("pre", state)
        s_vec = map_board(list(state[1:]), self.env.internal_board_size, self.env.board_size, init_state=used_state, turn=player)

        # used_state.players_turn = 0
        # print("post", s_vec.get_as_vec())

        self.env.display_state(used_state, display_internal=False)

        winning_move = self.env.get_state_winning_move(used_state)

        other_s = copy.deepcopy(used_state)
        other_s.change_turn()
        losing_move = self.env.get_state_winning_move(other_s)

        if winning_move is not None:
            print("has winning", winning_move)
            action = winning_move
        elif losing_move is not None:
            print("has losing", losing_move)
            action = losing_move
        else:

            prob_dist = self.agent.model.get_probability_distribution([used_state])[0]

            target_val = max(prob_dist)
            action_idx = prob_dist.index(target_val)
            action = self.env.get_action_space()[action_idx]

        # action = self.agent.get_tree_search_action(used_state, self.mcts)
        print(action)
        y_shift = self.env.internal_board_size - self.env.board_size
        x = action[0]
        y = action[1] - y_shift

        return y, x

    def handle_game_over(self,
                         winner,
                         end_state):
        print("\nGAME OVER\n ")
        print(f"winner is {winner}")
        self.current_game_num += 1

        if winner == 1:
            self.p_1_wins += 1
        else:
            self.p_2_wins += 1
        used_state = self.env.get_initial_state()
        s_vec = map_board(list(end_state[1:]), self.env.internal_board_size, self.env.board_size, init_state=used_state, turn=end_state[0])
        self.env.display_state(used_state)
        print("game {:>3} of {:>3} win /lose: \u001b[32m{:>6}\u001b[0m / \u001b[31m{:<6}\u001b[0m".format(self.current_game_num, self.tot_games, self.p_1_wins, self.p_2_wins))
        print("\n")


if __name__ == '__main__':
    # client = OHTClient(qualify=False, auth="b9e9a8d2199e48c7857967457401b14a")
    client = OHTClient(qualify=False, auth="9ce610bf4cbd4e20a79fdbbe51295981")
    # client = OHTClient()
    client.run()
