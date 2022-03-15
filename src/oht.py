import math

from ActorClient import ActorClient
from enviorments.hex.hex_game import HexGameEnvironment, HexBoardGameState
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.rl_agent import MonteCarloTreeSearchAgent
from rl_agent.util import NeuralNetworkConfig


def map_board(list,
              internal_s,
              b_size,
              init_state: HexBoardGameState):
    y_shift = internal_s - b_size
    for x in range(b_size):
        for y in range(b_size):
            val = list[(x * b_size) + y]

            if val == 2:
                val = -1

            init_state.hex_board[x][y + y_shift] = val
    return init_state


class OHTClient(ActorClient):
    def __init__(self,
                 qualify,
                 auth):
        super().__init__(qualify=qualify, auth=auth)

        self.env = HexGameEnvironment(
            board_size=5,
            internal_board_size=10
        )

        actor_nn_config = NeuralNetworkConfig(
            episode_train_time_ms=1000,
            train_iterations=0,
            data_passes=200,
            batch_size=10,
            lr=None,
        )

        critic_nn_config = NeuralNetworkConfig(
            episode_train_time_ms=1000,
            train_iterations=0,
            data_passes=200,
            batch_size=10,
            lr=None,
        )

        self.agent = MonteCarloTreeSearchAgent(
            ms_tree_search_time=900,
            topp_saves=10,
            environment=self.env,
            # exploration_c=math.sqrt(2),
            exploration_c=1,
            worker_thread_count=10,
            actor_nn_config=actor_nn_config,
            critic_nn_config=critic_nn_config,
        )
        self.mcts = MontecarloTreeSearch(
            exploration_c=self.agent.exploration_c,
            environment=self.agent.environment,
            agent=self.agent,
            worker_thread_count=self.agent.worker_thread_count
        )
        model_fp = "saved_models/model_10x10_vers_1"
        # model_fp = None
        self.agent.load_model_from_fp(model_fp)
        self.b_size = 0
        self.series_p_name = None
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

    def handle_get_action(self,
                          state):
        player = state.pop(0) % 2
        used_state = self.env.get_initial_state()
        used_state.players_turn = player
        print("pre", state)
        s_vec = map_board(list(state), self.env.internal_board_size, self.env.board_size, init_state=used_state)
        print("post", s_vec.get_as_vec())

        self.env.display_state(used_state)
        action = self.agent.get_tree_search_action(used_state, self.mcts)
        print(action)
        y_shift = self.env.internal_board_size - self.env.board_size
        x = action[0]
        y = action[1] - y_shift
        return x, y


if __name__ == '__main__':
    client = OHTClient(qualify=False, auth="b9e9a8d2199e48c7857967457401b14a")
    # client = OHTClient()
    client.run()
