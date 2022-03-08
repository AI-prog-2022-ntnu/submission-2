import math

from ActorClient import ActorClient
from enviorments.hex.hex_game import HexGameEnvironment, HexGameState
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.rl_agent import MonteCarloTreeSearchAgent

env = HexGameEnvironment(5)
agent = MonteCarloTreeSearchAgent(
    num_rollouts=2000,
    environment=env,
    worker_thread_count=10,
    exploration_c=math.sqrt(2)
)

model_fp = "saved_models/model_5x5_test"
agent.load_model_from_fp(model_fp)

mcts = MontecarloTreeSearch(
    exploration_c=math.sqrt(2),
    environment=env,
    agent=agent,
    worker_thread_count=10
)


def map_board(list):
    board_list = [[0 for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            val = list((x * 5) + y)

            if val == 2:
                val = -1

            board_list[x][y] = val


class OHTClient(ActorClient):
    def handle_get_action(self,
                          state):
        player = state.pop(0) - 1
        s_vec = map_board(state)

        actual_state = HexGameState(
            board=s_vec
        )
        actual_state.players_turn = player
        action = agent.get_tree_search_action(actual_state, mcts)
        return action[1], action[0]


if __name__ == '__main__':
    client = OHTClient(qualify=False, auth="b9e9a8d2199e48c7857967457401b14a")
    # client = OHTClient()
    client.run()
