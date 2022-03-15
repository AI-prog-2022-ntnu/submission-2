import math

from ActorClient import ActorClient
from enviorments.hex.hex_game import HexGameEnvironment, HexBoardGameState
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.rl_agent import MonteCarloTreeSearchAgent

env = HexGameEnvironment(5)

agent = MonteCarloTreeSearchAgent(
    num_rollouts=1000,
    ms_tree_search_time=900,
    topp_saves=10,
    environment=env,
    # exploration_c=math.sqrt(2),
    exploration_c=1,
    worker_thread_count=10,
)

model_fp = "saved_models/model_5x5_vers_5"
# model_fp = None
agent.load_model_from_fp(model_fp)

mcts = MontecarloTreeSearch(
    exploration_c=math.sqrt(2),
    environment=env,
    agent=agent,
    worker_thread_count=10
)


def map_board(list):
    b_size = math.floor(math.sqrt(len(list)))
    board_list = [[0 for _ in range(b_size)] for _ in range(b_size)]
    print(b_size)
    for x in range(b_size):
        for y in range(b_size):
            val = list[(x * 5) + y]

            if val == 2:
                val = -1

            board_list[x][y] = val
    return board_list


class OHTClient(ActorClient):
    def handle_get_action(self,
                          state):
        print(state)
        player = state.pop(0) - 1
        s_vec = map_board(state)
        print(s_vec)

        actual_state = HexBoardGameState(
            board=s_vec
        )
        env.display_state(actual_state)
        actual_state.players_turn = player
        action = agent.get_tree_search_action(actual_state, mcts)
        print(action)
        return action[1] + 1, action[0] + 1


if __name__ == '__main__':
    client = OHTClient(qualify=False, auth="b9e9a8d2199e48c7857967457401b14a")
    # client = OHTClient()
    client.run()
