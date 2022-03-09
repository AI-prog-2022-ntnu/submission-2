import numpy as np



# class NodeVisitCounter:
#     def __init__(self):
#         self.visits = 0
#         self.value = 0
from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState


class TreeSearchNode:
    def __init__(self,
                 game_state,
                 children,
                 action_from_parent,
                 visits,
                 value,
                 terminal,
                 fully_expanded=False,
                 rolled_out=False):

        #
        self.children: {int: TreeSearchNode} = children

        # has the node been rolled out
        self.rolled_out = rolled_out

        # is the node fully expanded
        self.fully_expanded = fully_expanded

        # The value of the node
        self.value = value

        # The number of visits to the node
        self.visits = visits

        # The action from the parent node to this
        self.action_from_parent = action_from_parent


        # The hash of the current game state
        # This is stored instead of the state to allow for changing the state object in place
        self.game_state_hash = game_state

        # Is the node a terminal (end game) node
        self.terminal = terminal

    @staticmethod
    def generate_child_node(state: BoardGameBaseState,
                            environment: BoardGameEnvironment,
                            action
                            ):
        if environment.game_has_reversible_moves():
            next_s, reward, done = environment.act(state=state,action=action,inplace=True)
            environment.reverse_move(state=state, action=action)
        pass


def calculate_upper_confidence_bound_parent_visit(self,
                                                  node_visits,
                                                  parent_visits):

    ucbt = self.exploration_c * np.sqrt(np.divide(np.log1p(parent_visits), (node_visits + 1)))
    return ucbt


class MontecarloTreeSearch:
    def __init__(self):
        pass