import math

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState
from rl_agent.rl_agent import MonteCarloTreeSearchAgent


class MonteCarloTreeNode:
    def __init__(self):
        self.visits = 0
        self.value = 0


class MonteCarloTreeSearch:
    pass


def _calculate_upper_confidence_bound_node_value(node: MonteCarloTreeNode, parent_node: MonteCarloTreeNode, exploration_c: float):
    node_v = node.value/node.visits
    ucbt = exploration_c * (math.log(node.visits) / parent_node.visits) ** 0.5
    """
    the powerpoint adds 1 to parent_node.visits abowe i assume this is to avoid div/0 errors 
    """
    return node_v+ucbt


def _mc_rollout(state: BaseState, environment: BaseEnvironment, agent: MonteCarloTreeSearchAgent, node_values: {BaseState: MonteCarloTreeNode}):
    """
    Preformes a MC rollout from the provided node with the provided agent
    :param state:
    :param environment:
    :param agent:
    :param node_values:
    :return:
    """

    action = agent.pick_action(environment=environment, state=state)
    next_state, reward, done = environment.act(state,action)

    node = node_values.get(next_state)
    if node is None:
        node = MonteCarloTreeNode()
        node_values[next_state] = node
    node.visits += 1

    backprop_value = reward if done else _mc_rollout(next_state, environment, agent, node_values)
