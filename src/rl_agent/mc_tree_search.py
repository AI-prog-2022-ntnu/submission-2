import math
import random
import copy

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, GameBaseState

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import JoinableQueue, Queue, Process


# executor = ProcessPoolExecutor()


# class MonteCarloTreeNode:
#     def __init__(self):
#         self.visits = 0
#         self.value = 0

class MonteCarloTreeNode:
    def __init__(self):
        self.visits = 0
        self.value = 0


class _SearchNode:
    def __init__(self,
                 state: GameBaseState,
                 node: MonteCarloTreeNode,
                 parent_node: MonteCarloTreeNode = None,
                 action_from_parent=None,
                 has_been_rolled_out=False,
                 has_been_expanded=False,
                 terminal=False):
        self.terminal = terminal
        self.action_from_parent = action_from_parent
        self.has_been_expanded = has_been_expanded
        self.has_been_rolled_out = has_been_rolled_out
        self.children = []
        self.parent_node = parent_node
        self.node = node
        self.state = state


def _parallel_mc_rollout(
        state: GameBaseState,
        agent,
        environment,
        update_list: []):
    action = agent.pick_action(state=state)
    next_state, reward, done = environment.act(state, action, inplace=True)

    if done:
        update_list.append(1)
    else:
        _parallel_mc_rollout(next_state, agent, environment, update_list)

    update_list.append(hash(state))


def parallel_rollout(agent,
                     env,
                     in_que: Queue,
                     out_que: Queue):
    run = True
    while run:
        root_state = in_que.get()

        if root_state is not None:
            update_list = []
            _parallel_mc_rollout(root_state, agent, env, update_list)
            out_que.put(update_list)
        else:
            run = False


class MontecarloTreeSearch:
    def __init__(self,
                 exploration_c,
                 environment: BaseEnvironment,
                 agent):
        self.agent = agent
        self.environment = environment
        self.exploration_c = exploration_c
        self.node_map = {}
        self.search_node_map = {}
        self.debug = False

        self.test_to_thread_q = Queue()
        self.test_from_thread_q = Queue()
        p = Process(target=parallel_rollout, args=(self.agent, self.environment, self.test_to_thread_q, self.test_from_thread_q))
        p.start()

    def _update_from_workers(self):
        self.test_from_thread_q.
    def _get_node(self,
                  state: GameBaseState) -> MonteCarloTreeNode:
        node = self.node_map.get(hash(state))
        if node is None:
            node = MonteCarloTreeNode()
            self.node_map[hash(state)] = node
        return node

    def _calculate_upper_confidence_bound_node_value(self,
                                                     search_node: _SearchNode):
        # if search_node.node.visits == 0:
        #     # how best solve problems -> pretend they are not there
        #     return 0
        node_v = search_node.node.value / (search_node.node.visits + 1)
        ucbt = self.exploration_c * (math.log1p(search_node.node.visits) / (search_node.parent_node.visits + 1)) ** 0.5
        """
        the powerpoint adds 1 to parent_node.visits abowe i assume this is to avoid div/0 errors 
        """
        return node_v + ucbt

    # TODO: maybe multithread this
    def _mc_rollout(self,
                    state: GameBaseState):
        action = self.agent.pick_action(state=state)
        # print(state.get_as_vec())
        # print(action)
        next_state, reward, done = self.environment.act(state, action, inplace=True)
        node = self._get_node(next_state)
        # self.environment.display_state(next_state)
        # print(reward, done)
        # print(next_state.get_as_vec())
        # exit()

        if done:

            visits, value = 1, 1
        else:
            visits, value = self._mc_rollout(next_state)

        node.visits += visits
        node.value += value
        return visits, value

    def _expand_node(self,
                     search_node: _SearchNode):
        # may be optimized by saving the state action edges to avoid unnecessarily initializing new memory for new states already found
        possible_actions = self.environment.get_valid_actions(search_node.state)
        child_states = []

        search_nodes = []
        for action in possible_actions:
            new_s, r, done = self.environment.act(search_node.state, action)

            node = self._get_node(new_s)

            new_search_node = _SearchNode(
                state=new_s,
                node=node,
                parent_node=search_node.node,
                action_from_parent=action,
                terminal=done,
                has_been_rolled_out=done,
                has_been_expanded=done
            )
            self.search_node_map[hash(new_s)] = new_search_node
            search_nodes.append(new_search_node)
            child_states.append(new_s)

        search_node.children = search_nodes

    def _policy_pick_child(self,
                           search_node: _SearchNode):
        player_0_turn = search_node.state.current_player_turn() == 0
        if player_0_turn:
            best_child = search_node.children[0]
            best_child_value = 0
        else:
            best_child = search_node.children[0]
            best_child_value = float("inf")

        # print(search_node.children)
        if random.random() > 0.1:  # TODO: sadfj;sadfsad;fldjsfaljk
            best_child = random.choice(search_node.children)
        else:
            for child in search_node.children:
                confidence_bound = self._calculate_upper_confidence_bound_node_value(child)
                if player_0_turn:
                    # val = child.parent_current_q_val + confidence_bound
                    val = (child.node.value / (child.node.visits + 1)) + confidence_bound
                    if val > best_child_value:
                        best_child_value = val
                        best_child = child
                else:
                    val = (child.node.value / (child.node.visits + 1)) - confidence_bound
                    # val = child.parent_current_q_val - confidence_bound
                    if val < best_child_value:
                        best_child_value = val
                        best_child = child
        return best_child

    def _paralell_rollout(self,
                          child):
        # future = executor.submit(parallel_rollout, child.state, self.agent, self.environment)
        # value = future.result()
        self.test_to_thread_q.put(child.state)
        value = self.test_from_thread_q.get()

        prop_val = value[0]
        for node_hash in value[1:]:
            node = self.node_map.get(node_hash)
            if node is None:
                node = MonteCarloTreeNode()
                self.node_map[node_hash] = node

            node.visits += 1
            node.value += prop_val

        return prop_val

    def _tree_search(self,
                     parent: _SearchNode,
                     d=0):
        if parent.terminal:
            return 0, 0
        if self.debug:
            print("tree search depth at: ", d)
            d += 1

        child: _SearchNode = self._policy_pick_child(parent)

        # print(child)
        # print(child.has_been_rolled_out)
        # print(child.has_been_expanded)
        # print()
        if child.has_been_rolled_out:
            if not child.has_been_expanded:
                self._expand_node(child)
            visits, value = self._tree_search(child, d)
        else:
            value = self._paralell_rollout(child)
            visits = 1
            # visits, value = self._mc_rollout(copy.deepcopy(child.state))
            child.node.visits += visits
            child.node.value += value

        parent.node.visits += visits
        parent.node.value += value
        return visits, value

    def close_helper_threads(self):
        self.test_to_thread_q.put(None)

    def mc_tree_search(self,
                       num_rollouts,
                       root_state: GameBaseState):
        root_s_node: _SearchNode = self.search_node_map.get(hash(root_state))
        root_node = self._get_node(root_state)

        if root_s_node is None:
            root_s_node = _SearchNode(
                state=root_state,
                node=root_node,
                has_been_rolled_out=False,
                has_been_expanded=False
            )

        if not root_s_node.has_been_expanded:
            self._expand_node(root_s_node)

        for i in range(num_rollouts):
            if self.debug:
                print(f"Tree search for node {i} started")
            self._tree_search(root_s_node)
            # for k, v in self.node_map.items():
            #     print(f"{v.value} visits to node {k.get_as_vec()}")
            # print("node_visits: ", [a.value for a in self.node_map.values()])

        ret = {}  # ehhhh
        for child in root_s_node.children:
            child: _SearchNode = child
            ret[child.action_from_parent] = child.node.visits

        return ret


# def _get_child_nodes(from_state: BaseState,
#                      environment: BaseEnvironment):
#     pass
#
#
# def _tree_policy_pick(
#         search_nodes: [_SearchNode],
#         node_values: {BaseState: MonteCarloTreeNode},
#         exp_c):
#     best_node: _SearchNode = search_nodes[0]
#     best_node_value = best_node.q_val + _calculate_upper_confidence_bound_node_value(best_node, exp_c)
#
#     while True:
#         for s_node in search_nodes:
#             value = s_node.q_val + _calculate_upper_confidence_bound_node_value(s_node, exp_c)
#             if best_node_value < value:
#                 best_node_value = value
#                 best_node = s_node
#         if best_node


"""

select-> expand -> sim -> backup

tror bare man skal traversjere ned og expande/rulle ut nodene as you go


Selection:
1. select some node
2. if node is:
    - already rolled out -> expand
    - not rolled out -> roll out

Expansion:
if any of the expanded nodes have been part of a rollout mark the node as already rolled out

Simulation:


Backup

"""

# def _tree_search(state: BaseState,
#                  environment: BaseEnvironment,
#                  agent: MonteCarloTreeSearchAgent,
#                  node_values: {BaseState: MonteCarloTreeNode},
#                  num_to_expand,
#                  expanded: [BaseState],
#                  exp_c):
#     possible_actions = environment.get_valid_actions(state)
#
#     # TODO: Highly memory and cpu inefficent, optimize
#     vals = []
#     fetch = []
#     parent_node = node_values.get(state)
#
#     for action in possible_actions:
#         new_s, r, done = environment.act(state, action)
#         c = cache[new_s]
#         node = node_values.get(new_s)
#         if node is None:
#             node = MonteCarloTreeNode()
#             node_values[new_s] = node
#
#         ucf = _calculate_upper_confidence_bound_node_value(node, parent_node, exploration_c=exp_c)
#         if c is None:
#             pass
#         else:
#             vals.append((c + ucf, new_s))
#
#
# def mc_tree_search(state: BaseState,
#                    environment: BaseEnvironment,
#                    agent: MonteCarloTreeSearchAgent,
#                    num_to_expand: int):
#     node_values = {}
#     root_node = MonteCarloTreeNode()
#     node_values[state] = root_node
#     val = _tree_search(state, environment, agent, node_values)
#
#     pass
