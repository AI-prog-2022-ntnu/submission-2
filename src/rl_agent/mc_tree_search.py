import math
import queue
import random
import copy

import numpy as np

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
                 node_hash,
                 reward=0,
                 parent_hash=None,
                 action_from_parent=None,
                 has_been_rolled_out=False,
                 has_been_expanded=False,
                 terminal=False):
        self.reward = reward
        self.node_hash = node_hash
        self.parent_hash = parent_hash
        self.terminal = terminal
        self.action_from_parent = action_from_parent
        self.has_been_expanded = has_been_expanded
        self.has_been_rolled_out = has_been_rolled_out
        self.children = []
        self.state = state


def _parallel_mc_rollout(
        state: GameBaseState,
        agent,
        environment,
        update_list: []):
    action = agent.pick_action(state=state)
    next_state, reward, done = environment.act(state, action, inplace=True)

    if done:
        update_list.insert(0, reward)
    else:
        _parallel_mc_rollout(next_state, agent, environment, update_list)

    update_list.append(hash(state))


def parallel_rollout(agent,
                     env,
                     in_que: Queue,
                     out_que: Queue):
    run = True
    while run:
        msg = in_que.get()

        if msg is not None:

            root_state, update_list = msg
            # update_list = []
            _parallel_mc_rollout(root_state, agent, env, update_list)
            out_que.put(update_list)
        else:
            run = False


# this is super tight copled but practical
class TreePolicy:
    def __init__(self,
                 exploration_c,
                 s_tree):

        self.s_tree = s_tree
        self.exploration_c = exploration_c

    def _calculate_upper_confidence_bound_node_value(self,
                                                     s_node: _SearchNode):
        # if search_node.node.visits == 0:
        #     # how best solve problems -> pretend they are not there
        #     return 0
        node = self.s_tree._get_node_by_hash(s_node.node_hash)
        parent_node = self.s_tree._get_node_by_hash(s_node.parent_hash)
        # node_v = node.value / (node.visits + 1)
        ucbt = self.exploration_c * np.sqrt(np.divide(np.log1p(parent_node.visits), (node.visits + 1)))
        # goodness = self.node_goodness(s_node)
        # td = goodness if goodness is not None else np.log1p(parent_node.visits)
        # ucbt = self.exploration_c * np.sqrt(np.divide(td, (node.visits + 1)))

        # if ucbt < 0:
        #     print("aaaa")
        """
        the powerpoint adds 1 to parent_node.visits abowe i assume this is to avoid div/0 errors 
        """
        return ucbt

    def _has_terminal_child(self,
                            search_node):

        if search_node.has_been_expanded:
            for c in search_node.children:
                if c.terminal:
                    return True
        return False

    def will_lose_next(self,
                       search_node: _SearchNode):
        if search_node.has_been_expanded:
            term = any(map(lambda x: x.terminal, search_node.children))
        else:
            term = False
        return term

    def node_goodness(self,
                      search_node: _SearchNode):
        # leads directly to victory #
        if search_node.terminal:
            return 100

        # guarantee defeat
        for child in search_node.children:
            # if child.has_been_expanded:
            #     print(child.has_been_expanded)
            if self._has_terminal_child(child):
                print("\n\naaaaaaaaaaaaaa")
                return -100

    def pick_child(self,
                   search_node: _SearchNode):
        player_0_turn = search_node.state.current_player_turn() == 0
        found = False

        # guaranteed_win_cand = self.get_guaranteed_win(search_node, player_0_turn)
        # if guaranteed_win_cand is not None:
        #     return guaranteed_win_cand

        unsutable = []
        best_child = random.choice(search_node.children)

        if player_0_turn:
            best_child_value = 0
        else:
            best_child_value = float("inf")

        # while not found:
        ## find candidate ##
        if random.random() > 0.9:  # TODO: sadfj;sadfsad;fldjsfaljkif not child.has_been_expanded:
            best_child = random.choice(search_node.children)
        else:
            for child in search_node.children:
                confidence_bound = self._calculate_upper_confidence_bound_node_value(child)
                node = self.s_tree._get_node_by_hash(child.node_hash)
                if player_0_turn:
                    val = (node.value / (node.visits + 1)) + confidence_bound
                    if val > best_child_value:
                        best_child_value = val
                        best_child = child
                else:
                    val = (node.value / (node.visits + 1)) - confidence_bound
                    if val < best_child_value:
                        best_child_value = val
                        best_child = child

        return best_child


class MontecarloTreeSearch:
    def __init__(self,
                 exploration_c,
                 environment: BaseEnvironment,
                 agent,
                 worker_thread_count: int):
        self.worker_thread_count = worker_thread_count
        self.agent = agent
        self.environment = environment
        self.exploration_c = exploration_c

        self.search_node_map = {}
        self.debug = False

        self.tree_policy = TreePolicy(exploration_c, self)

        self.to_workers_message_que = Queue()
        self.from_worker_message_que = Queue()

        for _ in range(worker_thread_count):
            p = Process(target=parallel_rollout, args=(self.agent, self.environment, self.to_workers_message_que, self.from_worker_message_que))
            p.start()

        self.active_p_semaphore = 0

        self.node_map = {}
        pass

    def _apply_node_change_list(self,
                                change_list):
        prop_val = change_list[0]
        for node_hash in change_list[1:]:
            node = self._get_node_by_hash(node_hash)
            s_node: _SearchNode = self.search_node_map.get(node_hash)
            if s_node is not None:
                s_node.has_been_rolled_out = True

            node.visits += 1
            node.value += prop_val

    def _get_node_by_state(self,
                           state: GameBaseState) -> MonteCarloTreeNode:
        node = self.node_map.get(hash(state))
        if node is None:
            node = MonteCarloTreeNode()
            self.node_map[hash(state)] = node
        return node

    def _get_node_by_hash(self,
                          node_hash) -> MonteCarloTreeNode:
        node = self.node_map.get(node_hash)
        if node is None:
            node = MonteCarloTreeNode()
            self.node_map[node_hash] = node
        return node

    def _concurrency_tickler(self):
        value = None
        if self.active_p_semaphore < self.worker_thread_count + 5:
            # if there are more slots left try to fetch and continue if no one is free
            # TODO: mabye loop this to fetch untill the que is empty
            try:
                value = self.from_worker_message_que.get_nowait()
            except queue.Empty as e:
                pass
        else:
            # if all the slots are full wait untill a worker is done
            value = self.from_worker_message_que.get()

        if value is not None:
            # if we got a value from the que apply it to the current tree
            self.active_p_semaphore -= 1
            self._apply_node_change_list(value)

    def _expand_node(self,
                     search_node: _SearchNode):
        # may be optimized by saving the state action edges to avoid unnecessarily initializing new memory for new states already found
        possible_actions = self.environment.get_valid_actions(search_node.state)
        child_states = []

        search_nodes = []
        for action in possible_actions:
            new_s, r, done = self.environment.act(search_node.state, action)

            new_search_node = _SearchNode(
                state=new_s,
                node_hash=hash(new_s),
                parent_hash=search_node.node_hash,
                action_from_parent=action,
                terminal=done,
                reward=r,
                has_been_rolled_out=done,
                has_been_expanded=done
            )

            self.search_node_map[hash(new_s)] = new_search_node
            search_nodes.append(new_search_node)
            child_states.append(new_s)

        search_node.children = search_nodes

    def _paralell_rollout(self,
                          child,
                          update_list):

        self.active_p_semaphore += 1
        self.to_workers_message_que.put((child.state, update_list))

    def _tree_search(self,
                     parent: _SearchNode,
                     update_list: [],
                     d=0):

        if self.debug:
            print("tree search depth at: ", d)
            d += 1

        if parent.terminal:
            update_list.insert(0, parent.reward)
            update_list.append(parent.node_hash)
            self._apply_node_change_list(change_list=update_list)
        else:

            child: _SearchNode = self.tree_policy.pick_child(parent)
            update_list.append(child.node_hash)

            if child.has_been_rolled_out and random.random() < 0.5:
                if not child.has_been_expanded:
                    self._expand_node(child)

                self._tree_search(parent=child, update_list=update_list, d=d)
            else:
                # update_list = self._paralell_rollout(child)
                if child.terminal:
                    update_list.insert(0, child.reward)
                    update_list.append(child.node_hash)
                    self._apply_node_change_list(change_list=update_list)
                else:
                    self._paralell_rollout(child, update_list)

    def close_helper_threads(self):
        for _ in range(self.worker_thread_count * 2):
            self.to_workers_message_que.put(None)

    def mc_tree_search(self,
                       num_rollouts,
                       root_state: GameBaseState):
        root_s_node: _SearchNode = self.search_node_map.get(hash(root_state))
        root_node = self._get_node_by_state(root_state)

        if root_s_node is None:
            root_s_node = _SearchNode(
                state=root_state,
                node_hash=hash(root_state),
                has_been_rolled_out=False,
                has_been_expanded=False
            )

        if not root_s_node.has_been_expanded:
            self._expand_node(root_s_node)

        # check if possible to end
        winning_c = None
        for c in root_s_node.children:
            if c.terminal:
                winning_c = c
                break

        if winning_c is not None:
            ret = {}  # ehhhh

            for child in root_s_node.children:
                child_s_node: _SearchNode = child
                if winning_c.node_hash == child_s_node.node_hash:
                    ret[child.action_from_parent] = 1
                else:
                    ret[child.action_from_parent] = 0

            return ret, None
        else:
            for i in range(num_rollouts):
                self._concurrency_tickler()
                if self.debug:
                    print(f"Tree search for node {i} started")
                self._tree_search(parent=root_s_node, update_list=[root_s_node.node_hash])
                # for k, v in self.node_map.items():
                #     print(f"{v.value} visits to node {k.get_as_vec()}")
                # print("node_visits: ", [a.value for a in self.node_map.values()])

            if False:
                print(f"parent visits: {root_node.visits}, value: {root_node.value}")
                for c in root_s_node.children:
                    node = self._get_node_by_hash(c.node_hash)
                    print(f"action {c.action_from_parent} -> {c.state.get_as_vec()} node has {node.visits} visits an {node.value} value -> value {node.value / (node.visits + 1)} + {self.tree_policy._calculate_upper_confidence_bound_node_value(c)}")

            while self.active_p_semaphore > 0:
                change_list = self.from_worker_message_que.get()
                self._apply_node_change_list(change_list)
                self.active_p_semaphore -= 1
            ret = {}  # ehhhh
            ret_2_electric_bogaloo = {}  # ehhhh

            for child in root_s_node.children:
                child_s_node: _SearchNode = child
                node = self._get_node_by_hash(child_s_node.node_hash)
                ret[child.action_from_parent] = node.visits
                ret_2_electric_bogaloo[child.state] = (node.value / node.visits)

            return ret, ret_2_electric_bogaloo


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
