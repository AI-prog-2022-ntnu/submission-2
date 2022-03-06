import math
# import multiprocessing
import queue
import random
import copy

import numpy as np

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, GameBaseState

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Queue
from torch import multiprocessing as mp

# executor = ProcessPoolExecutor()


# class MonteCarloTreeNode:
#     def __init__(self):
#         self.visits = 0
#         self.value = 0
from rl_agent.util import EGreedy


class MonteCarloTreeNode:
    def __init__(self):
        self.visits = 0
        self.value = 0


class UpdateDataUnit:
    def __init__(self):
        tree_search = []
        rollout = []
        final_val = None


class _SearchNode:
    def __init__(self,
                 state: GameBaseState,
                 node_hash,
                 reward=0,
                 has_terminal_child=False,
                 parent_hash=None,
                 action_from_parent=None,
                 has_been_rolled_out=False,
                 has_been_expanded=False,
                 terminal=False):
        self.has_terminal_child = has_terminal_child
        self.reward = reward
        self.node_hash = node_hash
        self.parent_hash = parent_hash
        self.terminal = terminal
        self.action_from_parent = action_from_parent
        self.has_been_expanded = has_been_expanded
        self.has_been_rolled_out = has_been_rolled_out
        self.children = []
        self.state = state


class RapidMaps:

    def __init__(self,
                 zeroed_board):

        self.rapid_win_map = copy.deepcopy(zeroed_board)
        self.rapid_lose_map = copy.deepcopy(zeroed_board)

        self.rapid_move_count_map = copy.deepcopy(zeroed_board)

    def update_maps(self,
                    map,
                    value):
        if value == 1:
            target_value = self.rapid_win_map
        elif value == -1:
            target_value = self.rapid_lose_map

        for n, row in enumerate(map):
            for i, col_v in enumerate(row):
                self.rapid_move_count_map[n][i] += 1
                target_value[n][i] += 1


def _parallel_mc_rollout(
        state: GameBaseState,
        agent,
        environment: BaseEnvironment,
        update_list: [],
        e_greedy: EGreedy,
):
    if e_greedy.should_pick_greedy(increment_round=True):
        action = agent.pick_action(state=state)
    else:
        action = random.choice(environment.get_valid_actions(state))
    try:
        next_state, reward, done = environment.act(state, action, inplace=True)
    except Exception:
        print(environment.get_valid_actions(state))
        print(action)
        raise Exception("\n\nWAAAAAAAA\n\n")

    if done:
        update_list.insert(0, reward)

        # TODO: god please save me this is horrible
        update_list.insert(0, state)

    else:
        _parallel_mc_rollout(next_state, agent, environment, update_list, e_greedy)

    update_list.append(hash(state))


def parallel_rollout(agent,
                     env,
                     in_que: Queue,
                     out_que: Queue,
                     e_greedy: EGreedy):
    run = True
    while run:
        msg = in_que.get()

        if msg is not None:

            root_state, update_list = msg
            # update_list = []
            _parallel_mc_rollout(root_state, agent, env, update_list, e_greedy)

            v = update_list.pop(0)
            update_list.append(v)
            out_que.put(update_list)
            e_greedy.reset()
        else:
            run = False


# this is super tight copled but practical
class TreePolicy:
    def __init__(self,
                 exploration_c):

        self.exploration_c = exploration_c
        self.exploration_b = exploration_c

    def calculate_upper_confidence_bound_node_value(self,
                                                    node,
                                                    parent_node):
        # if search_node.node.visits == 0:
        #     # how best solve problems -> pretend they are not there
        #     return 0

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

    def _rapid_b(self,
                 num,
                 num_won_with_n):
        return (num_won_with_n) / (1 + num + num_won_with_n + (4 * self.exploration_b * num * num_won_with_n))

    def calculate_RAPID_value(self,
                              search_node_parent: _SearchNode,
                              search_node: _SearchNode,
                              node: MonteCarloTreeNode,
                              parent_node: MonteCarloTreeNode,
                              rapid_maps: RapidMaps):
        action = search_node.action_from_parent
        if search_node_parent.state.current_player_turn() == 0:
            num_won_with_move = rapid_maps.rapid_win_map[action[0]][action[1]]
        else:
            num_won_with_move = rapid_maps.rapid_lose_map[action[0]][action[1]]
        num_with_move = rapid_maps.rapid_move_count_map[action[0]][action[1]]
        l1 = (1 - self._rapid_b(node.visits, num_with_move)) * (num_won_with_move / (node.visits + 1))
        l2 = (self._rapid_b(node.visits, num_with_move)) * (num_won_with_move / (node.visits + 1))
        l3 = self.exploration_c * np.sqrt(np.divide(np.log1p(parent_node.visits), (node.visits + 1)))

        return l1 + l2 + l3

    def get_first_terminal_child(self,
                                 search_node):

        if search_node.has_been_expanded:
            for c in search_node.children:
                if c.terminal:
                    return c
        return None

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

        self.tree_policy = TreePolicy(exploration_c)

        mp_context = mp.get_context('fork')

        self.to_workers_message_que = mp_context.Queue()
        self.from_worker_message_que = mp_context.Queue()

        # DO NOT REMOVE
        # mp.set_start_method("spawn")

        e_greedy = EGreedy(init_val=0.5, min_val=0.01, rounds_to_min=10)
        for _ in range(worker_thread_count):
            p = mp_context.Process(target=parallel_rollout, args=(self.agent, self.environment, self.to_workers_message_que, self.from_worker_message_que, e_greedy))
            p.start()

        self.active_p_semaphore = 0

        self.node_map = {}

        # TODO: FIX HARDCODE
        zeroed_board = environment.get_initial_state().hex_board
        self.rapid_map = RapidMaps(zeroed_board)

        pass

    ################################
    #          Utilitys            #
    ################################

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

    def _paralell_rollout(self,
                          child,
                          update_list):
        self.active_p_semaphore += 1
        self.to_workers_message_que.put((child.state, update_list))

    ################################
    #       Node management        #
    ################################

    def _apply_node_change_list(self,
                                change_list):
        prop_val = change_list[0]
        last_state = change_list.pop()

        # TODO: hardcode
        # print(change_list)
        self.rapid_map.update_maps(last_state.hex_board, prop_val)

        for node_hash in change_list[1:]:
            node = self._get_node_by_hash(node_hash)
            # s_node: _SearchNode = self.search_node_map.get(node_hash)
            # if s_node is not None:
            #     s_node.has_been_rolled_out = True

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

    ################################
    #        Tree navigation       #
    ################################

    def _create_node(self,
                     from_node: _SearchNode,
                     action):
        new_s, r, done = self.environment.act(from_node.state, action)

        new_search_node = _SearchNode(
            state=new_s,
            node_hash=hash(new_s),
            parent_hash=from_node.node_hash,
            action_from_parent=action,
            terminal=done,
            reward=r,
            has_been_rolled_out=done,
            has_been_expanded=done
        )

        self.search_node_map[hash(new_s)] = new_search_node
        from_node.children.append(new_search_node)

        return new_search_node

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

            if done:
                search_node.has_terminal_child = True

            self.search_node_map[hash(new_s)] = new_search_node
            search_nodes.append(new_search_node)
            child_states.append(new_s)

        search_node.children = search_nodes

    def _tree_search_old(self,
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

    def _tree_search(self,
                     parent: _SearchNode,
                     update_list: [],
                     d=0):
        d += 1

        # check if any of the children are terminal if so pick them
        terminal_child = self.tree_policy.get_first_terminal_child(parent)
        if terminal_child is not None:
            update_list.insert(0, terminal_child.reward)
            update_list.append(terminal_child.node_hash)
            update_list.append(parent.state)
            self._apply_node_change_list(change_list=update_list)
            return

        # else use the default tree policy to pick the child and add it to the update list
        child: _SearchNode = self.pick_child(parent)
        update_list.append(child.node_hash)

        # if the child has been rolled out we continue down the tree
        if child.has_been_rolled_out:
            # if self._get_node_by_hash(child.node_hash).visits > 0:
            if not child.has_been_expanded:
                self._expand_node(child)

            self._tree_search(parent=child, update_list=update_list, d=d)
        else:
            # if the child has not been rolled out we roll it out
            self._paralell_rollout(child, update_list)

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
        # if random.random() > 0.9:  # TODO: sadfj;sadfsad;fldjsfaljkif not child.has_been_expanded:
        #     best_child = random.choice(search_node.children)
        # else:
        parent_node = self._get_node_by_hash(search_node.node_hash)
        for child in search_node.children:
            if child.has_terminal_child:
                # if the next move after can be terminal, means the opponent can win.
                continue

            node = self._get_node_by_hash(child.node_hash)
            confidence_bound = self.tree_policy.calculate_upper_confidence_bound_node_value(node, parent_node)
            # confidence_bound = self.tree_policy.calculate_RAPID_value(
            #     search_node_parent=search_node,
            #     search_node=child,
            #     node=node,
            #     parent_node=parent_node,
            #     rapid_maps=self.rapid_map
            # )
            # print(confidence_bound)
            node = self._get_node_by_hash(child.node_hash)
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

    ################################
    #           Public             #
    ################################

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
                self._tree_search(parent=root_s_node, update_list=[root_s_node.node_hash])
                # for k, v in self.node_map.items():
                #     print(f"{v.value} visits to node {k.get_as_vec()}")
                # print("node_visits: ", [a.value for a in self.node_map.values()])

            if self.debug:
                print(f"parent visits: {root_node.visits}, value: {root_node.value}")
                for c in root_s_node.children:
                    node = self._get_node_by_hash(c.node_hash)
                    rapid_v = self.tree_policy.calculate_RAPID_value(
                        search_node_parent=root_s_node,
                        search_node=c,
                        node=node,
                        parent_node=root_node,
                        rapid_maps=self.rapid_map
                    )
                    print(f"action {c.action_from_parent} -> {c.state.get_as_vec()} node has {node.visits} visits an {node.value} value -> value {node.value / (node.visits + 1)} + {self.tree_policy.calculate_upper_confidence_bound_node_value(node, root_node)}")

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
