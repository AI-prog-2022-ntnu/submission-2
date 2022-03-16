import math
# import multiprocessing
import queue
import random
import copy
import time

import numpy as np
import torch

from enviorments.base_environment import BaseEnvironment, BoardGameEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Queue
from torch import multiprocessing as mp

# executor = ProcessPoolExecutor()
from rl_agent.critic import Critic
from rl_agent.util import EGreedy


class NodeValueCounter:
    def __init__(self):
        self.visits = 0
        self.value = 0
        self.discounted_value = 0

    @staticmethod
    def get_node_hash(
            parent_state_hash,
            action_from_parent
    ):
        return parent_state_hash + hash(action_from_parent)


class StateValuePropegator:
    def __init__(self):
        self.tree_search_node_state_hashes = []
        self.tree_search_nodes_has_visits_pre_propegated = False
        self.rollout_nodes_state_hashes = []
        self.value = 0


class _SearchNode:
    def __init__(self,
                 parent_s_node,
                 environment,
                 reward=0,
                 action_from_parent=None,
                 has_been_rolled_out=False,
                 has_been_expanded=False,
                 terminal=False,
                 initialized=False
                 ):
        self.initialized = initialized
        self.environment = environment
        self.parent_node_hash = parent_s_node.node_hash if parent_s_node is not None else None
        self._state = None

        if parent_s_node is None:
            self.node_hash = hash("-1")
        else:
            self.node_hash = NodeValueCounter.get_node_hash(hash(parent_s_node.state), action_from_parent)
        self.action_from_parent = action_from_parent
        self.has_been_expanded = has_been_expanded
        self.has_been_rolled_out = has_been_rolled_out

        self.children = {}

        self.node_hash = None
        # lazyily evaluated
        self.terminal = terminal
        self.reward = reward

        self.parent_s_node = parent_s_node

    @property
    def state(self):
        return self._get_node_state()

    @state.setter
    def state(self,
              value):
        self._state = value

    def _get_node_state(self):
        state = self._state
        if state is not None:
            return state
        else:
            self._state = self._lazy_fetch_state()
            return self._state

    def _lazy_fetch_state(self):
        parent_s = self.parent_s_node._get_node_state()
        node_s, r, done = self.environment.act(parent_s, self.action_from_parent)
        self.initialized = True
        self.state = node_s
        self.terminal = done
        self.reward = r
        return self.state


def probe_n(
        state: BoardGameBaseState,
        environment: BoardGameEnvironment,
        n
):
    possible_moves = environment.get_valid_actions(state)
    hit, val = 0, 0
    n -= 1

    for action in possible_moves:
        # print(action)
        n_state, reward, done = environment.act(state, action, inplace=True)
        if done:
            val += reward
            hit += 1
        else:
            if n > 0:
                probe_h, probe_v = probe_n(n_state, environment, n)
                hit += probe_h
                val += probe_v

        environment.reverse_move(n_state, action)

    return hit, val


def min_max_search(
        state: BoardGameBaseState,
        environment: BaseEnvironment,
        current_max: bool,
        d=0
):
    if d < 4:
        print("chk ", d)
    nd = d + 1
    possible_moves = environment.get_valid_actions(state)
    for action in possible_moves:
        # print(action)
        n_state, reward, done = environment.act(state, action, inplace=True)
        if done:
            environment.reverse_move(n_state, action)
            return reward
        else:
            current_max ^= True
            r = min_max_search(n_state, environment, current_max, nd)

        if current_max and r == 1:
            environment.reverse_move(n_state, action)
            return r
        elif not current_max and r == -1:
            environment.reverse_move(n_state, action)
            return r

        environment.reverse_move(n_state, action)

    return r


def _parallel_mc_rollout(
        state: BoardGameBaseState,
        agent,
        environment: BoardGameEnvironment,
        value_prop: StateValuePropegator,
        e_greedy: EGreedy,
        use_prob: bool = False
):
    state_hash = hash(state)
    winning_move = environment.get_state_winning_move(state)

    if winning_move is not None:
        # this may be optimized by combining it with the traversal has won chehk
        action = winning_move
    elif e_greedy.should_pick_greedy(increment_round=True):
        action = agent.pick_action(state=state, use_prob_not_max=use_prob)
    else:
        action = random.choice(environment.get_valid_actions(state))
    try:
        next_state, reward, done = environment.act(state, action, inplace=True)
    except Exception:
        print(environment.get_valid_actions(state))
        print(action)
        raise Exception("\n\nWAAAAAAAA\n\n")

    if done:
        value_prop.value = reward
    else:
        _parallel_mc_rollout(next_state, agent, environment, value_prop, e_greedy, use_prob)

    value_prop.rollout_nodes_state_hashes.append(NodeValueCounter.get_node_hash(state_hash, action))


def parallel_rollout(agent,
                     env,
                     in_que: Queue,
                     out_que: Queue,
                     e_greedy: EGreedy,
                     use_prob_not_max_action: bool):
    run = True
    timeout_ms = 1000
    while run:
        # s_time = time.monotonic_ns()
        msg = in_que.get(timeout=timeout_ms)
        # s2_time = time.monotonic_ns()
        if msg is not None:
            root_state, value_prop, use_critic_prob = msg

            if random.random() < use_critic_prob:
                value = agent.critic.get_state_value(root_state)[0]
                value_prop.value = value
            else:
                _parallel_mc_rollout(root_state, agent, env, value_prop, e_greedy, use_prob=use_prob_not_max_action)

            out_que.put(value_prop)
            e_greedy.reset()
        else:
            run = False

        # s3_time = time.monotonic_ns()
        # print(f"wait time {s2_time - s_time}ns run time search {s3_time - s2_time}")


def calculate_upper_confidence_bound_node_value(exploration_c,
                                                node_visits,
                                                parent_visits):
    ucbt = exploration_c * np.sqrt(np.divide(np.log1p(parent_visits), (node_visits + 1)))

    return ucbt


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
        self.value_discount = 0.9

        self.debug = False

        mp_context = mp.get_context('fork')

        self.to_workers_message_que = mp_context.Queue()
        self.from_worker_message_que = mp_context.Queue()

        # DO NOT REMOVE
        # mp.set_start_method("spawn")
        self.e_greedy: EGreedy = None

        self.agent.model.share_memory()
        e_greedy = EGreedy(init_val=0.8, min_val=0.01, rounds_to_min=3)
        # e_greedy = EGreedy(init_val=0, min_val=0.0, rounds_to_min=10)
        for _ in range(worker_thread_count):
            use_prob = random.random() > 0.5
            p = mp_context.Process(target=parallel_rollout, args=(self.agent, self.environment, self.to_workers_message_que, self.from_worker_message_que, e_greedy, use_prob))
            p.start()

        self.active_p_semaphore = 0

        self.node_map = {}

        self.critic_eval_frac = 0

    ################################
    #          Utilitys            #
    ################################

    def _calculate_critic_eval_frac(self):
        critic: Critic = self.agent.critic

        if len(critic.latest_set_loss_list) < 5:
            for _ in range(10):
                critic.latest_set_loss_list.insert(0, 100)

        last_5_loss_avg = sum(critic.latest_set_loss_list[-5:]) / 5
        if last_5_loss_avg < 0.10:
            critic_prob = max((1 - last_5_loss_avg), 0) / 4
        else:
            critic_prob = 0

        return critic_prob

    def _concurrency_tickler(self):
        value = None
        if self.active_p_semaphore < self.worker_thread_count + 5:
            # if there are more slots left try to fetch and continue if no one is free
            # TODO: mabye loop this to fetch untill the que is empty
            try:
                value: StateValuePropegator = self.from_worker_message_que.get_nowait()
            except queue.Empty as e:
                pass
        else:
            # if all the slots are full wait untill a worker is done
            value = self.from_worker_message_que.get()

        if value is not None:
            # if we got a value from the que apply it to the current tree
            self.active_p_semaphore -= 1
            self._apply_value_propegator(value)

    def _paralell_rollout(self,
                          child,
                          value_prop):
        self.active_p_semaphore += 1
        self.to_workers_message_que.put((child.state, value_prop, self.critic_eval_frac))

    ################################
    #       Node management        #
    ################################

    def _apply_value_propegator(self,
                                value_prop: StateValuePropegator):
        prop_val = value_prop.value
        discounted_prop_val = value_prop.value
        # TODO: URGENT figure out why the last and second element of the change list are equal
        # print(change_list)
        # change_list.pop(-1)
        for node_hash in reversed(value_prop.rollout_nodes_state_hashes):
            discounted_prop_val *= self.value_discount
            node = self._get_node_by_hash(node_hash)
            node.visits += 1
            node.value += prop_val
            node.discounted_value += prop_val

        for node_hash in reversed(value_prop.tree_search_node_state_hashes):
            discounted_prop_val *= self.value_discount
            node = self._get_node_by_hash(node_hash)

            node.value += prop_val
            node.discounted_value += prop_val
            if not value_prop.tree_search_nodes_has_visits_pre_propegated:
                node.visits += 1

    def _pre_apply_tree_search_node_visits(self,
                                           value_prop):
        if value_prop.tree_search_nodes_has_visits_pre_propegated:
            raise Exception("illigal state")

        value_prop.tree_search_nodes_has_visits_pre_propegated ^= True

        for node_hash in value_prop.tree_search_node_state_hashes:
            node = self._get_node_by_hash(node_hash)
            node.visits += 1

    def _get_node_by_hash(self,
                          node_hash) -> NodeValueCounter:
        node = self.node_map.get(node_hash)
        if node is None:
            node = NodeValueCounter()
            self.node_map[node_hash] = node
        return node

    ################################
    #        Tree navigation       #
    ################################

    def _create_lazy_node(self,
                          from_node: _SearchNode,
                          action):
        # new_s, r, done = self.environment.act(from_node.state, action)

        new_search_node = _SearchNode(
            parent_s_node=from_node,
            environment=self.environment,
            reward=0,
            action_from_parent=action,
            has_been_rolled_out=False,
            has_been_expanded=False,
            terminal=False
        )

        from_node.children[action] = new_search_node

        return new_search_node

    def _expand_node(self,
                     search_node: _SearchNode):
        # may be optimized by saving the state action edges to avoid unnecessarily initializing new memory for new states already found
        possible_actions = self.environment.get_valid_actions(search_node.state)
        child_states = []

        search_nodes = []
        for action in possible_actions:
            self._create_lazy_node(from_node=search_node, action=action)
        #     new_s, r, done = self.environment.act(search_node.state, action, inplace=False)
        #
        #     new_search_node = _SearchNode(
        #         state=new_s,
        #         node_hash=hash(new_s),
        #         parent_hash=search_node.node_hash,
        #         action_from_parent=action,
        #         terminal=done,
        #         reward=r,
        #         has_been_rolled_out=done,
        #         has_been_expanded=done
        #     )
        #
        #     if done:
        #         search_node.has_terminal_child = True
        #
        #     self.search_node_map[hash(new_s)] = new_search_node
        #     search_nodes.append(new_search_node)
        #     child_states.append(new_s)
        #
        # search_node.children = search_nodes
        search_node.has_been_expanded = True

    def _tree_search(self,
                     parent: _SearchNode,
                     value_prop: StateValuePropegator,
                     d=0):

        # # if the node is not expanded, expand it
        if not parent.has_been_expanded:
            self._expand_node(parent)

        # check if any of the nodes children are terminal, if they are terminate the search and apply value propegation for all the terminal states
        # if parent.has_terminal_child:
        #     propegators = []
        #     for child in parent.children:
        #         child_s_node: _SearchNode = child
        #         if child_s_node.terminal:
        #             prop_copy = copy.deepcopy(value_prop)
        #             prop_copy.value = child_s_node.reward
        #             prop_copy.tree_search_node_state_hashes.append(child.node_hash)
        #             propegators.append(prop_copy)
        #
        #     for prop in propegators:
        #         self._apply_value_propegator(prop)
        #     return

        # else use the default tree policy to pick the child and add it to the update list
        if self.e_greedy.should_pick_greedy(increment_round=True):
            child: _SearchNode = self.pick_child(parent)
        else:
            child: _SearchNode = random.choice(list(parent.children.values()))

        value_prop.tree_search_node_state_hashes.append(child.node_hash)

        # if the child has been rolled out we continue down the tree
        if child.has_been_rolled_out:
            # if self._get_node_by_hash(child.node_hash).visits > 0:
            if not child.has_been_expanded:
                self._expand_node(child)

            self._tree_search(parent=child, value_prop=value_prop, d=d)
        else:
            # if the child has not been rolled out we roll it out
            # self._paralell_rollout(child, update_list)
            if not child.initialized:
                _ = child.state
            self._pre_apply_tree_search_node_visits(value_prop)
            self._paralell_rollout(child, value_prop)

            # value = self.agent.critic.get_state_value(child.state)[0]
            # update_list.insert(0, value)
            # update_list.append(child.node_hash)
            # self._apply_node_change_list(change_list=update_list)

    def pick_child(self,
                   search_node: _SearchNode):
        player_0_turn = search_node.state.current_player_turn() == 0
        found = False

        # guaranteed_win_cand = self.get_guaranteed_win(search_node, player_0_turn)
        # if guaranteed_win_cand is not None:
        #     return guaranteed_win_cand

        unsutable = []
        best_child = random.choice(list(search_node.children.values()))

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
        for child in search_node.children.values():
            # if child.has_terminal_child:
            #     # if the next move after can be terminal, means the opponent can win.
            #     continue

            node = self._get_node_by_hash(child.node_hash)
            confidence_bound = calculate_upper_confidence_bound_node_value(self.exploration_c, node.visits, parent_node.visits)

            node = self._get_node_by_hash(child.node_hash)
            if player_0_turn:
                # val = ((node.p1_wins - node.p2_wins) / (node.visits + 1)) + confidence_bound
                # val = ((node.value + 1) / (node.visits + 1)) + confidence_bound
                val = ((node.discounted_value + 1) / (node.visits + 1)) + confidence_bound
                if val > best_child_value:
                    best_child_value = val
                    best_child = child
            else:
                # val = ((node.p1_wins - node.p2_wins) / (node.visits + 1)) - confidence_bound
                # val = ((node.value + 1) / (node.visits + 1)) - confidence_bound
                val = ((node.discounted_value + 1) / (node.visits + 1)) - confidence_bound
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
                       root_state: BoardGameBaseState):

        self.critic_eval_frac = 0  # self._calculate_critic_eval_frac()

        print(f"Using a critic eval frac of {self.critic_eval_frac * 100}%")

        self.e_greedy = EGreedy(
            init_val=0.8,
            min_val=0.0,
            rounds_to_min=20,  # TODO: FIX IMPORTANT!
        )

        root_s_node = _SearchNode(
            parent_s_node=None,
            environment=self.environment,
            reward=0,
            action_from_parent="-1",
            has_been_rolled_out=False,
            has_been_expanded=False,
            terminal=False
        )
        root_s_node.state = root_state

        if not root_s_node.has_been_expanded:
            self._expand_node(root_s_node)
        else:
            print(root_s_node.children)
            child: _SearchNode = root_s_node.children[self.environment.get_valid_actions(root_state)[0]]
            root_s_node.node_hash = child.parent_node_hash

        root_node = self._get_node_by_hash(root_s_node.node_hash)

        # reset the visit counts to avoid polluting the action selection

        # possible_actions = self.environment.get_valid_actions(root_s_node.state)
        # # print("parent ", hash(root_state))
        # for action in possible_actions:
        # for child in root_s_node.children:
        #     # child_s, r, done = self.environment.act(root_state, action)
        #
        #     c_node: NodeValueCounter = self._get_node_by_hash(hash(child.state))
        #     # print("child ", hash(child.state))
        #     # print(c_node.value)
        #     # print(c_node.visits)
        #     c_node.value = 0
        #     c_node.visits = 0

        # check if possible to end

        if True:

            wait_milli_sec = self.agent.ms_tree_search_time
            c_time = time.monotonic_ns()
            stop_t = c_time + (wait_milli_sec * 1000000)
            rnds = 0
            # for i in range(num_rollouts):

            while time.monotonic_ns() < stop_t:
                rnds += 1

                value_prop = StateValuePropegator()
                value_prop.tree_search_node_state_hashes.append(root_s_node.node_hash)

                self._concurrency_tickler()
                self._tree_search(parent=root_s_node, value_prop=value_prop)
                # for k, v in self.node_map.items():
                #     print(f"{v.value} visits to node {k.get_as_vec()}")
                # print("node_visits: ", [a.value for a in self.node_map.values()])

            while self.active_p_semaphore > 0:
                change_list = self.from_worker_message_que.get()
                self._apply_value_propegator(change_list)
                self.active_p_semaphore -= 1
            ret = {}  # ehhhh
            ret_2_electric_bogaloo = {}  # ehhhh

            max_v, max_a = 0, None
            v_sum = 0
            for child in root_s_node.children.values():
                child_s_node: _SearchNode = child
                node = self._get_node_by_hash(child_s_node.node_hash)
                ret[child.action_from_parent] = node.visits

                v_sum += node.visits
                if node.visits > max_v:
                    max_a = child_s_node.action_from_parent
                    max_v = node.visits

                # TODO: CONFIUGURE FROM ELSWHERE
                # if node.visits > 30:
                #     # val = node.p1_wins if child_s_node.state.current_player_turn() == 0 else node.p2_wins
                #     val = node.value
                #     v = (val / node.visits)
                #     ret_2_electric_bogaloo[child.state] = v

            ret_2_electric_bogaloo[root_s_node.state] = root_node.discounted_value / root_node.visits

            print(f"completed {rnds} rollouts in the {wait_milli_sec}ms limit")
            if self.debug:
                p_dists = self.agent.model.get_probability_distribution([root_s_node.state])[0]

                print(f"parent visits: {root_node.visits}, value: {root_node.value}")
                v_count_sum = 0
                for c in root_s_node.children.values():
                    node = self._get_node_by_hash(c.node_hash)
                    v_count_sum += node.visits

                    action_idx = self.environment.get_action_space().index((c.action_from_parent))
                    p_dist_val = p_dists[action_idx]
                    critic_q_val = self.agent.critic.get_state_value(c.state)[0]

                    # print(f"action {c.action_from_parent} -> {c.state.get_as_vec()} node has {node.visits} visits value {node.value} value ->  {node.value / (node.visits + 1)}")  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
                    print(
                        "action {} ->  node visits: {:<4} value: {:<9.3f} | agent| pred: {:<9.3} actual: {:<8.4} error: {:<9.4}  |critic Q(s,a)(with discount)| pred: {:<9.3} actual: {:<8.4}  error: {:<8.4} ".format(
                            c.action_from_parent,
                            # c.state.get_as_vec(),
                            node.visits,
                            float(node.value),
                            p_dist_val,
                            node.visits / v_sum,
                            p_dist_val - ((node.visits + 1) / (v_sum + 1)),
                            critic_q_val,
                            (node.discounted_value + 1) / (node.visits + 1),
                            critic_q_val - ((node.discounted_value + 1) / (node.visits + 1))
                        ))  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
                print(f"v count sum: {v_count_sum}")
            # print(self.node_map)

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
