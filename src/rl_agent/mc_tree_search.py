import copy
import math
# import multiprocessing
import random
import time
from multiprocessing import Queue, Value
from threading import Thread

import numpy as np
from torch import multiprocessing as mp

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
# executor = ProcessPoolExecutor()
from rl_agent.util import EGreedy


class NodeValueCounter:
    def __init__(self):
        self.visits = 0
        self.value = 0
        self.discounted_value = 0


class StateValuePropegator:
    def __init__(self):
        self.tree_search_node_state_hashes = []
        self.tree_search_nodes_has_visits_pre_propegated = False
        self.rollout_nodes_state_hashes = []
        self.value = 0


class MonteCarloTreeSearchNode:
    def __init__(self,
                 state: BoardGameBaseState,
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


def _parallel_mc_rollout(
        state: BoardGameBaseState,
        agent,
        environment: BoardGameEnvironment,
        value_prop: StateValuePropegator,
        e_greedy: EGreedy,
        use_critic_prob: float,
        use_prob: bool = False,
):
    state_hash = hash(state)

    winning_move = environment.get_state_winning_move(state)

    # environment.display_state(state)
    if winning_move is not None:
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
    rn = random.random()
    if done:
        value_prop.value = reward
        value_prop.rollout_nodes_state_hashes.append(hash(next_state))
    elif rn < use_critic_prob:
        value = agent.critic.get_state_value(state)[0]
        value_prop.value = value
        value_prop.rollout_nodes_state_hashes.append(hash(next_state))
    else:
        _parallel_mc_rollout(next_state, agent, environment, value_prop, e_greedy, use_critic_prob, use_prob)

    value_prop.rollout_nodes_state_hashes.append(state_hash)


def parallel_rollout(agent,
                     env: BoardGameEnvironment,
                     in_que: Queue,
                     out_que: Queue,
                     e_greedy: EGreedy,
                     fork_factor: int,
                     use_prob_not_max_action: bool,
                     done_flag):
    run = True
    timeout_s = 60
    while run:
        try:
            msg = in_que.get(timeout=timeout_s)
        except Exception as e:
            print("THREAD TIMEOUT")
            break
        if msg == -1:
            break  # it's fucking horrible but normal python practice
        current_fork_num = 0
        if msg is not None:
            org_root_state, org_value_prop, use_critic_prob = msg
            while current_fork_num < fork_factor:
                if done_flag.value:
                    break

                # todo: figure out wheter to uncomment this and apply the forked rollouts visits
                # if current_fork_num > 0:
                #     org_value_prop.tree_search_nodes_has_visits_pre_propegated = False
                root_state = copy.deepcopy(org_root_state)
                value_prop = copy.deepcopy(org_value_prop)

                _parallel_mc_rollout(root_state, agent, env, value_prop, e_greedy, use_prob=use_prob_not_max_action,
                                     use_critic_prob=use_critic_prob)

                out_que.put(value_prop)
                e_greedy.reset()
                current_fork_num += 1
        else:
            run = False


def calculate_upper_confidence_bound_node_value(exploration_c,
                                                node_visits,
                                                parent_visits):
    ucbt = exploration_c * np.sqrt(np.divide(np.log1p(parent_visits), (node_visits + 1)))
    return ucbt


class MontecarloTreeSearch:
    def __init__(self,
                 exploration_c,
                 environment: BoardGameEnvironment,
                 agent,
                 worker_thread_count: int,
                 worker_fork_number,
                 is_training=True):
        self.is_training = is_training
        self.worker_thread_count = worker_thread_count
        self.agent = agent
        self.environment = environment
        self.exploration_c = exploration_c
        self.value_discount = 0.9
        self.fork_factor = worker_fork_number

        self.search_node_map = {}
        self.debug = False
        self.rnds = 0
        mp_context = mp.get_context('fork')

        self.to_workers_message_que = Queue()
        self.from_worker_message_que = Queue()

        self.clean_thread = None

        self.e_greedy: EGreedy = None
        self.done_flag = Value('i', False)

        self.agent.model.share_memory()
        e_greedy = EGreedy(init_val=0.8, min_val=0.01, rounds_to_min=3)
        # e_greedy = EGreedy(init_val=0, min_val=0.0, rounds_to_min=10)
        self.mp_processes = []
        for n in range(worker_thread_count):
            use_prob = random.random() > 0.5
            p = mp_context.Process(target=parallel_rollout,
                                   args=(self.agent, self.environment, self.to_workers_message_que,
                                         self.from_worker_message_que, e_greedy, self.fork_factor, use_prob,
                                         self.done_flag))
            p.start()
            self.mp_processes.append(p)

        self.active_p_semaphore = 0
        self.node_map = {}

        self.critic_eval_frac = 0

    ################################
    #          Utilitys            #
    ################################

    def _calculate_critic_eval_frac(self):
        return 0

    def _clear_worker_que(self):

        try_fetch = True
        num_burned = 0
        num_applied = 0

        while try_fetch:
            try:
                _ = self.to_workers_message_que.get_nowait()
                num_burned += 1
                self.active_p_semaphore -= self.fork_factor
            except Exception as e:
                try_fetch = False

        try_fetch = True
        while try_fetch:
            try:
                v = self.from_worker_message_que.get_nowait()
                self._apply_value_propegator(v)
                num_applied += 1
                self.active_p_semaphore -= 1
            except Exception as e:
                try_fetch = False

        if self.debug:
            print(f"burned {num_burned} que tasks. applied {num_applied} que tasks")

    def _concurrency_tickler(self):
        continue_fetching = True
        # clear the que if there are any items in it:
        while continue_fetching:
            try:
                # if we got a value from the que apply it to the current tree
                value: StateValuePropegator = self.from_worker_message_que.get_nowait()
                self.active_p_semaphore -= 1
                self._apply_value_propegator(value)
            except Exception as e:
                continue_fetching = False

        search_round = False
        # if there are to many threads active hang untill some finishes
        if self.active_p_semaphore > self.worker_thread_count * self.fork_factor:
            try:
                value = self.from_worker_message_que.get(timeout=20 / 1000)
                self.active_p_semaphore -= 1
                self._apply_value_propegator(value)
            except Exception as e:
                pass
        else:
            search_round = True

        return search_round

    def _paralell_rollout(self,
                          child,
                          value_prop):

        self.active_p_semaphore += self.fork_factor
        self.to_workers_message_que.put((child.state, value_prop, self.critic_eval_frac))

    ################################
    #       Node management        #
    ################################

    def _apply_value_propegator(self,
                                value_prop: StateValuePropegator):

        self.rnds += 1
        prop_val = value_prop.value
        discounted_prop_val = value_prop.value
        for node_hash in reversed(value_prop.rollout_nodes_state_hashes):
            discounted_prop_val *= self.value_discount
            node = self._get_node_by_hash(node_hash)
            node.visits += 1
            node.value += prop_val
            node.discounted_value += discounted_prop_val

        for node_hash in reversed(value_prop.tree_search_node_state_hashes):
            discounted_prop_val *= self.value_discount
            node = self._get_node_by_hash(node_hash)

            node.value += prop_val
            node.discounted_value += discounted_prop_val
            if not value_prop.tree_search_nodes_has_visits_pre_propegated:
                node.visits += 1

    def _pre_apply_tree_search_node_visits(self,
                                           value_prop):
        """
        becase the rollouts are done asycronosly to the traversal we need to pre apply the visit count for the nodes.
        :param value_prop:
        :return:
        """
        if value_prop.tree_search_nodes_has_visits_pre_propegated:
            raise Exception("illigal state")

        value_prop.tree_search_nodes_has_visits_pre_propegated ^= True

        for _ in range(self.fork_factor):
            for node_hash in value_prop.tree_search_node_state_hashes:
                node = self._get_node_by_hash(node_hash)
                node.visits += 1

    def _get_node_by_state(self,
                           state: BoardGameBaseState) -> NodeValueCounter:
        node = self.node_map.get(hash(state))
        if node is None:
            node = NodeValueCounter()
            self.node_map[hash(state)] = node
        return node

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

    def _expand_node(self,
                     search_node: MonteCarloTreeSearchNode):
        # may be optimized by saving the state action edges to avoid unnecessarily initializing new memory for new states already found
        possible_actions = self.environment.get_valid_actions(search_node.state)

        search_nodes = []

        for action in possible_actions:
            new_s, r, done = self.environment.act(search_node.state, action, inplace=False)

            new_search_node = MonteCarloTreeSearchNode(
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

        search_node.children = search_nodes

    def _tree_search(self,
                     parent: MonteCarloTreeSearchNode,
                     value_prop: StateValuePropegator,
                     stop_t,
                     d=0):

        # # if the node is not expanded, expand it
        if not parent.has_been_expanded:
            self._expand_node(parent)

        # check if any of the nodes children are terminal, if they are terminate the search and apply value propegation for all the terminal states
        if parent.has_terminal_child:
            propegators = []
            for child in parent.children:
                child_s_node: MonteCarloTreeSearchNode = child
                if child_s_node.terminal:
                    prop_copy = copy.deepcopy(value_prop)
                    prop_copy.value = child_s_node.reward
                    prop_copy.tree_search_node_state_hashes.append(child.node_hash)
                    propegators.append(prop_copy)

            for prop in propegators:
                self._apply_value_propegator(prop)
            return

        if parent.terminal:
            value_prop.tree_search_node_state_hashes.append(parent.node_hash)
            self._apply_value_propegator(value_prop)
            return

        # else use the default tree policy to pick the child and add it to the update list
        if self.e_greedy.should_pick_greedy(increment_round=True):
            child: MonteCarloTreeSearchNode = self.pick_child(parent)
        else:
            child: MonteCarloTreeSearchNode = random.choice(parent.children)

        value_prop.tree_search_node_state_hashes.append(child.node_hash)

        # if the child has been rolled out we continue down the tree
        if child.has_been_rolled_out:
            if not child.has_been_expanded:
                self._expand_node(child)

            self._tree_search(parent=child, value_prop=value_prop, d=d, stop_t=stop_t)
        else:
            # if the child has not been rolled out we roll it out
            self._pre_apply_tree_search_node_visits(value_prop)
            self._paralell_rollout(child, value_prop)

    def pick_child(self,
                   search_node: MonteCarloTreeSearchNode):
        player_0_turn = search_node.state.current_player_turn() == 0

        best_child = random.choice(search_node.children)

        if player_0_turn:
            best_child_value = 0
        else:
            best_child_value = float("inf")

        parent_node = self._get_node_by_hash(search_node.node_hash)
        for child in search_node.children:
            if child.has_terminal_child:
                # if the next move after can be terminal, means the opponent can win.
                continue

            node = self._get_node_by_hash(child.node_hash)
            confidence_bound = calculate_upper_confidence_bound_node_value(self.exploration_c, node.visits,
                                                                           parent_node.visits)

            node = self._get_node_by_hash(child.node_hash)
            if player_0_turn:
                val = ((node.discounted_value + 1) / (node.visits + 1)) + confidence_bound
                if val > best_child_value:
                    best_child_value = val
                    best_child = child
            else:
                val = ((node.discounted_value + 1) / (node.visits + 1)) - confidence_bound
                if val < best_child_value:
                    best_child_value = val
                    best_child = child

        return best_child

    ################################
    #           Public             #
    ################################

    def close_helper_threads(self):
        for _ in range(self.worker_thread_count * 10):
            self.to_workers_message_que.put(-1)

        while True:
            try:
                self.from_worker_message_que.get_nowait()
            except Exception:
                break

        while True:
            try:
                self.to_workers_message_que.get_nowait()
            except Exception:
                break

        self.from_worker_message_que.close()
        self.from_worker_message_que.join_thread()
        self.to_workers_message_que.close()
        self.to_workers_message_que.join_thread()
        for p in self.mp_processes:
            p.kill()

    def searh_loop(self,
                   stop_t,
                   root_s_node,
                   ):
        while time.monotonic_ns() < stop_t:
            search_round = self._concurrency_tickler()
            if search_round:
                value_prop = StateValuePropegator()
                value_prop.tree_search_node_state_hashes.append(root_s_node.node_hash)
                self._tree_search(parent=root_s_node, value_prop=value_prop, stop_t=stop_t)

        while self.active_p_semaphore > 0:
            self._concurrency_tickler()
        self._clear_worker_que()

    def mc_tree_search(self,
                       root_state: BoardGameBaseState):
        """
        The Monte Carlo tree search.
        TODO: Refactor code into smaller pieces.
        """

        start_time = time.monotonic_ns()

        wait_milli_sec = self.agent.ms_tree_search_time
        c_time = time.monotonic_ns()
        stop_t = c_time + (wait_milli_sec * 1000000)

        if self.debug:
            print("=" * 50)
            print(f"Player {root_state.current_player_turn()} turn")
            print()

        if self.clean_thread is not None:
            self.clean_thread.join()
            self.clean_thread = None

        # print(f"clean thread start at start t +: {math.floor((time.monotonic_ns() - start_time) / 1000000)}ms")

        root_s_node: MonteCarloTreeSearchNode = self.search_node_map.get(hash(root_state))
        root_node = self._get_node_by_hash(hash(root_state))

        # print(f"fetch root at start t +: {math.floor((time.monotonic_ns() - start_time) / 1000000)}ms")

        self.critic_eval_frac = 0.0  # self._calculate_critic_eval_frac()

        # add some randomness to the 100 first traversals
        self.e_greedy = EGreedy(
            init_val=0.8,
            min_val=0.0,
            rounds_to_min=20,  # TODO: FIX IMPORTANT!
        )

        if root_s_node is None:
            root_s_node = MonteCarloTreeSearchNode(
                state=root_state,
                node_hash=hash(root_state),
                has_been_rolled_out=False,
                has_been_expanded=False
            )

        if not root_s_node.has_been_expanded:
            self._expand_node(root_s_node)

        winning_m = self.environment.get_state_winning_move(root_s_node.state)
        if winning_m is not None:
            print("found winning move ", winning_m)
            valid_actons = self.environment.get_valid_actions(root_s_node.state)
            ret = {}
            for act in valid_actons:

                # this is a shit show
                s2, r, d = self.environment.act(state=root_state, action=act, inplace=False)
                if self.environment.get_winning_player_id(s2) is not None:
                    ret[act] = 1
                else:
                    ret[act] = 0
            return ret

        root_s_node.state.change_turn()
        losing_m = self.environment.get_state_winning_move(root_s_node.state)
        root_s_node.state.change_turn()
        if losing_m is not None:
            ret = {losing_m: 1}
            print("found losing move ", losing_m)
            for c in root_s_node.children:
                c: MonteCarloTreeSearchNode = c
                if ret.get(c.action_from_parent) is None:
                    ret[c.action_from_parent] = 0

            return ret

        self.rnds = 0

        self.active_p_semaphore = 0
        self.searh_loop(stop_t, root_s_node)

        ret = {}  # ehhhh

        max_v, max_a = 0, None
        v_sum = 0

        # print(f"start agr at stop_t -: {math.floor((time.monotonic_ns() - stop_t) / 1000000)}ms")
        for child in root_s_node.children:
            child_s_node: MonteCarloTreeSearchNode = child
            node = self._get_node_by_hash(child_s_node.node_hash)
            ret[child.action_from_parent] = node.visits

            v_sum += node.visits
            if node.visits > max_v:
                max_a = child_s_node.action_from_parent
                max_v = node.visits

        if self.debug:
            print(f"completed {self.rnds} rollouts in the {wait_milli_sec}ms limit")
            p_dists = self.agent.model.get_probability_distribution([root_s_node.state])[0]

            print(f"parent visits: {root_node.visits}, value: {root_node.value}")
            v_count_sum = 0
            for c in root_s_node.children:
                node = self._get_node_by_hash(c.node_hash)
                v_count_sum += node.visits

                action_idx = self.environment.get_action_space().index((c.action_from_parent))
                p_dist_val = p_dists[action_idx]

                # print(f"action {c.action_from_parent} -> {c.state.get_as_vec()} node has {node.visits} visits value {node.value} value ->  {node.value / (node.visits + 1)}")  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
                print(
                    "action {} ->  node visits: {:<4} value: {:<9.3f} v{:<9} | agent| pred: {:<9.3} actual: {:<8.4} error: {:<9.4}".format(
                        c.action_from_parent,
                        # c.state.get_as_vec(),
                        node.visits,
                        float(node.discounted_value),
                        node.value,
                        p_dist_val,
                        node.visits / v_sum,
                        p_dist_val - ((node.visits + 1) / (v_sum + 1)),
                    ))  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
            print(f"v count sum: {v_count_sum}")
            # print(self.node_map)

            end_time = time.monotonic_ns()
            print(f"mc tree search RTT: {math.floor((end_time - start_time) / 1000000)}ms")
        return ret
