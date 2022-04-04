import copy
import math
# import multiprocessing
import queue
import random
import time
from multiprocessing import Queue, Value
from threading import Thread

import numpy as np
from torch import multiprocessing as mp

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
# executor = ProcessPoolExecutor()
from rl_agent.critic import Critic
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
            while current_fork_num <= fork_factor:
                if done_flag.value:
                    break
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
        mp_context = mp.get_context('fork')  # Todo - Needed to change this from "fork" so it didn't crash

        self.to_workers_message_que = mp_context.Queue()
        self.from_worker_message_que = mp_context.Queue()

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

    # def _concurrency_tickler(self):
    #     value = None
    #     if self.active_p_semaphore < self.worker_thread_count + 5:
    #         # if there are more slots left try to fetch and continue if no one is free
    #         # TODO: mabye loop this to fetch untill the que is empty
    #         try:
    #             value: StateValuePropegator = self.from_worker_message_que.get_nowait()
    #         except queue.Empty as e:
    #             pass
    #     else:
    #         # if all the slots are full wait untill a worker is done
    #         value = self.from_worker_message_que.get()
    #         self.active_p_semaphore -= 1
    #         self._apply_value_propegator(value)

    def _clear_worker_que(self):
        try_fetch = True
        num_burned = 0
        num_applied = 0

        while try_fetch:
            try:
                _ = self.to_workers_message_que.get_nowait()
                num_burned += 1
                self.active_p_semaphore -= 1
            except queue.Empty as e:
                try_fetch = False

        # self._have_all_registerd_done()

        try_fetch = True
        while try_fetch:
            try:
                v = self.from_worker_message_que.get_nowait()
                self._apply_value_propegator(v)
                num_applied += 1
                self.active_p_semaphore -= 1
            except queue.Empty as e:
                try_fetch = False

        if self.debug:
            print(f"burned {num_burned} que tasks. applied {num_applied} que tasks")

    def _concurrency_tickler(self):
        try_fetch = True
        # clear the que if there are any items in it:
        while try_fetch:
            try:
                # if we got a value from the que apply it to the current tree
                value: StateValuePropegator = self.from_worker_message_que.get_nowait()
                self.active_p_semaphore -= 1
                self._apply_value_propegator(value)
            except queue.Empty as e:
                try_fetch = False

        # if there are to many threads active hang untill some finishes
        if self.active_p_semaphore > self.worker_thread_count * self.fork_factor:
            try:
                value = self.from_worker_message_que.get(timeout=(20 / 1000))
                self.active_p_semaphore -= 1
                self._apply_value_propegator(value)
            except Exception as e:
                return False
        return True

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
        # TODO: URGENT figure out why the last and second element of the change list are equal
        # print(change_list)
        # change_list.pop(-1)
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

        # if time.monotonic_ns() > stop_t:
        #     return

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
            # if self._get_node_by_hash(child.node_hash).visits > 0:
            if not child.has_been_expanded:
                self._expand_node(child)

            self._tree_search(parent=child, value_prop=value_prop, d=d, stop_t=stop_t)
        else:
            # if the child has not been rolled out we roll it out
            # self._paralell_rollout(child, update_list)
            self._pre_apply_tree_search_node_visits(value_prop)

            # new_s, _, term = self.environment.act(parent.state, child.action_from_parent)
            # if term:
            #     print("WAAAAA")
            # child.state = new_s
            self._paralell_rollout(child, value_prop)

            # value = self.agent.critic.get_state_value(child.state)[0]
            # update_list.insert(0, value)
            # update_list.append(child.node_hash)
            # self._apply_node_change_list(change_list=update_list)

    def pick_child(self,
                   search_node: MonteCarloTreeSearchNode):
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
            confidence_bound = calculate_upper_confidence_bound_node_value(self.exploration_c, node.visits,
                                                                           parent_node.visits)

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
        for _ in range(self.worker_thread_count * 10):
            self.to_workers_message_que.put(-1)

        # if self.clean_thread is not None:
        #     self.clean_thread.kill()
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
            # p.kill()
            # p.join(timeout=0.01)
            p.kill()

    def searh_loop(self,
                   stop_t,
                   root_s_node,
                   is_training
                   ):
        while time.monotonic_ns() < stop_t:

            # print(self.active_p_semaphore)
            search_round = self._concurrency_tickler()
            if search_round:
                value_prop = StateValuePropegator()
                value_prop.tree_search_node_state_hashes.append(root_s_node.node_hash)

                self._tree_search(parent=root_s_node, value_prop=value_prop, stop_t=stop_t)

        # if not self.is_training:
        #     time.sleep(0.2)
        # else:
        while self.active_p_semaphore > 0:
            # print(self.active_p_semaphore)
            self._concurrency_tickler()
        # time.sleep(0.1)
        # print(f"start_clear at stop_t -: {math.floor((time.monotonic_ns() - stop_t) / 1000000)}ms")
        self._clear_worker_que()
        # print(f"end_clear at stop_t -: {math.floor((time.monotonic_ns() - stop_t) / 1000000)}ms")

    def mc_tree_search(self,
                       root_state: BoardGameBaseState):
        """
        The Monte Carlo tree search.
        TODO: Refactor code into smaller pieces.
        """

        start_time = time.monotonic_ns()
        self.active_p_semaphore = 0

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
        # print(f"Using a critic eval frac of {self.critic_eval_frac * 100}%")

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

        # if root_s_node.has_terminal_child:
        #     print("\n\n\n\n")

        # winning_m = self.environment.get_state_winning_move(root_s_node.state)
        # if winning_m is not None:
        #     ret = {}
        #     # print("\n\n\nHAS WINNING MOVE")
        #     for c in root_s_node.children:
        #         c: MonteCarloTreeSearchNode = c
        #         # c.state.change_turn()
        #         # done = self.environment.is_state_won(c.state)  # no fucking clue why this does not work like everything else. the same shitty method is called elswher and works
        #         if c.terminal:
        #             ret[c.action_from_parent] = 1
        #         else:
        #             ret[c.action_from_parent] = 0
        #     return ret, {}

        winning_m = self.environment.get_state_winning_move(root_s_node.state)
        if winning_m is not None:
            # print("\n" * 20)
            valid_actons = self.environment.get_valid_actions(root_s_node.state)
            ret = {}
            # print("\n\n\nHAS WINNING MOVE ", winning_m)
            # self.environment.display_state(root_state)
            for act in valid_actons:

                # this is a shit show
                s2, r, d = self.environment.act(state=root_state, action=act, inplace=False)
                if self.environment.get_winning_player_id(s2) is not None:
                    # self.environment.display_state(s2)
                    ret[act] = 1
                else:
                    ret[act] = 0
            # print(ret)
            return ret, {}

        root_s_node.state.change_turn()
        losing_m = self.environment.get_state_winning_move(root_s_node.state)
        root_s_node.state.change_turn()
        if losing_m is not None:
            ret = {losing_m: 1}
            # print("\n\n\nFound losing move MOVE", losing_m)
            for c in root_s_node.children:
                c: MonteCarloTreeSearchNode = c
                if ret.get(c.action_from_parent) is None:
                    ret[c.action_from_parent] = 0

            return ret, {}

        # print(f"calc stop_t at start t +: {math.floor((time.monotonic_ns() - start_time) / 1000000)}ms")

        # for i in range(num_rollouts):

        # is_training = True
        self.rnds = 0
        # self.clean_thread = Thread(target=self.searh_loop, args=(stop_t, root_s_node, is_training))
        # self.clean_thread.start()
        # wait_t_ajusted = ((stop_t - time.monotonic_ns()) / 1000000)

        self.searh_loop(stop_t, root_s_node, True)
        # print(f"wait t -: {(wait_t_ajusted)}ms,  f conf ms{self.agent.ms_tree_search_time}")

        # time.sleep(max(0, wait_t_ajusted) / 1000)

        ret = {}  # ehhhh
        ret_2_electric_bogaloo = {}  # ehhhh

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

            # TODO: CONFIUGURE FROM ELSWHERE
            # if node.visits > 30:
            #     # val = node.p1_wins if child_s_node.state.current_player_turn() == 0 else node.p2_wins
            #     val = node.value
            #     v = (val / node.visits)
            #     ret_2_electric_bogaloo[child.state] = v

        ret_2_electric_bogaloo[root_s_node.state] = root_node.discounted_value / root_node.visits

        # print(f"end agr at stop_t -: {math.floor((time.monotonic_ns() - stop_t) / 1000000)}ms")

        if self.debug:
            print(f"completed {self.rnds} rollouts in the {wait_milli_sec}ms limit")
            p_dists = self.agent.model.get_probability_distribution([root_s_node.state])[0]

            state_vals = self.agent.critic.get_states_value([c.state for c in root_s_node.children])

            # print(self.agent.model.forward(torch.Tensor([root_s_node.state.get_as_vec()])))
            # print(p_dists)

            print(f"parent visits: {root_node.visits}, value: {root_node.value}")
            v_count_sum = 0
            for c, critic_q_val in zip(root_s_node.children, state_vals):
                critic_q_val = critic_q_val[0]
                node = self._get_node_by_hash(c.node_hash)
                v_count_sum += node.visits

                action_idx = self.environment.get_action_space().index((c.action_from_parent))
                p_dist_val = p_dists[action_idx]

                # print(f"action {c.action_from_parent} -> {c.state.get_as_vec()} node has {node.visits} visits value {node.value} value ->  {node.value / (node.visits + 1)}")  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
                print(
                    "action {} ->  node visits: {:<4} value: {:<9.3f} | agent| pred: {:<9.3} actual: {:<8.4} error: {:<9.4}  |critic Q(s,a)(with discount)| pred: {:<9.3} actual: {:<8.4}  error: {:<8.4} ".format(
                        c.action_from_parent,
                        # c.state.get_as_vec(),
                        node.visits,
                        float(node.discounted_value),
                        p_dist_val,
                        node.visits / v_sum,
                        p_dist_val - ((node.visits + 1) / (v_sum + 1)),
                        critic_q_val,
                        (node.discounted_value + 1) / (node.visits + 1),
                        critic_q_val - ((node.discounted_value + 1) / (node.visits + 1))
                    ))  # + {calculate_upper_confidence_bound_node_value(node, root_node)}")
            print(f"v count sum: {v_count_sum}")
            # print(self.node_map)

            end_time = time.monotonic_ns()
            print(f"mc tree search RTT: {math.floor((end_time - start_time) / 1000000)}ms")
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
