import itertools
import math
import random
import os

import numpy as np
import pandas
import torch
from torch import nn

from abc import abstractmethod
from os import path
from enviorments.base_environment import BaseEnvironment, BoardGameEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState

# TODO: generalize
# TODO: have not checked that this actually works at all
from rl_agent.agent_net import ActorNeuralNetwork
from rl_agent.critic import Critic, CriticNeuralNet
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.util import generate_batch, get_action_visit_map_as_target_vec


class MonteCarloTreeSearchAgent:

    def __init__(self,
                 environment: BaseEnvironment,
                 num_rollouts: int,
                 worker_thread_count,
                 exploration_c,
                 topp_saves,
                 ms_tree_search_time
                 ):
        # TODO: implement layer config
        self.ms_tree_search_time = ms_tree_search_time
        self.topp_saves = topp_saves
        self.exploration_c = exploration_c
        self.worker_thread_count = worker_thread_count
        self.num_rollouts = num_rollouts
        self.environment = environment
        self.train_buffer: [([int], [float])] = []

        self.model: ActorNeuralNetwork = None
        self._build_network()

        self.debug = False
        self.display = False
        self.loss_hist = []

        input_s = self.environment.get_observation_space_size()
        self.critic = Critic(input_s, math.floor(math.sqrt(input_s)))

        self.critic.model.share_memory()

    def load_model_from_fp(self,
                           fp):
        if fp is not None and path.exists(fp):
            input_s = self.environment.get_observation_space_size()
            output_s = self.environment.get_action_space_size()
            xs = math.floor(math.sqrt(input_s))
            model = ActorNeuralNetwork.load_model(fp, input_s, output_s, xs)
            self.model = model

            critic_fp = fp + "_critic"
            critic_model = CriticNeuralNet.load_model(critic_fp, input_s, xs)
            self.critic.model = critic_model

            self.model.share_memory()
            self.critic.model.share_memory()
        else:
            print("#" * 10)
            print("path not found model not loaded")
            print("#" * 10)

    def save_actor_critic_to_fp(self,
                                fp):
        if fp is not None:
            critic_fp = fp + "_critic"
            self.model.save_model(fp)
            self.critic.model.save_model(critic_fp)

    def _build_network(self):

        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        b_size = math.floor(math.sqrt(input_s))
        self.model = ActorNeuralNetwork(input_s, output_s, b_size).to("cpu")

    def _train_network(self,
                       r_buffer):
        # device_used = "cuda"
        # model = self.model.to(device=device_used)
        # model.train(True)
        # torch.set_printoptions(profile="full", linewidth=1000)
        # loss_fn = torch.nn.CrossEntropyLoss()
        # opt = torch.optim.Adam(model.parameters())

        passes_over_data = 70
        batch_size = 3
        train_itrs = math.ceil((len(r_buffer) * passes_over_data) / batch_size)
        # print(train_itrs)

        inp_x, inp_y = [], []
        for state, v_count_map in r_buffer:
            inp_x.append([*state.get_as_vec(), state.current_player_turn()])
            inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=False))
            # if state.current_player_turn == 0:
            #     inp_x.append(state.get_as_vec())
            #     inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=False))
            # else:
            #     inp_x.append(state.get_as_inverted_vec())
            #     inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=True))
            #
        self.loss_hist.extend(self.model.train_network(inp_x, inp_y, train_itrs, batch_size))
        # for _ in range(train_itrs):
        #     x, y = generate_batch(inp_x, inp_y, batch_size)

        # x = torch.tensor(x, dtype=torch.cuda.FloatTensor, device=device_used)
        # y = torch.tensor(y, dtype=torch.cuda.FloatTensor, device=device_used)

        # # TODO: mabye set up to inp the list and make the tensor on the model
        # x = torch.tensor(x, dtype=torch.float)
        # y = torch.tensor(y, dtype=torch.float)
        #
        # # pred = model.forward(x)
        # # print(y)
        # # print(pred)
        # # print("y:    ", y)
        # # print("pred: ", pred)
        # # print(pred)
        #
        # # loss: torch.Tensor = loss_fn(pred, y)
        # loss = self.model.train_pass(x, y)
        # self.loss_hist.append(loss.item())

        # print(loss)
        # opt.zero_grad()
        # loss.backward()
        # opt.step()

        # model.train(False)
        # self.model = model.to("cpu")

    def print_progress(self,
                       n,
                       episodes,
                       res_list):
        """
        Prints the progress in the terminal.
        """
        bar_size = 50
        update_n_rounds = 10

        # update each
        bars = math.floor(bar_size * (n / episodes))
        prints = "[{:<50}] episode{:>4}/{:<4} tot wins: {:<4}, wins % last 10: {:>4.2}".format(
            "|" * bars,
            n,
            episodes,
            sum(res_list),
            sum(res_list[-11:]) / 10,
        )

        print(prints, end="\r")

    def get_tree_search_action(self,
                               current_state,
                               mcts,
                               ):
        # do the montecarlo tree rollout
        mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)

        # convert the vec to target dist
        all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)

        # pick the action to do
        target_idx = all_action_dist.index(max(all_action_dist))

        return self.environment.get_action_space_list()[target_idx]

    def _take_game_move(self,
                        current_state,
                        mcts,
                        replay_buffer,
                        critic_train_set):

        # do the montecarlo tree rollout
        mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)

        # convert the vec to target dist
        all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)

        # put the target dist in the replay buffer
        replay_buffer.append((current_state, mc_visit_counts_map))

        # pick the action to do
        target_idx = all_action_dist.index(max(all_action_dist))

        action = self.environment.get_action_space_list()[target_idx]
        next_s, r, game_done = self.environment.act(current_state, action)

        if self.display:
            self.environment.display_state(next_s)

        if critic_train_map is not None:
            # x, y = [], []
            for vec, val in critic_train_map.items():
                # print(val)
                # x.append(vec)
                # y.append(val)
                # if vec.current_player_turn() == 0:
                critic_train_set.append((vec.get_as_vec(), val))

        return next_s, r, game_done

    def run_episode(self,
                    player_2=None,
                    flip_start=False,
                    train_critic=False,
                    ):
        game_done = False
        replay_buffer = []
        critic_train_set = []

        mcts = MontecarloTreeSearch(
            exploration_c=self.exploration_c,
            environment=self.environment,
            agent=self,
            worker_thread_count=self.worker_thread_count
        )
        mcts.debug = self.debug
        current_state = self.environment.get_initial_state()

        if flip_start:
            current_state.change_turn()

        torch.set_num_threads(self.worker_thread_count)
        while not game_done:

            if player_2 is None or current_state.current_player_turn() == 0:
                current_state, r, game_done = self._take_game_move(current_state, mcts, replay_buffer, critic_train_set)
            else:
                current_state, r, game_done = player_2(current_state)

        if train_critic:
            self.critic.train_network(critic_train_set)

        did_win = self.environment.winning_player_id(current_state) == 0
        mcts.close_helper_threads()

        torch.set_num_threads(1)
        return replay_buffer, did_win

    # def __run_episode(self,
    #                 flip_start=False):
    #     game_done = False
    #     replay_buffer = []
    #     critic_train_set = []
    #
    #     mcts = MontecarloTreeSearch(
    #         exploration_c=self.exploration_c,
    #         environment=self.environment,
    #         agent=self,
    #         worker_thread_count=self.worker_thread_count
    #     )
    #     mcts.debug = self.debug
    #
    #     current_state = self.environment.get_initial_state()
    #     if flip_start:
    #         current_state.change_turn()
    #
    #     torch.set_num_threads(10)
    #     while not game_done:
    #         if self.debug:
    #             print("started new episode")
    #         mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)
    #         all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)
    #         replay_buffer.append((current_state, mc_visit_counts_map))
    #
    #         # TODO: should probably not always be greedy
    #         if current_state.current_player_turn() == 0:
    #             target_idx = all_action_dist.index(max(all_action_dist))
    #         else:
    #             target_idx = all_action_dist.index(max(all_action_dist))
    #
    #         action = self.environment.get_action_space_list()[target_idx]
    #         next_s, r, game_done = self.environment.act(current_state, action)
    #         current_state = next_s
    #
    #         if self.display:
    #             self.environment.display_state(next_s)
    #
    #         if critic_train_map is not None:
    #             # x, y = [], []
    #             for vec, val in critic_train_map.items():
    #                 # print(val)
    #                 # x.append(vec)
    #                 # y.append(val)
    #                 # if vec.current_player_turn() == 0:
    #                 critic_train_set.append((vec.get_as_vec(), val))
    #             # else:
    #             #     critic_train_set.append((vec.get_as_inverted_vec(), val))
    #
    #     ## critic ##
    #     # res = self.critic.get_states_value(x)
    #     # avg_critic_error = np.mean([math.dist((r,), (yv,)) for r, yv in zip(res, y)])
    #     # print("avg critic error ", avg_critic_error)
    #     self.critic.train_network(critic_train_set)
    #     ## critic ##
    #
    #     did_win = r == 1
    #     mcts.close_helper_threads()
    #
    #     torch.set_num_threads(1)
    #     return replay_buffer, did_win

    def human_move(self,
                   current_state):
        valid_move = False
        valid_actions = self.environment.get_valid_actions(current_state)
        while not valid_move:
            print(f"available moves: {valid_actions}")

            inp = input("input move(21,4) -> 21 4: ")
            inp = inp.strip()

            try:
                splits = inp.split(" ")
                n1, n2 = splits[0], splits[1]
                user_action = (int(n1), int(n2))
                # print(user_action)
                # print(valid_actions)
                if user_action not in valid_actions:
                    print("invalid user action")
                else:
                    valid_move = True
                    next_s, r, game_done = self.environment.act(current_state, user_action)
            except Exception:
                pass

        return next_s, r, game_done

    def play_against_human(self):
        self.run_episode(
            player_2=self.human_move,
            train_critic=False,
            flip_start=False,
        )

    def run_topp(self,
                 n,
                 num_games):
        topp = TOPP(
            total_itrs=n,
            game_rollouts=self.num_rollouts,
            num_games_in_matches=num_games,
            num_models_to_save=self.topp_saves,
            environment=self.environment)

        topp.run_tournaments()

    def train_n_episodes(self,
                         n,
                         fp=None,
                         games_in_topp_matches=500):

        topp = TOPP(
            total_itrs=n,
            game_rollouts=self.num_rollouts,
            num_games_in_matches=games_in_topp_matches,
            num_models_to_save=self.topp_saves,
            environment=self.environment
        )

        win_count = [0 for _ in range(50)]
        for v in range(n):
            flip = random.random() > 0.5
            r_buf, win = self.run_episode(
                flip_start=flip,
                train_critic=True
            )
            if win:
                win_count.append(1)
            else:
                win_count.append(0)
            self._train_network(r_buf)

            self.print_progress(v, n, win_count)
            if self.display:
                print()

            if v % 10 == 0:
                self.save_actor_critic_to_fp(fp)

            topp.register_policy(self, v, run_tornament=True)

            actor_loss = pandas.DataFrame(self.loss_hist)
            actor_loss.to_csv(fp + "_actor_loss.csv", index=False)
            critic_loss = pandas.DataFrame(self.critic.loss_hist)
            critic_loss.to_csv(fp + "_critic_loss.csv", index=False)
        print()
        self.save_actor_critic_to_fp(fp)

    def get_prob_dists(self,
                       state_list: []):

        x = [[*state.get_as_vec(), state.current_player_turn()] for state in state_list]

        prob_dist = self.model.forward(torch.tensor(x, dtype=torch.float))

        prob_dist = prob_dist.tolist()

        return prob_dist  # [random.random() if a != 0 else 0 for a in state_list[0]]

    def pick_action(self,
                    state: BoardGameBaseState,
                    get_prob_not_max=False):

        """
        TODO: usikker på den her, mpt player 1/2. 
            spørsmålet er skal man "snu brettet" hver gang man gjør et trekk for motstanderen eller skal man bare plukke det trekke som er minst gunsig for spiller 1
            for øyeblikket tar jeg bare trekket som er minst gunsig for spiller 1 men kan være lurt og sjekke forsjellen når vi endrer dette
        """

        # if state.current_player_turn() == 0:
        #     x = state.get_as_vec()
        #     pass
        # else:
        #     x = state.get_as_inverted_vec()

        x = state.get_as_vec()
        prob_dist = self.model.forward(torch.tensor([[*x, state.current_player_turn()]], dtype=torch.float))[0]

        if torch.any(torch.isnan(prob_dist)):
            print("na in dist: inp: ", x)
            print("na in diat: prob dist: ", prob_dist)
            print("na in diat: params: ", self.model.parameters())
            # prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))[0]
            # print(prob_dist)

            # print(self.model.network(torch.tensor([x], dtype=torch.float)))
            pass

        prob_dist = prob_dist.tolist()

        # if all probs are zero pick a random value
        # TODO: implement the some action picker
        if sum(prob_dist) == 0 or get_prob_not_max:
            action = random.choice(self.environment.get_valid_actions(state))
        else:
            target_val = max(prob_dist)

            # print(prob_dist)
            # else:
            #     target_val = float("inf")
            #     for v in prob_dist:
            #         if v < target_val and v != 0:
            #             target_val = v
            action_idx = prob_dist.index(target_val)

            # if state.current_player_turn() == 0:
            action = self.environment.get_action_space_list()[action_idx]
            # else:
            #     action_inv = self.environment.get_action_space_list()[action_idx]
            #     action = (action_inv[1], action_inv[0])

        leagal_actions = self.environment.get_valid_actions(state)
        if action not in leagal_actions:
            print("Not in leagal dist:", prob_dist)
            print("Not in leagal action:", action)
            print("Not in leagal inp vec:", state.get_as_vec())
            # print(self.model.network(torch.tensor([x], dtype=torch.float)))
        return action


# because the python import system is fucking usless this has to sit at the botom here
class TOPP:

    def __init__(self,
                 total_itrs,
                 num_models_to_save,
                 num_games_in_matches,
                 game_rollouts,
                 environment: BoardGameEnvironment):
        self.game_rollouts = game_rollouts
        self.environment = environment
        self.num_games_in_matches = num_games_in_matches
        self.num_models_to_save = num_models_to_save
        self.total_itrs = total_itrs

        self._save_points = [0]
        every_n = math.floor(total_itrs / num_models_to_save)
        self._save_points.extend(np.cumsum(np.repeat(every_n, num_models_to_save - 1)).tolist())

        self._save_dir = "./topp_models"
        self._model_name = "topp_model_itr_{}"
        if not path.exists(self._save_dir):
            os.mkdir(self._save_dir)

    def _model_save_path(self,
                         itr):
        return self._save_dir + "/" + str.format(self._model_name, itr)

    def register_policy(self,
                        model: MonteCarloTreeSearchAgent,
                        itr,
                        run_tornament=False):
        if itr in self._save_points:
            model.save_actor_critic_to_fp(self._model_save_path(itr))

            if run_tornament:
                self._single_model_tournament(itr)

    def _only_single_model_tornament(self,
                                     itr):
        pass

    def _tornament_model_pick_action(self,
                                     state,
                                     model,
                                     ):
        # if state.current_player_turn() == 0:
        #     x = state.get_as_vec()
        # else:
        #     x = state.get_as_inverted_vec()
        x = state.get_as_vec()

        prob_dist = model.forward(torch.tensor([[*x, state.current_player_turn()]], dtype=torch.float))[0]
        prob_dist = prob_dist.tolist()

        if sum(prob_dist) == 0:
            action = random.choice(self.environment.get_valid_actions(state))
        else:
            target_val = max(prob_dist)
            action_idx = prob_dist.index(target_val)

            action = self.environment.get_action_space_list()[action_idx]
            # if state.current_player_turn() == 0:
            #     action = self.environment.get_action_space_list()[action_idx]
            # else:
            #     action_inv = self.environment.get_action_space_list()[action_idx]
            #     action = (action_inv[1], action_inv[0])

        return action

    def _torney(self,
                pl1,
                pl2,
                leader_board):

        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        xs = math.floor(math.sqrt(input_s))

        p1_score = 0
        p2_score = 0
        model_1 = ActorNeuralNetwork.load_model(self._model_save_path(pl1), input_s, output_s, xs)
        model_2 = ActorNeuralNetwork.load_model(self._model_save_path(pl2), input_s, output_s, xs)
        # print(f"iter {pl1} vs {pl2}")

        for n in range(self.num_games_in_matches):
            game_done = False
            game_state: BoardGameBaseState = self.environment.get_initial_state()

            # 50% chance for model 2 to start
            if random.random() > 0.5:
                game_state.change_turn()

            while not game_done:
                if game_state.current_player_turn() == 0:
                    action = self._tornament_model_pick_action(game_state, model_1)
                else:
                    action = self._tornament_model_pick_action(game_state, model_2)
                game_state, r, game_done = self.environment.act(game_state, action, inplace=True)

            if r == 1:
                p1_score += 1
                winner = pl1
                loser = pl2
            elif r == -1:
                p2_score += 1
                winner = pl2
                loser = pl1
            else:
                raise Exception("state error")

            leader_board[winner] = (leader_board[winner][0] + 1, leader_board[winner][1])
            leader_board[loser] = (leader_board[loser][0], leader_board[loser][1] + 1)

            frac_p1 = (p1_score + 1) / (p2_score + p1_score + 2)
            p1_bar = math.floor(frac_p1 * 100) * "|"
            p2_bar = math.floor((1 - frac_p1) * 100) * "|"
            print("iter: \u001b[32m{:<5}\u001b[0m -{:->4}-> \u001b[32m{}\u001b[31m{}\u001b[0m <-{:-<4}-iter: \u001b[31m{:>5}\u001b[0m".format(pl1,
                                                                                                                                              p1_score,
                                                                                                                                              p1_bar,
                                                                                                                                              p2_bar,
                                                                                                                                              p2_score,
                                                                                                                                              pl2) + " " * 100, end="\r")
        print()

    def _single_model_tournament(self,
                                 itr):
        used_save_p = [sp for sp in self._save_points if os.path.exists(self._model_save_path(sp))]
        pairs = []
        for n in used_save_p:
            if n != itr:
                pairs.append((itr, n))

        leader_board = {k: (0, 0) for k in used_save_p}

        for pl1, pl2 in pairs:
            self._torney(pl1, pl2, leader_board)

        print("#" * 10)
        print(f"total {len(pairs)} matches each agent plays {self.num_models_to_save}")
        best = list(sorted([(v[0], k) for k, v in leader_board.items()]))

        for value, key in best:
            wins, losses = leader_board[key]
            print(f"iteration {key} won: {wins} loss: {losses} matches")

    def run_tournaments(self, ):
        used_save_p = [sp for sp in self._save_points if os.path.exists(self._model_save_path(sp))]
        competition_pairs = list(itertools.combinations(used_save_p, 2))

        leader_board = {k: (0, 0) for k in used_save_p}

        for pl1, pl2 in competition_pairs:
            self._torney(pl1, pl2, leader_board)

        print("#" * 10)
        print(f"total {len(competition_pairs)} matches each agent plays {self.num_models_to_save}")
        best = list(sorted([(v[0], k) for k, v in leader_board.items()]))

        for value, key in best:
            wins, losses = leader_board[key]
            print(f"iteration {key} won: {wins} loss: {losses} matches")
