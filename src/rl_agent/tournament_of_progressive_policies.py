import itertools
import math
import os
import random

import numpy as np

from os import path

# because the python import system is fucking usless this has to sit at the botom here
import torch

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
from rl_agent.agent_net import BoardGameActorNeuralNetwork
from rl_agent.util import NeuralNetworkConfig


class TOPP:

    def __init__(self,
                 total_itrs,
                 num_models_to_save,
                 num_games_in_matches,

                 actor_nn_config: NeuralNetworkConfig,
                 environment: BoardGameEnvironment):
        self.actor_nn_config = actor_nn_config
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
                        model,
                        itr,
                        run_tornament=False):
        if itr in self._save_points:
            model.save_actor_critic_to_fp(self._model_save_path(itr))

            if run_tornament:
                self._single_model_tournament(itr)

    def _tornament_model_pick_action(self,
                                     state,
                                     model,
                                     ):
        # if state.current_player_turn() == 0:
        #     x = state.get_as_vec()
        # else:
        #     x = state.get_as_inverted_vec()
        x = state.get_as_vec()

        prob_dist = model.get_probability_distribution([state])[0]

        if sum(prob_dist) == 0:
            action = random.choice(self.environment.get_valid_actions(state))
        else:
            action_space = self.environment.get_action_space()
            idx = random.choices(range(len(action_space)), weights=prob_dist)[0]

            action = action_space[idx]

            # target_val = max(prob_dist)
            # action_idx = prob_dist.index(target_val)
            # action = self.environment.get_action_space()[action_idx]

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

        p1_score = 0
        p2_score = 0
        model_1 = BoardGameActorNeuralNetwork.load_model(self._model_save_path(pl1), self.actor_nn_config, self.environment, input_s, output_s)
        model_2 = BoardGameActorNeuralNetwork.load_model(self._model_save_path(pl2), self.actor_nn_config, self.environment, input_s, output_s)
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
                                                                                                                                              pl2) + " " * 20, end="\r")
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
            print("iteration {:<5} win /lose: \u001b[32m{:<6}\u001b[0m / \u001b[31m{:<6}\u001b[0m".format(key, wins, losses))

    def run_tournaments(self, ):
        used_save_p = [sp for sp in self._save_points if os.path.exists(self._model_save_path(sp))]
        competition_pairs = list(itertools.combinations(used_save_p, 2))

        leader_board = {k: (0, 0) for k in used_save_p}

        for pl1, pl2 in competition_pairs:
            self._torney(pl1, pl2, leader_board)

        print("#" * 10)
        print(f"total {len(competition_pairs)} matches each agent plays {self.num_models_to_save}")
        best = list(reversed(sorted([(v[0], k) for k, v in leader_board.items()])))

        for value, key in best:
            wins, losses = leader_board[key]
            print("iteration {:<5} win /lose: \u001b[32m{:<6}\u001b[0m / \u001b[31m{:<6}\u001b[0m".format(key, wins, losses))
