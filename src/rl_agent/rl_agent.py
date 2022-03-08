import itertools
import math
import random
import os

import numpy as np
import torch
from torch import nn

from abc import abstractmethod
from os import path
from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, GameBaseState

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
                 ):
        # TODO: implement layer config
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

        passes_over_data = 200
        batch_size = 5
        train_itrs = math.ceil((len(r_buffer) * passes_over_data) / batch_size)
        # print(train_itrs)

        inp_x, inp_y = [], []
        for state, v_count_map in r_buffer:
            if state.current_player_turn == 0:
                inp_x.append(state.get_as_vec())
                inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=False))
            else:
                inp_x.append(state.get_as_inverted_vec())
                inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=True))

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
            player_2=self.play_against_human,
            train_critic=False,
            flip_start=False,
        )

    def train_n_episodes(self,
                         n,
                         fp=None):

        topp = TOPP(
            total_itrs=n,
            game_rollouts=self.num_rollouts,
            num_games_in_matches=1,
            num_models_to_save=3,
            env=self.environment
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

            topp.register_policy(self, v)

        print()
        self.save_actor_critic_to_fp(fp)

    def get_prob_dists(self,
                       state_list: []):

        x = [state.get_as_vec() if state.current_player_turn() == 0 else state.get_as_inverted_vec() for state in state_list]

        prob_dist = self.model.forward(torch.tensor(x, dtype=torch.float))

        prob_dist = prob_dist.tolist()

        return prob_dist  # [random.random() if a != 0 else 0 for a in state_list[0]]

    def pick_action(self,
                    state: GameBaseState,
                    get_prob_not_max=False):

        """
        TODO: usikker på den her, mpt player 1/2. 
            spørsmålet er skal man "snu brettet" hver gang man gjør et trekk for motstanderen eller skal man bare plukke det trekke som er minst gunsig for spiller 1
            for øyeblikket tar jeg bare trekket som er minst gunsig for spiller 1 men kan være lurt og sjekke forsjellen når vi endrer dette
        """

        if state.current_player_turn() == 0:
            x = state.get_as_vec()
            pass
        else:
            x = state.get_as_inverted_vec()
        prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))[0]

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

            if state.current_player_turn() == 0:
                action = self.environment.get_action_space_list()[action_idx]
            else:
                action_inv = self.environment.get_action_space_list()[action_idx]
                action = (action_inv[1], action_inv[0])

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
                 env):
        self.game_rollouts = game_rollouts
        self.env = env
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
                        itr):
        if itr in self._save_points:
            model.save_actor_critic_to_fp(self._model_save_path(itr))

    def run_tournaments(self):
        competition_pairs = itertools.combinations(self._save_points, 2)
        leader_board = {}

        leader_board.setdefault(0)

        for pl1, pl2 in competition_pairs:
            for _ in range(self.num_games_in_matches):
                player1 = MonteCarloTreeSearchAgent(
                    num_rollouts=self.game_rollouts,
                    environment=self.env,
                    worker_thread_count=10,
                    exploration_c=math.sqrt(2)
                )

                player2 = MonteCarloTreeSearchAgent(
                    num_rollouts=self.game_rollouts,
                    environment=self.env,
                    worker_thread_count=10,
                    exploration_c=math.sqrt(2)
                )

                player1.load_model_from_fp(self._model_save_path(pl1))
                player2.load_model_from_fp(self._model_save_path(pl2))

                pl1_start = random.random() > 0.5

                # manually make the player 2 mcts

                player_2_mcts = MontecarloTreeSearch(
                    exploration_c=player2.exploration_c,
                    environment=player2.environment,
                    agent=player2,
                    worker_thread_count=player2.worker_thread_count
                )

                p1_win = player1.run_episode(
                    player_2=lambda state: player2._take_game_move(
                        current_state=state,
                        mcts=player_2_mcts,
                        replay_buffer=[],
                        critic_train_set=[]
                    ),
                    flip_start=pl1_start,
                    train_critic=False
                )

                player_2_mcts.close_helper_threads()

                if p1_win:
                    leader_board[pl1] += 1
                else:
                    leader_board[pl2] += 1

        print("#" * 10)
        print(f"total {len(competition_pairs)} matches each agent plays {len(self.num_models_to_save)}")
        best = list(sorted([(v, k) for k, v in leader_board.items()]))

        for val, iter in best:
            print(f"iter {iter} won {val} matches")
