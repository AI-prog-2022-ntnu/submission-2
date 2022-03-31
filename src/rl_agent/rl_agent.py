import math
import random
from os import path

import pandas
import torch

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BoardGameBaseState
# TODO: generalize
# TODO: have not checked that this actually works at all
from rl_agent.agent_net import BoardGameActorNeuralNetwork
from rl_agent.critic import Critic, CriticNeuralNet
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.tournament_of_progressive_policies import TOPP
from rl_agent.util import get_action_visit_map_as_target_vec, NeuralNetworkConfig


class MonteCarloTreeSearchAgent:

    def __init__(self,
                 environment: BoardGameEnvironment,
                 actor_nn_config: NeuralNetworkConfig,
                 critic_nn_config: NeuralNetworkConfig,
                 worker_thread_count,
                 worker_fork_number,
                 exploration_c,
                 topp_saves,
                 ms_tree_search_time
                 ):
        # TODO: implement layer config
        self.worker_fork_number = worker_fork_number
        self.critic_nn_config = critic_nn_config
        self.actor_nn_config = actor_nn_config
        self.ms_tree_search_time = ms_tree_search_time
        self.topp_saves = topp_saves
        self.exploration_c = exploration_c
        self.worker_thread_count = worker_thread_count
        self.environment = environment
        self.train_buffer: [([int], [float])] = []
        self.max_buffer_size = 1000

        self.model: BoardGameActorNeuralNetwork = None
        self._build_network()

        self.debug = False
        self.display = False
        self.loss_hist = []

        self.nn_input_size = self.environment.get_observation_space_size()

        self.critic = Critic(
            nn_config=self.critic_nn_config,
            environment=self.environment,
            input_size=self.nn_input_size,
        )

        self.critic.model.share_memory()

    def load_model_from_fp(self,
                           fp):
        """
        Loads the pre-trained model from the path "fp".
        """
        if fp is not None and path.exists(fp):
            output_size = self.environment.get_action_space_size()
            model = BoardGameActorNeuralNetwork.load_model(fp, self.actor_nn_config, self.environment,
                                                           self.nn_input_size, output_size)
            self.model = model

            critic_fp = fp + "_critic"
            critic_model = CriticNeuralNet.load_model(critic_fp, self.actor_nn_config, self.environment,
                                                      self.nn_input_size)
            self.critic.model = critic_model

            self.model.share_memory()
            self.critic.model.share_memory()
        else:
            print("#" * 10)
            print("path not found model not loaded")
            print("#" * 10)

    def save_actor_critic_to_fp(self,
                                fp):
        """
        Saves the actor critic.
        """
        if fp is not None:
            critic_fp = fp + "_critic"
            self.model.save_model(fp)
            self.critic.model.save_model(critic_fp)

    def _build_network(self):
        """
        Creates the board game neural network model.
        """
        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        b_size = math.floor(math.sqrt(input_s))
        self.model = BoardGameActorNeuralNetwork(self.actor_nn_config, self.environment, input_s, output_s)

    def _expand_replay_buffer(self,
                              buffer):
        """
        Expands the replay buffer list.
        """
        self.train_buffer = [*self.train_buffer, *buffer]
        new_b_len = len(self.train_buffer)

        if new_b_len > self.max_buffer_size:
            self.train_buffer = self.train_buffer[(new_b_len - self.max_buffer_size):]

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
        """
        Does the Monte Carlo tree rollout and picks the action to do.
        """
        # do the montecarlo tree rollout
        mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(root_state=current_state)

        # convert the vec to target dist
        all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)

        # pick the action to do
        target_idx = all_action_dist.index(max(all_action_dist))

        return self.environment.get_action_space()[target_idx]

    def _take_game_move(self,
                        current_state,
                        mcts,
                        replay_buffer,
                        critic_train_set):

        # do the montecarlo tree rollout
        mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(root_state=current_state)

        # convert the vec to target dist
        all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)
        print(f'Target distance: {all_action_dist}')

        # put the target dist in the replay buffer
        replay_buffer.append((current_state, mc_visit_counts_map))

        # pick the action to do
        target_idx = all_action_dist.index(max(all_action_dist))

        action = self.environment.get_action_space()[target_idx]
        next_s, r, game_done = self.environment.act(current_state, action)

        if self.display:
            self.environment.display_state(next_s)

        if critic_train_map is not None:
            for state, val in critic_train_map.items():
                critic_train_set.append((state, val))

        return next_s, r, game_done

    def run_episode(self,
                    player_2=None,
                    flip_start=False,
                    train_critic=False,
                    ):
        """
        Runs an episode of the game.
        """
        game_done = False
        replay_buffer = []
        critic_train_set = []

        mcts = MontecarloTreeSearch(
            exploration_c=self.exploration_c,
            environment=self.environment,
            agent=self,
            worker_thread_count=self.worker_thread_count,
            worker_fork_number=self.worker_fork_number
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
            actual_has_won = self.environment.get_winning_player_id(current_state)
            if actual_has_won is not None:
                if actual_has_won == 0:
                    r = 1
                else:
                    r = -1
                game_done = True

        mcts.close_helper_threads()
        if train_critic:
            self.critic.train_from_buffer(critic_train_set)

        did_win = self.environment.get_winning_player(current_state) == 0

        torch.set_num_threads(1)
        return replay_buffer, did_win

    def human_move(self,
                   current_state):
        """
        Waits for an input from the human player and checks if it is valid or not.
        The input should be two numbers for example: 5 5 .
        """
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
        """
        Plays a game against a human player in the terminal.
        """
        self.run_episode(
            player_2=self.human_move,
            train_critic=False,
            flip_start=False,
        )

    def run_topp(self,
                 n,
                 num_games):
        """
        Runs the Tournament of Progressive Policies.
        """
        topp = TOPP(
            total_itrs=n,
            actor_nn_config=self.actor_nn_config,
            num_games_in_matches=num_games,
            num_models_to_save=self.topp_saves,
            environment=self.environment)

        topp.run_tournaments()

    def run_self_training(self,
                          num_games: int,
                          topp_saves,
                          lr: float,
                          discount: float):
        """
        Trains the model.
        """
        topp = TOPP(
            total_itrs=num_games,
            actor_nn_config=self.actor_nn_config,
            num_games_in_matches=200,
            num_models_to_save=topp_saves,
            environment=self.environment)

        topp.training_torney(
            actor_model=self.model,
            num_games=num_games,
            lr=lr,
            discount=discount

        )

    def train_n_episodes(self,
                         n,
                         fp=None,
                         games_in_topp_matches=500):
        """
        Trains a number of episodes.
        """

        topp = TOPP(
            total_itrs=n,
            actor_nn_config=self.actor_nn_config,
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
            self._expand_replay_buffer(r_buf)
            if win:
                win_count.append(1)
            else:
                win_count.append(0)

            loss = self.model.train_from_state_buffer(self.train_buffer)
            # self.loss_hist.append(np.mean(loss))
            self.loss_hist.extend(loss)

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

    def pick_action(self,
                    state: BoardGameBaseState,
                    use_prob_not_max=False):

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
        prob_dist = self.model.get_probability_distribution([state])[0]

        # if torch.any(torch.isnan(prob_dist)):
        #     print("na in dist: inp: ", x)
        #     print("na in diat: prob dist: ", prob_dist)
        #     print("na in diat: params: ", self.model.parameters())
        #     # prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))[0]
        #     # print(prob_dist)
        #
        #     # print(self.model.network(torch.tensor([x], dtype=torch.float)))
        #     pass

        # if all probs are zero pick a random value
        # TODO: implement the some action picker
        if sum(prob_dist) == 0:
            action = random.choice(self.environment.get_valid_actions(state))
        else:
            action_space = self.environment.get_action_space()
            if use_prob_not_max:
                action_idx = random.choices(range(len(action_space)), weights=prob_dist)[0]
            else:
                target_val = max(prob_dist)
                action_idx = prob_dist.index(target_val)

            action = action_space[action_idx]
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
