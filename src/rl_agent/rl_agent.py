import math
import random

import numpy as np
import torch
from torch import nn

from abc import abstractmethod
from os import path
from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, GameBaseState

# TODO: generalize
# TODO: have not checked that this actually works at all
from rl_agent.critic import Critic, CriticNeuralNet
from rl_agent.mc_tree_search import MontecarloTreeSearch
from rl_agent.util import generate_batch, get_action_visit_map_as_target_vec


class NeuralNetwork(nn.Module):
    def __init__(self,
                 inp_s,
                 out_s):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inp_s, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, out_s),
            # nn.ReLU()  # <- DONT CHANGE
            nn.Tanh()
            # nn.Softmax()
        )

    def forward(self,
                x):
        zero_inp = torch.where(x == 0, 1, 0)
        out: torch.Tensor = self.network(x)

        e_pov_v = torch.exp(out)
        soft_m_sum = torch.sum(input=torch.multiply(e_pov_v, zero_inp), dim=1)

        s = e_pov_v.size()
        s1, s2 = s[0], s[1]

        soft_m = torch.div(e_pov_v, soft_m_sum.unsqueeze(1).expand(s1, s2)).float()
        soft_m_filtered = torch.multiply(soft_m, zero_inp)

        # used_vals = torch.multiply(out, zero_inp)
        # soft_m_sum = torch.sum(input=used_vals, dim=1)
        # if soft_m_sum.float().sum() == 0:
        #     soft_m = used_vals
        # else:
        #     s = out.size()
        #     s1, s2 = s[0], s[1]
        #     soft_m = torch.div(used_vals, soft_m_sum.unsqueeze(1).expand(s1, s2)).float()
        # soft_m_sum = torch.where(soft_m != 0.0, soft_m_sum, 1.0)

        # soft_m_filtered = torch.multiply(soft_m, used_vals)
        # print("#############")
        # print("zero inp mask", zero_inp)
        # print("out vec", out)
        # print("x", x)
        # print("soft m filtered ", soft_m_filtered)
        # print("soft m sum ", soft_m_sum)
        # # print(soft_m_sum.unsqueeze(1).expand(s1, s2))
        # print(soft_m)
        # print("#############")
        return soft_m_filtered

        # if torch.any(torch.isnan(soft_m_filtered)):
        # print("#############")
        # print(x)
        # print(u)
        # print(soft_m)
        # print(out)
        # print("#############")
        # return soft_m_filtered

    def save_model(self,
                   fp):
        torch.save(self.state_dict(), fp)

    @staticmethod
    def load_model(fp,
                   *args):
        model = NeuralNetwork(*args)
        model.load_state_dict(torch.load(fp))
        model.eval()
        return model


class MonteCarloTreeSearchAgent:

    def __init__(self,
                 environment: BaseEnvironment,
                 num_rollouts: int):
        # TODO: implement layer config
        self.num_rollouts = num_rollouts
        self.environment = environment
        self.train_buffer: [([int], [float])] = []
        self.model: NeuralNetwork = None
        self._build_network()
        self.debug = False
        self.display = False
        self.loss_hist = []

        input_s = self.environment.get_observation_space_size()
        self.critic = Critic(input_s)

        self.model.share_memory()
        self.critic.model.share_memory()

    def load_model_from_fp(self,
                           fp):
        if path.exists(fp):
            input_s = self.environment.get_observation_space_size()
            output_s = self.environment.get_action_space_size()
            model = NeuralNetwork.load_model(fp, input_s, output_s)
            self.model = model

            critic_fp = fp + "_critic"
            critic_model = CriticNeuralNet.load_model(critic_fp, input_s)
            self.critic.model = critic_model

            self.model.share_memory()
            self.critic.model.share_memory()
        else:
            print("#" * 10)
            print("path not found model not loaded")
            print("#" * 10)

    def _save_model_to_fp(self,
                          fp):
        if fp is not None:
            critic_fp = fp + "_critic"
            self.model.save_model(fp)
            self.critic.model.save_model(critic_fp)

    def _build_network(self):

        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        self.model = NeuralNetwork(input_s, output_s).to("cpu")

    def _train_network(self,
                       r_buffer):
        self.model.train(True)
        torch.set_printoptions(profile="full", linewidth=1000)
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.model.parameters())

        passes_over_data = 50
        batch_size = 5
        train_itrs = math.ceil((len(r_buffer) * passes_over_data) / batch_size)
        print(train_itrs)

        inp_x, inp_y = [], []
        for state, v_count_map in r_buffer:
            if state.current_player_turn == 0:
                inp_x.append(state.get_as_vec())
                inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=False))
            else:
                inp_x.append(state.get_as_inverted_vec())
                inp_y.append(get_action_visit_map_as_target_vec(self.environment, action_visit_map=v_count_map, invert=True))

        for _ in range(train_itrs):
            x, y = generate_batch(inp_x, inp_y, batch_size)

            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            pred = self.model.forward(x)
            # print("y:    ", y)
            # print("pred: ", pred)
            # print(pred)

            loss: torch.Tensor = loss_fn(pred, y)
            self.loss_hist.append(loss.item())

            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.model.train(False)

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

    def run_episode(self,
                    flip_start=False):
        game_done = False
        replay_buffer = []
        critic_train_set = []

        mcts = MontecarloTreeSearch(
            exploration_c=1,
            environment=self.environment,
            agent=self,
            worker_thread_count=10
        )
        mcts.debug = self.debug

        current_state = self.environment.get_initial_state()
        if flip_start:
            current_state.change_turn()

        torch.set_num_threads(10)
        while not game_done:
            if self.debug:
                print("started new episode")
            mc_visit_counts_map, critic_train_map = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)
            all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)
            replay_buffer.append((current_state, mc_visit_counts_map))

            # TODO: should probably not always be greedy
            if current_state.current_player_turn() == 0:
                target_idx = all_action_dist.index(max(all_action_dist))
            else:
                target_idx = all_action_dist.index(max(all_action_dist))

            action = self.environment.get_action_space_list()[target_idx]
            next_s, r, game_done = self.environment.act(current_state, action)
            current_state = next_s

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
                # else:
                #     critic_train_set.append((vec.get_as_inverted_vec(), val))

        ## critic ##
        # res = self.critic.get_states_value(x)
        # avg_critic_error = np.mean([math.dist((r,), (yv,)) for r, yv in zip(res, y)])
        # print("avg critic error ", avg_critic_error)
        self.critic.train_network(critic_train_set)
        ## critic ##

        did_win = r == 1
        mcts.close_helper_threads()

        torch.set_num_threads(1)
        return replay_buffer, did_win

    def play_against_human(self):
        game_done = False

        mcts = MontecarloTreeSearch(
            exploration_c=1,
            environment=self.environment,
            agent=self,
            worker_thread_count=10
        )

        current_state = self.environment.get_initial_state()
        while not game_done:

            # print(mc_visit_counts_map)
            # print(all_action_dist)
            # TODO: should probably not always be greedy
            if current_state.current_player_turn() == 0:
                mc_visit_counts_map, _ = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)
                all_action_dist = get_action_visit_map_as_target_vec(self.environment, mc_visit_counts_map)
                target_idx = all_action_dist.index(max(all_action_dist))
                action = self.environment.get_action_space_list()[target_idx]
                next_s, r, game_done = self.environment.act(current_state, action)
            else:
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

            current_state = next_s

            self.environment.display_state(next_s)

        mcts.close_helper_threads()

    def train_n_episodes(self,
                         n,
                         fp=None):

        win_count = [0 for _ in range(50)]
        for v in range(n):
            flip = random.random() > 0.5
            r_buf, win = self.run_episode(flip)
            if win:
                win_count.append(1)
            else:
                win_count.append(0)
            self._train_network(r_buf)

            self.print_progress(v, n, win_count)
            if self.display:
                print()

            if v % 10 == 0:
                self._save_model_to_fp(fp)

        print()
        self._save_model_to_fp(fp)

    def get_prob_dists(self,
                       state_list: [int]):
        # x = [s.get_as_vec() for s in state_list]
        x = state_list
        # print(x)
        # prob_dist = self.model.predict(x)
        # return prob_dist.tolist()

        return [random.random() if a != 0 else 0 for a in state_list[0]]

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
            print(x)
            print(self.model.parameters())
            prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))[0]
            print(prob_dist)

            print(self.model.network(torch.tensor([x], dtype=torch.float)))
            pass

        prob_dist = prob_dist.tolist()

        # print(prob_dist)

        # prob_dist = [random.random() if a == 0 else 0 for a in x]

        if sum(prob_dist) == 0:
            pass

        # TODO: implement the some action picker
        # if state.current_player_turn() == 0:
        if get_prob_not_max:
            target_val = random.choices(range(len(prob_dist)), prob_dist)
            if target_val == 0:
                raise Exception("wat")
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
            print(prob_dist)
            print(action)
            print(state.get_as_vec())
            print(self.model.network(torch.tensor([x], dtype=torch.float)))
        return action


"""
behavior policy == target policy ==  Default policyu-> i critic
ansvarilig for og velge moves

Tree policy -> ?
kontrolerer hvordan og rulle ut treet i søk





1. i s = save interval for ANET (the actor network) parameters
2. Clear Replay Buffer (RBUF)
3. Randomly initialize parameters (weights and biases) of ANET
4. For g a in number actual games:
(a) Initialize the actual game board (B a ) to an empty board.
(b) s init ← starting board state
(c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents s init
(d) While B a not in a final state:
• Initialize Monte Carlo game board (B mc ) to same state as root.
• For g s in number search games:
– Use tree policy P t to search from root to a leaf (L) of MCT. Update B mc with each move.
– Use ANET to choose rollout actions from L to a final state (F). Update B mc with each move.
– Perform MCTS backpropagation from F to root.
• next g s
• D = distribution of visit counts in MCT along all arcs emanating from root.
• Add case (root, D) to RBUF
• Choose actual move (a*) based on D
• Perform a* on root to produce successor state s*
• Update B a to s*
• In MCT, retain subtree rooted at s*; discard everything else.
• root ← s*
(e) Train ANET on a random minibatch of cases from RBUF
(f) if g a modulo i s == 0:
• Save ANET’s current parameters for later use in tournament play.
"""
