import math
import random

import numpy as np
import torch
from torch import nn

# import tensorflow as tf
# from keras import layers
# from keras import activations
# from keras import optimizers
# from keras import losses
# import keras
from abc import abstractmethod

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, GameBaseState

# TODO: generalize
# TODO: have not checked that this actually works at all
from rl_agent.mc_tree_search import MontecarloTreeSearch


class NeuralNetwork(nn.Module):
    def __init__(self,
                 inp_s,
                 out_s):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(inp_s, 100),
            nn.ReLU(),
            nn.Linear(100, out_s),
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
        return soft_m_filtered

        # soft_m = torch.where(torch.not_equal(e_pov_v.float(), 0.0), vl.float(), 0.0)
        # print(soft_m)

        # soft_m_sum = sum([math.exp(v) if z else 0 for v, z in zip(out[:], zero_inp[:])])
        # soft_m = [math.exp(v) / soft_m_sum if z else 0 for v, z in zip(out[:], zero_inp[:])]
        # return soft_m


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
        self.debug = True

    def _build_network(self):

        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        self.model = NeuralNetwork(input_s, output_s).to("cpu")
        pass
        # input_s = self.environment.get_observation_space_size()
        # output_s = self.environment.get_action_space_size()
        # input_layer = layers.Input(shape=(input_s,))
        # output_layer = layers.Dense(output_s, activation="softmax")
        #
        # x = layers.Dense(100, activation=activations.relu)(input_layer)
        #
        # x = output_layer(x)
        # x = SofMaxNonZeroInputs()(x, input_layer)
        #
        # self.model = keras.Model(
        #     inputs=[input_layer],
        #     outputs=[x]
        # )
        #
        # self.model.compile(
        #     optimizer=optimizers.adam_v2.Adam(),
        #     loss=losses.MSE,
        #     run_eagerly=True
        # )
        # tf.compat.v1.disable_eager_execution()
        # tf.keras.utils.plot_model(self.model, )

    def _train_network(self,
                       r_buffer):
        self.model.train(True)

        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters())
        for x, y in r_buffer:
            x = torch.tensor([x], dtype=torch.float)
            y = torch.tensor([y], dtype=torch.float)
            pred = self.model.forward(x)
            # print(pred)

            loss = loss_fn(pred, y)

            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.model.train(False)

        # x, y = [], []
        # for inp, target in self.train_buffer:
        #     x.append(inp)
        #     y.append(target)
        #
        # x = np.array(x)
        # y = np.array(y)
        # self.model.fit(x, y, batch_size=5, epochs=10)

    def save_model(self,
                   fp):
        pass

    def load_model(self,
                   fp):
        pass

    def _get_action_visit_map_as_target_vec(self,
                                            action_visit_map: {}):

        possible_actions = self.environment.get_action_space_list()
        visit_sum = sum(action_visit_map.values())

        ret = []
        for action in possible_actions:
            value = action_visit_map.get(action)
            if value is None:
                ret.append(0)
            else:
                ret.append(value / visit_sum)
        return ret

    def run_episode(self):
        game_done = False
        replay_buffer = []

        mcts = MontecarloTreeSearch(
            exploration_c=1,
            environment=self.environment,
            agent=self
        )

        current_state = self.environment.get_initial_state()
        while not game_done:
            if self.debug:
                print("started new episode")
            mc_visit_counts_map = mcts.mc_tree_search(num_rollouts=self.num_rollouts, root_state=current_state)
            all_action_dist = self._get_action_visit_map_as_target_vec(mc_visit_counts_map)
            replay_buffer.append((current_state.get_as_vec(), all_action_dist))

            # TODO: should probably not always be greedy
            if current_state.current_player_turn() == 0:
                target_idx = all_action_dist.index(max(all_action_dist))
            else:
                # TODO: ehhhdjshe not good
                target_idx = 0
                best_v = float("inf")
                for n, v in enumerate(all_action_dist):
                    if v < best_v and v != 0:
                        target_idx = n

            action = self.environment.get_action_space_list()[target_idx]
            next_s, r, game_done = self.environment.act(current_state, action)
            self.environment.display_state(next_s)
            current_state = next_s

        mcts.close_helper_threads()
        return replay_buffer

    def train_n_episodes(self,
                         n):

        for v in range(n):
            r_buf = self.run_episode()
            self._train_network(r_buf)

    def get_prob_dists(self,
                       state_list: [int]):
        # x = [s.get_as_vec() for s in state_list]
        x = state_list
        # print(x)
        # prob_dist = self.model.predict(x)
        # return prob_dist.tolist()

        return [random.random() if a != 0 else 0 for a in state_list[0]]

    def pick_action(self,
                    state: GameBaseState):
        """
        returns the action picked by the agen from the provided state.
        :param state:
        :param environment:
        :return:
        """

        """
        TODO: usikker på den her, mpt player 1/2. 
            spørsmålet er skal man "snu brettet" hver gang man gjør et trekk for motstanderen eller skal man bare plukke det trekke som er minst gunsig for spiller 1
            for øyeblikket tar jeg bare trekket som er minst gunsig for spiller 1 men kan være lurt og sjekke forsjellen når vi endrer dette
        """

        x = state.get_as_vec()
        # prob_dist = self.model.predict([x]).tolist()[0]
        # print(prob_dist)
        # prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))

        prob_dist = self.model.forward(torch.tensor([x], dtype=torch.float))[0].tolist()

        # print(prob_dist)

        # prob_dist = [random.random() if a == 0 else 0 for a in x]

        # TODO: implement the some action picker
        if state.current_player_turn() == 0:
            target_val = max(prob_dist)
        else:
            target_val = float("inf")
            for v in prob_dist:
                if v < target_val and v != 0:
                    target_val = v
        action_idx = prob_dist.index(target_val)

        return self.environment.get_action_space_list()[action_idx]


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
