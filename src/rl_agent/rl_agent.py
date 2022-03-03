import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from abc import abstractmethod


from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState


class MonteCarloTreeSearchAgent:

    def __init__(self, environment: BaseEnvironment):
        #TODO: implement layer config
        self.environment = environment


    def _build_network(self):
        input_s = self.environment.get_observation_space_size()
        output_s = self.environment.get_action_space_size()
        input_layer = layers.



    def pick_action(self, state: BaseState, environment: BaseEnvironment):
        """
        returns the action picked by the agen from the provided state.
        :param state:
        :param environment:
        :return:
        """
        pass














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