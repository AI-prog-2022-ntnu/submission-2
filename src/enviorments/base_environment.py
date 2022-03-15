from abc import abstractmethod, ABC

from enviorments.base_state import BaseState, BoardGameBaseState

"""
The basis for an environment for the rl agent to act in
"""


class BaseEnvironment:

    #
    #   env interaction
    #

    @abstractmethod
    def act(self,
            state,
            action,
            inplace=False) -> (BaseState, int, bool):
        """
        :param state: current state
        :param action: the action to take
        :param inplace: wheter to change the state inplace or copy it. defaults to false
        :return: (the new state of the environment, the reward for the step, flag indicating whether the episode is done)
        """
        pass

    #
    #   env info
    #
    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Returns the size of the action space for the environment
        :return:
        """
        pass

    @abstractmethod
    def get_action_space(self) -> []:
        """
        Returns a list of all the possible actions in the environment
        :return:
        """
        pass

    @abstractmethod
    def get_observation_space_size(self) -> int:
        """
        Returns the size of the observation space for the environment
        :return:
        """
        pass

    #
    #   state info
    #

    @abstractmethod
    def get_initial_state(self) -> BaseState:
        """
        Returns the initial state of the environment
        :return:
        """
        pass

    @abstractmethod
    def get_valid_actions(self,
                          state):
        """
        Returns a list of the valid actions from the provided state

        :param state:
        :return:
        """
        pass

    @abstractmethod
    def get_valid_action_space_list(self,
                                    state: BaseState) -> [bool]:
        """
        Returns a list of bool values indicating what actions in the get_action_space_list() actions are available
        this is usefull for interacting with the neural network
        :return:
        """
        pass

    @abstractmethod
    def is_state_won(self,
                     state) -> bool:
        """
        Returns if the provided state is won for the current agent
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def display_state(self,
                      state):
        """
        Displays the provided state
        :param state:
        :return:
        """
        pass


class BoardGameEnvironment(BaseEnvironment):

    @abstractmethod
    def get_state_winning_move(self,
                               state: BoardGameBaseState):
        """
        in some senarios one can quickly calculate if any of the next moves are a winning one. if this is possible
        and sutch a move exist it will be returned, if not None is returned
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def get_winning_player(self,
                           state):
        """
        returns the winning player in the provided state
        :return:
        """

    #
    #   move reversal
    #
    @abstractmethod
    def game_has_reversible_moves(self) -> bool:
        """
        given: s1 -a1-> s2
        in some games it is possible to navigate back to s1 from s2 given you know a1. in other games information
        is lost after some actions. If the moves are reversible more effective tree traversal methods may be used
        :return:
        """
        pass

    @abstractmethod
    def reverse_move(self,
                     state: BoardGameBaseState,
                     action) -> BoardGameBaseState:
        """
        if the game has reversible moves return the state with the move reversed

        :param state:
        :param action:
        :return:
        """

        pass

    #
    # board inversal
    #

    @abstractmethod
    def invert_observation_space_vec(self,
                                     observation_space_vec):
        """
        invert the observation space vector. usually the input of the neural net
        :param observation_space_vec:
        :return:
        """
        pass

    @abstractmethod
    def invert_action_space_vec(self,
                                action_space_vec):
        """
        invert the action space vector, usually the output of the neuralnet
        :param observation_space_vec:
        :return:
        """
        pass
