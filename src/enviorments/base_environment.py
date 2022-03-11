from abc import abstractmethod, ABC

from enviorments.base_state import BaseState, BoardGameBaseState

"""
The basis for an environment for the rl agent to act in
"""


class BaseEnvironment:

    @abstractmethod
    def act(self,
            state,
            action,
            inplace=False) -> (BaseState, int, bool):
        """
        :param state: current state
        :param action: the action to take
        :param inplace: whether to change the state inplace or copy it. defaults to false
        :return: (the new state of the environment, the reward for the step, flag indicating whether the episode is done)
        """
        pass

    @abstractmethod
    def reverse_move(self,
                     state,
                     action):
        """
        reverse the move inplace and return the state to its previos state
        :param state:
        :param action:
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
    def get_initial_state(self) -> BaseState:
        """
        Returns the initial state of the environment
        :return:
        """
        pass

    @abstractmethod
    def is_state_won(self,
                     state) -> bool:
        """
        Returns if the provided state is won
        :param state:
        :return:
        """
        pass

    @abstractmethod
    def winning_player_id(self,
                          state):
        """
        Returns the id of the player who has won the current state.
        if the state is not won null is returned
        :param state:
        :return:
        """

    @abstractmethod
    def display_state(self,
                      state):
        """
        Displays the provided state
        :param state:
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

    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Returns the size of the action space for the environment
        :return:
        """
        pass

    @abstractmethod
    def get_action_space_list(self) -> []:
        """
        Returns a list of all the possible actions in the environment
        :return:
        """
        pass

    @abstractmethod
    def get_inverted_action_space_list(self,
                                       state: BaseState) -> []:
        pass

    @abstractmethod
    def get_valid_action_space_list(self,
                                    state: BaseState) -> [bool]:
        """
        Returns a list of bool values indicating witch of the get_action_space_list() actions are available
        :return:
        """
        pass


class BoardGameEnvironment(BaseEnvironment):

    @abstractmethod
    def game_has_reversible_moves(self) -> bool:
        pass

    @abstractmethod
    def reverse_move(self,
                     state: BoardGameBaseState,
                     action) -> BoardGameBaseState:
        """

        if the game has reversible moves return the state with the move reversed
        Args:
            state:
            action:

        Returns:

        """

    @abstractmethod
    def get_state_winning_move(self,
                               state: BoardGameBaseState):
        pass
