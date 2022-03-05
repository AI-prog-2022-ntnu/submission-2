from abc import abstractmethod, ABC

from enviorments.base_state import BaseState

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
        :param inplace: wheter to change the state inplace or copy it. defaults to false
        :return: (the new state of the environment, the reward for the step, flag indicating whether the episode is done)
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
    def get_valid_action_space_list(self,
                                    state: BaseState) -> [bool]:
        """
        Returns a list of bool values indicating witch of the get_action_space_list() actions are available
        :return:
        """
        pass
