from abc import abstractmethod, ABC


class BaseState:
    '''
    shold have some kind of score display mechanism

    '''

    @abstractmethod
    def get_as_vec(self) -> [float]:
        """
        returns the current state as a vector of the same size as the env observation space
        :return:
        """
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self,
               other):
        pass


class BoardGameBaseState(BaseState):
    @abstractmethod
    def get_as_inverted_vec(self) -> [float]:
        """
        returns the the inverted state as a vector of the same size as the env observation space.
        i.e. player 1's representation is switched to look like player 2's

        this is usefull for training.
        :return:
        """
        pass

    @abstractmethod
    def current_player_turn(self):
        pass

    @abstractmethod
    def change_turn(self):
        pass
