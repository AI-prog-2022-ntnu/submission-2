from abc import abstractmethod, ABC


class BaseState:
    '''
    shold have some kind of score display mechanism

    '''

    @abstractmethod
    def get_as_vec(self) -> [float]:
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self,
               other):
        pass


class GameBaseState(BaseState):

    @abstractmethod
    def get_as_inverted_vec(self) -> [float]:
        """
        return the board inverted i.e. player 1 pos -> player 2 pos
        :return:
        """
        pass

    @abstractmethod
    def current_player_turn(self):
        pass
