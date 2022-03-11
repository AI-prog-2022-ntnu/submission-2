import random

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState

import copy


class NimGameState(BaseState):
    """
    Nim:
              |
           |  |  |
        |  |  |  |  |
     |  |  |  |  |  | |
    is represented as [[1],[1,1,1],[1,1,1,1,1], [1,1,1,1,1,1,1,]]
    """

    def __init__(self, board, start_number):
        self.nim_pieces = board
        self.players_turn = 0
        self.start_pieces = start_number
        self.current_nr = start_number
        self.players_turn = 0

    def change_turn(self):
        self.players_turn = (self.players_turn + 1) % 2

    def current_player_turn(self):
        return self.players_turn

    def get_as_vec(self) -> [float]:
        z_padded = self._get_zero_padded()

        actions = list(range(1, self.current_nr + 1))
        for nr in actions:
            z_padded[nr-1] = nr
        return z_padded

    def _get_zero_padded(self):
        """
        Fills a list with all possible values as zeroes.
        """
        zero_padded_state = []

        for nr in range(1, self.start_pieces + 1):
            zero_padded_state.append(0)
        return zero_padded_state

    def get_as_inverted_vec(self) -> [float]:
        pass

    def __hash__(self):
        return hash(str(self.get_as_vec()))

    def __eq__(self, other):
        return hash(self) == hash(other)


def _make_nim_board(size: int):
    ret = []
    for nr in range(size):
        ret.append(1)
    return ret


class NimGameEnvironment(BaseEnvironment):

    def __init__(self, nr_pieces, max_nr_moved):
        self.nr_pieces = nr_pieces
        self.max_nr_moved = max_nr_moved
        self.min_nr_moved = 1
        self._action_space_list = self.get_valid_actions(self.get_initial_state())

        self.inverted_action_space_list = []

    def game_has_reversible_moves(self) -> bool:
        return False

    def reverse_move(self, state: NimGameState, action: int):
        """No need to reverse"""
        pass

    def winning_player_id(self, state: NimGameState):
        if state.current_nr <= 0:
            return state.players_turn

    def get_observation_space_size(self) -> int:
        return self.nr_pieces ** 2

    def get_action_space_size(self) -> int:
        return self.nr_pieces ** 2

    def get_action_space_list(self) -> []:
        return self._action_space_list

    def get_inverted_action_space_list(self, state: NimGameState) -> []:
        # Todo - check if this needs to be implemented
        pass

    def get_valid_action_space_list(self, state: NimGameState) -> [bool]:
        state_act = self.get_valid_actions(state)
        all_act = self.get_action_space_list()
        return [act in state_act for act in all_act]

    def act(self, state: NimGameState, action: int, inplace=False) -> (BaseState, int, bool):
        pieces_left = state.nim_pieces.count(1)
        self._check_for_action_error(action, pieces_left)  # Checks for wrong amounts
        print(f'Pieces left: {pieces_left}')
        new_state = self._substract_pieces(state, pieces_left)

        done = self.is_state_won(new_state)
        reward = 0
        if done and new_state.players_turn == 0:
            reward = 1
        elif done and new_state.players_turn == 1:
            reward = -1

        return new_state, reward, done

    def _check_for_action_error(self, action, nr_pieces_left):
        if action > nr_pieces_left:
            raise Exception("Subtracted more than remaining amount")
        if action > self.max_nr_moved:
            raise Exception("Invalid amount. The amount subtracted exceeds the number maximum number.")
        elif action < self.min_nr_moved:
            raise Exception(
                "Invalid amount. The amount subtracted is less than minimum pieces allowed.")

    def get_valid_actions(self, state: NimGameState):
        # Todo - Check when not tired
        if state.current_nr > self.max_nr_moved:
            actions = list(range(self.min_nr_moved, self.max_nr_moved + 1))
        elif state.current_nr == self.min_nr_moved:
            actions = [self.min_nr_moved]
        else:
            actions = list(range(self.min_nr_moved, state.current_nr + 1))
        return actions

    def get_initial_state(self) -> NimGameState:
        return NimGameState(_make_nim_board(self.nr_pieces), self.nr_pieces)

    def is_state_won(self, state) -> bool:
        return state.current_nr <= 0

    def display_state(self, state: NimGameState):
        pass

    def _substract_pieces(self, state: NimGameState, amount: int):
        next_state = copy.deepcopy(state)
        next_state.current_nr -= amount
        next_state.change_turn()
        return next_state
