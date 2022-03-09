import random

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState

import copy
import marshal


class colors:
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    RESET = '\u001b[0m'
    # RESET = '\033[0;0m'


def invert(x):
    if x == -1:
        return 1
    if x == 1:
        return -1
    else:
        return 0


class HexBoardGameState(BoardGameBaseState):
    '''
    boards are represented as vectors turned 34 degrees i.e.
        a
      b   c
    d   e   f
      g   h
        i

    is represented as [[a,b,d],[c,e,g],[f,h,i]]

    '''

    def __init__(self,
                 board):
        self.hex_board = board
        self.board_hash = hash(marshal.dump(board))
        self.players_turn = 0

    def change_turn(self):
        self.players_turn = (self.players_turn + 1) % 2

    def current_player_turn(self):
        return self.players_turn
        pass

    def get_as_vec(self) -> [float]:
        ret = []
        for r in self.hex_board:
            ret.extend(r)
        return ret

    def get_as_inverted_vec(self) -> [float]:
        ret = []
        inverted_board = copy.deepcopy(self.hex_board)
        axis_size = len(self.hex_board)

        for x in range(axis_size):
            for y in range(axis_size):
                inverted_board[x][y] = invert(self.hex_board[y][x])

        for r in inverted_board:
            ret.extend(r)

        return ret

    def __hash__(self):
        return self.board_hash

    def __eq__(self,
               other):
        return hash(self) == hash(other)


def _make_hex_board(size: int):
    ret = []
    for _ in range(size):
        # ret.append([random.randint(-1, 1) for _ in range(size)])
        ret.append([0 for _ in range(size)])

    return ret


def _terminal_display_hex_board(hex_board: [[int]]):
    max_size = len(hex_board[0])
    width = ((max_size - 1) * 2) + 1
    skewed_board = [[] for _ in range(width)]

    # build the skewed table
    cur_height = 0
    for diagonal in hex_board:
        for n in range(max_size):
            skewed_board[cur_height + n].append(diagonal[n])
        cur_height += 1

    # pad the table for display
    for n in range(width):
        if n < max_size:
            to_pad = (max_size) - (n + 1)
        else:
            to_pad = n - max_size + 1

        to_pad = to_pad + 1
        skewed_board[n] = [None for _ in range(to_pad)] + skewed_board[n] + [None for _ in range(to_pad)]

    for n, row in enumerate(skewed_board):
        disp_r = ""
        padd_r = ""
        last = None
        flip = n >= max_size - 1

        a_sym = colors.GREEN + "● " + colors.RESET
        b_sym = colors.RED + "● " + colors.RESET
        n_sym = "○ "

        for sym in row:
            sufx = "──"
            if sym is None:
                if last is not None:
                    padd_r.rstrip("╱")
                    disp_r = disp_r[:-len(sufx)]
                    if flip:
                        padd_r = padd_r[:-len("╱  ")]

                disp_r += "  "
                padd_r += "  "
            else:
                padd_addition = "╱╲  "

                if last is None:
                    padd_r.rstrip("  ")
                    if flip:
                        padd_addition = " ╲  "
                padd_r += padd_addition

                if sym == 0:
                    disp_r += (n_sym + sufx)
                elif sym == 1:
                    disp_r += (a_sym + sufx)
                elif sym == -1:
                    disp_r += b_sym + sufx

            last = sym
            pass

        if n == 0:
            disp_r = colors.RED + "R" + colors.RESET + disp_r[1:-1] + colors.GREEN + "G" + colors.RESET
        elif n == len(skewed_board) - 1:
            disp_r = colors.GREEN + "G" + colors.RESET + disp_r[1:-1] + colors.RED + "R" + colors.RESET
        print(disp_r)
        if n != width - 1:
            print(padd_r)


def _get_connected_cells(x: int,
                         y: int,
                         board: [[int]]) -> [(int), (int)]:
    # idx_up_r = (x - 1, y)
    # idx_r = (x - 1, y - 1)
    # idx_down_r = (x, y + 1)
    # idx_down_l = (x + 1, y)
    # idx_l = (x + 1, y - 1)
    # idx_up_l = (x, y - 1)

    # TODO: some of theese may be wrong
    idx_up_r = (x, y - 1)
    idx_r = (x + 1, y - 1)
    idx_down_r = (x + 1, y)
    idx_down_l = (x, y + 1)
    idx_l = (x - 1, y + 1)
    idx_up_l = (x - 1, y)
    tests = [idx_up_r, idx_r, idx_down_r, idx_down_l, idx_l, idx_up_l]

    valid = []
    upper_b = len(board[0])
    # print()
    for test in tests:
        # print(test)
        if test[0] < upper_b and test[0] >= 0 and test[1] < upper_b and test[1] >= 0:
            valid.append(test)

    return valid


def _win_traverse(board: [[int]],
                  checked: [(int, int)],
                  current_node: (int, int),
                  term_nodes: [(int, int)]):
    x, y = current_node[0], current_node[1]
    checked.append(current_node)
    cur_node_v = board[x][y]
    connected = _get_connected_cells(x, y, board)
    suc = False

    # print("cheked", checked)
    # print("con", connected)
    # print("node ", current_node)
    for node in connected:
        # print("node v ", board[node[0]][node[1]])
        # print(cur_node_v)
        # print("chke", checked)
        # print(board[node[0]][node[1]] == cur_node_v)
        if node not in checked and board[node[0]][node[1]] == cur_node_v:
            # print("traversing ", node)
            # print("traversing ",board[node[0]][node[1]])
            if node in term_nodes:
                suc = True
            else:
                suc = _win_traverse(board, checked, node, term_nodes)
        if suc:
            break

    return suc


def _is_game_won(state: HexBoardGameState,
                 team_0: bool):
    checked = []
    board = state.hex_board

    b_size = len(board)

    win = False

    if team_0:
        node_v = 1  # TODO: is BAD, fix
        term_nodes = [(n, b_size - 1) for n in range(b_size)]
        start_nodes = [(n, 0) for n in range(b_size)]
    else:
        node_v = -1  # TODO: is BAD, fix
        term_nodes = [(b_size - 1, n) for n in range(b_size)]
        start_nodes = [(0, n) for n in range(b_size)]

    # print(start_nodes)
    # print(term_nodes)
    # print(node_v)
    for node in start_nodes:
        # print(node, board[node[0]][node[1]])
        if board[node[0]][node[1]] == node_v:
            # print(node)
            win = _win_traverse(board, checked, node, term_nodes)

        if win:
            break
    return win


def _get_free_positions(state: HexBoardGameState) -> [(int)]:
    ret = []
    for n, row in enumerate(state.hex_board):
        for i, tile in enumerate(row):
            if tile == 0:
                ret.append((n, i))

    return ret


def _gen_inverted_action_space_list(asl):
    ret = asl.c


class HexGameEnvironment(BaseEnvironment):

    def __init__(self,
                 board_size,
                 player_0_value=1,
                 player_1_value=- 1):
        self.player_0_value = player_0_value
        self.player_1_value = player_1_value
        self.board_size = board_size
        self._action_space_list = self.get_valid_actions(self.get_initial_state())

        self.inverted_action_space_list = []

    def act(self,
            state: HexBoardGameState,
            action: (int, int),
            inplace=False) -> (BaseState, int, bool):
        next_s = copy.deepcopy(state) if not inplace else state
        put_val = self.player_0_value if state.current_player_turn() == 0 else self.player_1_value
        x, y = action[0], action[1]

        if next_s.hex_board[x][y] != 0:
            print("#" * 10)
            print("INVALID MOVE")
            print("#" * 10)
            print("move: ", action)
            print("from s", next_s.get_as_vec())
            self.display_state(next_s)
            raise Exception("invalid move")

        next_s.hex_board[x][y] = put_val
        next_s.change_turn()

        # check if game is done
        valid_actions = self.get_valid_actions(next_s)
        is_board_full = len(valid_actions) <= 0
        player_0_won = _is_game_won(next_s, team_0=True)
        player_1_won = _is_game_won(next_s, team_0=False)

        if player_0_won:
            reward = 1
        elif player_1_won:
            reward = -1
        elif is_board_full:
            reward = 0
        else:
            reward = 0

        done = is_board_full or player_0_won or player_1_won
        return next_s, reward, done

    def get_valid_actions(self,
                          state):
        return _get_free_positions(state)

    def get_initial_state(self) -> HexBoardGameState:
        return HexBoardGameState(_make_hex_board(self.board_size))

    def is_state_won(self,
                     state) -> bool:
        return _is_game_won(state, True)

    def display_state(self,
                      state: HexBoardGameState):
        _terminal_display_hex_board(state.hex_board)

        red_w = _is_game_won(state, False)
        green_w = _is_game_won(state, True)
        print("Red won: ", red_w)
        print("Green won: ", green_w)
        print("stat vec: ", state.hex_board)
        pass

    def get_observation_space_size(self) -> int:
        return self.board_size ** 2

    def get_action_space_size(self) -> int:
        return self.board_size ** 2

    def get_action_space_list(self) -> []:
        return self._action_space_list

    def get_inverted_action_space_list(self,
                                       state: BaseState) -> [bool]:
        state_act = self.get_valid_actions(state)
        all_act = self.get_action_space_list()
        return [act in state_act for act in all_act]

    def get_valid_action_space_list(self,
                                    state: BaseState) -> [bool]:
        state_act = self.get_valid_actions(state)
        all_act = self.get_action_space_list()
        return [act in state_act for act in all_act]

