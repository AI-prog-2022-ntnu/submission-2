import random

from enviorments.base_environment import BaseEnvironment
from enviorments.base_state import BaseState

import copy


class colors:
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    RESET = '\u001b[0m'
    # RESET = '\033[0;0m'


class HexGameState(BaseState):
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
        self.player_1_turn = True

    def get_as_vec(self) -> [float]:
        pass


def _make_hex_board(size: int):
    ret = []
    for _ in range(size):
        # ret.append([random.randint(-1,1) for _ in range(size)])
        ret.append([random.randint(-1, 1) for _ in range(size)])

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

        print(disp_r)
        if n != width - 1:
            print(padd_r)


def _get_connected_cells(x: int,
                         y: int,
                         board: [[int]]) -> [(int), (int)]:
    idx_up_r = (x - 1, y)
    idx_r = (x - 1, y - 1)
    idx_down_r = (x, y + 1)
    idx_down_l = (x + 1, y)
    idx_l = (x + 1, y - 1)
    idx_up_l = (x, y - 1)
    tests = [idx_up_r, idx_r, idx_down_r, idx_down_l, idx_l, idx_up_l]

    valid = []
    upper_b = len(board[0])
    for test in tests:
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
    for node in connected:
        # print("node v ",board[node[0]][node[1]])
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


def _is_game_won(state: HexGameState,
                 team_a: bool):
    checked = []
    board = state.hex_board

    b_size = len(board)

    win = False

    if team_a:
        node_v = 1  # TODO: is BAD, fix
        term_nodes = [(n, b_size - 1) for n in range(b_size)]
        start_nodes = [(n, 0) for n in range(b_size)]
    else:
        node_v = -1  # TODO: is BAD, fix
        term_nodes = [(b_size - 1, n) for n in range(b_size)]
        start_nodes = [(0, n) for n in range(b_size)]

    for node in start_nodes:
        if board[node[0]][node[1]] == node_v:
            win = _win_traverse(board, checked, node, term_nodes)

        if win:
            break
    return win


def _get_free_positions(state: HexGameState) -> [(int)]:
    ret = []
    for n, row in enumerate(state.hex_board):
        for i, tile in enumerate(state.hex_board):
            if tile == 0:
                ret.append((n, i))

    return ret


class HexGameEnvironment(BaseEnvironment):

    def __init__(self,
                 board_size,
                 player_1_value=1,
                 player_2_value=- 1):
        self.player_2_value = player_2_value
        self.player_1_value = player_1_value
        self.board_size = board_size

    def act(self,
            state: HexGameState,
            action: (int,int),
            inplace=False) -> (BaseState, int, bool):
        next_s = copy.deepcopy(state) if not inplace else state
        put_val = self.player_1_value if state.player_1_turn else self.player_2_value
        x,y = action[0],action[1]

        next_s.hex_board[x][y] = put_val
        next_s.player_1_turn ^= True
        return next_s


    def get_valid_actions(self,
                          state):
        return _get_free_positions(state)

    def get_initial_state(self) -> HexGameState:
        return HexGameState(_make_hex_board(self.board_size))

    def is_state_won(self,
                     state) -> bool:
        return _is_game_won(state, True)

    def display_state(self,
                      state: HexGameState):
        _terminal_display_hex_board(state.hex_board)

        red_w = _is_game_won(state, False)
        green_w = _is_game_won(state, True)
        print("Red won: ", red_w)
        print("Green won: ", green_w)
        pass

    def get_observation_space_size(self) -> int:
        return self.board_size**2

    def get_action_space_size(self) -> int:
        return self.board_size**2

