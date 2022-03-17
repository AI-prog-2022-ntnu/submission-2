import copy
import random

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState


class colors:
    RED = '\u001b[31m'
    GREEN = '\u001b[32m'
    RESET = '\u001b[0m'


def invert(x):
    """
    Changes 1 with -1 and -1 with ones 1. Else it returns 0
    """
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
        self.board_hash = None
        self.players_turn = 0
        self.update_state_hash()

    def update_state_hash(self):
        """
        Updates the state hash.
        """
        self.board_hash = hash(str(self.hex_board))

    def change_turn(self):
        """
        Changes the current player.
        If player 1 => player 2.
        If player 2 => player 1.
        """
        self.players_turn = (self.players_turn + 1) % 2

    def current_player_turn(self):
        """
        Returns the current player turn.
        """
        return self.players_turn

    def get_as_vec(self) -> [float]:
        """
        Returns the hex board as a vector.
        """
        ret = []
        for r in self.hex_board:
            ret.extend(r)
        return ret

    def get_as_inverted_vec(self) -> [float]:
        """
        Returns the hex board as an inverted vector.
        """
        ret = []
        inverted_board = copy.deepcopy(self.hex_board)
        axis_size = len(self.hex_board)

        for x in range(axis_size):
            for y in range(axis_size):
                inverted_board[x][y] = invert(self.hex_board[y][x])

        for r in inverted_board:
            ret.extend(r)

        return ret

    @staticmethod
    def invert_state_vec(vec, axis_size):
        """
        Todo - Remove? No usages
        """
        mod_vec = copy.deepcopy(vec)
        for x in range(axis_size):
            for y in range(axis_size):
                mod_vec[(x * axis_size) + y] = vec[(x * axis_size) + y]

    def __hash__(self):
        return self.board_hash

    def __eq__(self,
               other):
        return hash(self) == hash(other)


def _make_hex_board(size: int):
    """
    Creates the hex board.
    """
    ret = []
    for _ in range(size):
        ret.append([0 for _ in range(size)])

    return ret


def make_random_hex_board(size: int):
    """
    Creates a random hex board.
    """
    ret = []
    for _ in range(size):
        ret.append([random.randint(-1, 1) for _ in range(size)])

    return ret


def _terminal_display_hex_board(hex_board: [[int]]):
    """
    Displays the state when the terminal hex board is reach
    and display_state() is called.
    """
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


def _get_connected_cells(x: int, y: int, board: [[int]]) -> [(int), (int)]:
    """
    Returns the cells which is connected to the cell at x and y.
    """
    # TODO: some of these may be wrong
    idx_up_r = (x, y - 1)
    idx_r = (x + 1, y - 1)
    idx_down_r = (x + 1, y)
    idx_down_l = (x, y + 1)
    idx_l = (x - 1, y + 1)
    idx_up_l = (x - 1, y)
    tests = [idx_up_r, idx_r, idx_down_r, idx_down_l, idx_l, idx_up_l]

    valid = []
    upper_b = len(board[0])
    for test in tests:
        if test[0] < upper_b and test[0] >= 0 and test[1] < upper_b and test[1] >= 0:
            valid.append(test)

    return valid


def _win_traverse(board: [[int]], checked: [(int, int)], current_node: (int, int), term_nodes: [(int, int)]):
    """
    Traverses the tree until it finds the final state.
    """
    x, y = current_node[0], current_node[1]
    checked.append(current_node)
    cur_node_v = board[x][y]
    connected = _get_connected_cells(x, y, board)
    suc = False

    for node in connected:
        if node not in checked and board[node[0]][node[1]] == cur_node_v:
            if node in term_nodes:
                suc = True
            else:
                suc = _win_traverse(board, checked, node, term_nodes)
        if suc:
            break
    return suc


def _is_game_won(state: HexBoardGameState, team_0: bool):
    """
     Checks if the state has reached a winning state.
    """
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

    for node in start_nodes:
        if board[node[0]][node[1]] == node_v:
            win = _win_traverse(board, checked, node, term_nodes)

        if win:
            break
    return win


def _find_winning_move(player_board_value: int, board: [[int]], checked: [(int, int)], current_node: (int, int),
                       term_nodes: [(int, int)]):
    """

    """
    x, y = current_node[0], current_node[1]
    checked.append(current_node)
    connected = _get_connected_cells(x, y, board)
    move = None

    for node in connected:
        chek_node_value = board[node[0]][node[1]]
        if node not in checked:
            if chek_node_value == player_board_value:
                if node in term_nodes:
                    move = node
                else:
                    move = _find_winning_move(player_board_value, board, checked, node, term_nodes)
            elif chek_node_value == 0:
                move = _find_winning_move(player_board_value, board, checked, node, term_nodes)

            if move is not None:
                break
        return move


def find_winning_move(state: HexBoardGameState, team_0: bool):
    """

    """
    checked = []
    board = state.hex_board

    b_size = len(board)

    move = None

    if team_0:
        node_v = 1  # TODO: is BAD, fix
        term_nodes = [(n, b_size - 1) for n in range(b_size)]
        start_nodes = [(n, 0) for n in range(b_size)]
    else:
        node_v = -1  # TODO: is BAD, fix
        term_nodes = [(b_size - 1, n) for n in range(b_size)]
        start_nodes = [(0, n) for n in range(b_size)]

    for node in start_nodes:
        if board[node[0]][node[1]] == node_v:
            move = _find_winning_move(node_v, board, checked, node, term_nodes)

        if move is not None:
            break
    return move


def _get_free_positions(state: HexBoardGameState) -> [(int)]:
    """
    Returns a list of the positions that are still free.
    """
    ret = []
    for n, row in enumerate(state.hex_board):
        for i, tile in enumerate(row):
            if tile == 0:
                ret.append((n, i))

    return ret


def _slice_active_board_from_internal(board, slice_size):
    """
    Todo - Find out what does.
    """
    sliced_board = []
    for x in range(slice_size):
        sliced_board.append(board[x][-slice_size:])

    return sliced_board


class HexGameEnvironment(BoardGameEnvironment):

    def __init__(self, board_size, internal_board_size=10, player_0_value=1, player_1_value=- 1):
        self.internal_board_size = internal_board_size
        self.player_0_value = player_0_value
        self.player_1_value = player_1_value
        self.board_size = board_size

        val = []
        for x in range(internal_board_size):
            for y in range(internal_board_size):
                val.append((x, y))
        self._action_space_list = val

        self._observation_space_inversion_map = self._gen_observation_space_inversion_map()

    def _gen_observation_space_inversion_map(self):
        """
        Returns the observation space inversion mapping.
        The inversion is used to change between the player 1 and player 2 perspective.
        """
        observation_space_size = self.get_observation_space_size()
        map = [0 for _ in range(observation_space_size)]

        for x in range(self.internal_board_size):
            for y in range(self.internal_board_size):
                map[(x * self.internal_board_size) + y] = (x * self.internal_board_size) + y

        return map

    def invert_observation_space_vec(self, observation_space_vec):
        """
        Inverts the observation space vector.
        The inversion is used to change between the player 1 and player 2 perspective.
        """
        out = [0 for _ in range(self.get_observation_space_size())]
        for idx, map_idx in enumerate(self._observation_space_inversion_map):
            out[map_idx] = observation_space_vec[idx]
        return out

    def invert_action_space_vec(self, action_space_vec):
        """
        Inverts the observation space vector.
        The inversion is used to change between the player 1 and player 2 perspective.
        """
        out = [0 for _ in range(self.get_observation_space_size())]
        for idx, map_idx in enumerate(self._observation_space_inversion_map):
            out[map_idx] = action_space_vec[idx]
        return out

    def game_has_reversible_moves(self) -> bool:
        """
        Returns if the game has reversible moves or not.
        """
        return True

    def get_state_winning_move(self, state: HexBoardGameState) -> (int, int):
        """
        Returns the best move for state.
        """
        return find_winning_move(state, state.current_player_turn() != 0)

    def reverse_move(self, state: HexBoardGameState, action):
        """
        Reverses the move, changes the state and updates the state hash.
        """
        state.hex_board[action[0]][action[1]] = 0
        state.change_turn()
        state.update_state_hash()

    def act(self, state: HexBoardGameState, action: (int, int), inplace=False) -> (BaseState, int, bool):
        """
        Simulate the transition from the provided state to a new state using the provided action.
        The new state is returned, along with the reward and a bool indicating whether the new state is terminal.
        """
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
        next_s.update_state_hash()

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
            # this is not possible
            raise Exception("Hex Board is full. This should not be possible")
        else:
            reward = 0

        done = is_board_full or player_0_won or player_1_won
        return next_s, reward, done

    def get_winning_player(self, state):
        """
        Returns the number of the winning player or None if nobody has won yet.
        """

        player_0_won = _is_game_won(state, team_0=True)
        player_1_won = _is_game_won(state, team_0=False)

        if player_0_won:
            return 0
        elif player_1_won:
            return 1
        else:
            return None

    def get_valid_actions(self, state):
        """
        Returns the valid actions of the state.
        """
        return _get_free_positions(state)

    def get_initial_state(self) -> HexBoardGameState:
        """
        Returns the initial state.
        """
        if self.internal_board_size == self.board_size:
            board = _make_hex_board(self.internal_board_size)
        else:
            diff = self.internal_board_size - self.board_size
            board = _make_hex_board(self.internal_board_size)
            cnt = diff
            for x in range(self.internal_board_size):
                num_green = 0
                any_red = False

                if x < self.board_size:
                    num_green = diff
                else:
                    any_red = True
                    num_green = max(cnt, 0)
                    cnt -= 1

                for y in range(self.internal_board_size):
                    if num_green > 0:
                        board[x][y] = 1
                        num_green -= 1
                    else:
                        if any_red:
                            board[x][y] = -1

        return HexBoardGameState(board)

    def is_state_won(self, state) -> bool:
        """
        Checks if the state is currently in a winning state.
        """
        return _is_game_won(state, True)

    def display_state(self, state: HexBoardGameState, display_internal=False):
        """
        Displays information about the state.
        """
        if display_internal:
            board = state.hex_board
        else:
            board = _slice_active_board_from_internal(
                board=state.hex_board,
                slice_size=self.board_size
            )
        _terminal_display_hex_board(board)

        red_w = _is_game_won(state, False)
        green_w = _is_game_won(state, True)
        print("Red won: ", red_w)
        print("Green won: ", green_w)
        pass

    def get_observation_space_size(self) -> int:
        """
        Return self.board_size ** 2
        """
        return self.internal_board_size ** 2

    def get_action_space_size(self) -> int:
        """
        Return self.board_size ** 2
        """
        return self.internal_board_size ** 2

    def get_action_space(self) -> []:
        """
        Returns the action space list.
        """
        return self._action_space_list

    def get_valid_action_space_list(self, state: BaseState) -> [bool]:
        """
        Returns a list of the valid action spaces.
        """
        state_act = self.get_valid_actions(state)
        all_act = self.get_action_space()
        return [act in state_act for act in all_act]
