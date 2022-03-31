import copy
import marshal
import random

from enviorments.base_environment import BoardGameEnvironment
from enviorments.base_state import BaseState, BoardGameBaseState


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
    board_size = 10

    def __init__(self,
                 board,
                 board_padding=3):
        self.board_padding = board_padding
        self._board_vec = []
        self.board_hash = None
        self.players_turn = 0

        if board is not None:
            self.board_size_x = len(board)
            for r in board:
                self._board_vec.extend(r)
            self.update_state_hash()

    def update_state_hash(self):
        """
        Updates the state hash
        """
        self.board_hash = hash(marshal.dumps(self._board_vec))
        # self.board_hash = hash(str(self.hex_board))

    def change_turn(self):
        """
        Changes the turn of the player.
        """
        self.players_turn = (self.players_turn + 1) % 2

    def current_player_turn(self):
        """
        Returns which player is playing on the current turn.
        """
        return self.players_turn

    def get_as_vec(self) -> [float]:
        """
        Returns the game board as a vector.
        """
        return self._board_vec

    def get_board_val(self,
                      x,
                      y) -> float:
        """
        Returns the game board values.
        """
        return self._board_vec[(x * self.board_size_x) + y]

    def set_board_val(self,
                      x,
                      y,
                      val):
        """
        Sets the game board values.
        """
        self._board_vec[(x * self.board_size_x) + y] = val

    def get_as_inverted_vec(self) -> [float]:
        """
        Returns the game board as an inverted vector. The inversion is used so the model doesn't have to
        learn the moves for both player 1 and player 2.
        """
        ret = []
        inverted_board = copy.deepcopy(self.get_as_board())

        for x in range(self.board_size_x):
            for y in range(self.board_size_x):
                inverted_board[x][y] = invert(self.get_board_val([y], [x]))

        for r in inverted_board:
            ret.extend(r)

        return ret

    def get_as_board(self):
        """
        Returns the board size as as a list representing the board.
        """
        return [self._board_vec[n * self.board_size_x:(n + 1) * self.board_size_x] for n in range(self.board_size_x)]

    @staticmethod
    def invert_state_vec(vec,
                         axis_size):
        """
        Todo - Remove? Coulnd't find any usages.
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
    Used to make the internal hex board.
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
    Displays the state of the hex board in the terminal.
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


def _get_connected_cells(x: int,
                         y: int,
                         state: HexBoardGameState) -> [(int), (int)]:
    """
    Returns a list of all the valid connected cells to "x" and "y".
    """
    # TODO: some of theese may be wrong
    idx_up_r = (x, y - 1)
    idx_r = (x + 1, y - 1)
    idx_down_r = (x + 1, y)
    idx_down_l = (x, y + 1)
    idx_l = (x - 1, y + 1)
    idx_up_l = (x - 1, y)
    tests = [idx_up_r, idx_r, idx_down_r, idx_down_l, idx_l, idx_up_l]

    valid = []
    # print()
    for test in tests:
        # print(test)
        if test[0] < state.board_size_x - state.board_padding and test[0] >= 0 and test[1] < state.board_size_x and test[1] >= state.board_padding:
            valid.append(test)

    return valid


def _win_traverse(state: HexBoardGameState,
                  checked: set,
                  current_node: (int, int),
                  term_nodes: set,
                  cur_node_v):
    """
    Traverses the nodes connected to the current nodes and checks if any of the connected nodes leads to a state
    with a victory.
    """
    x, y = current_node[0], current_node[1]
    checked.add(current_node)
    connected = _get_connected_cells(x, y, state)
    suc = False

    for node in connected:
        node_value = state.get_board_val(node[0], node[1])
        if node not in checked and node_value == cur_node_v:
            if node in term_nodes:
                suc = True
            else:
                suc = _win_traverse(state, checked, node, term_nodes, node_value)
        if suc:
            break
    return suc


def _is_game_won(state: HexBoardGameState,
                 team_0: bool):
    """
    Checks if the game is won by either of the players.
    """
    checked = set()

    win = False

    term_nodes, start_nodes = set(), set()

    if team_0:
        node_v = 1  # TODO: is BAD, fix
        for n in range(state.board_size_x - state.board_padding):
            term_nodes.add((n, state.board_size_x - 1))
            checked.add((n, 0 + state.board_padding))
        start_nodes = [(n, 0 + state.board_padding) for n in range(state.board_size_x - state.board_padding)]
    else:
        node_v = -1  # TODO: is BAD, fix
        for n in range(state.board_size_x - state.board_padding):
            term_nodes.add((state.board_size_x - state.board_padding - 1, n + state.board_padding))
            checked.add((0, n + state.board_padding))
        start_nodes = [(0, n + state.board_padding) for n in range(state.board_size_x - state.board_padding)]

    for node in start_nodes:
        if state.get_board_val(node[0], node[1]) == node_v:
            win = _win_traverse(state, checked, node, term_nodes, node_v)

        if win:
            break

    return win


def _find_winning_move(
        player_board_value: int,
        state: HexBoardGameState,
        checked: set,
        current_node: (int, int),
        term_nodes: set):
        term_nodes: set,
        is_trying=None):
    """
    Tries to find a winning move. If the current node already is in term_nodes, it returns it. Else, it calls itself
    recursively while trying to find a move resulting in a victory. If it succeeds, it returns the tuple
    containing the move, if it fails it returns None.
    """
    x, y = current_node[0], current_node[1]
    checked.add(current_node)
    connected = _get_connected_cells(x, y, state)
    move = None

    # print("current ", current_node)
    # print(connected)
    try_hops = []
    for node in connected:
        chek_node_value = state.get_board_val(node[0], node[1])
        if node not in checked:
            if chek_node_value == player_board_value:
                if node in term_nodes:
                    # if is_trying is None:
                    #     return (-1,-1)
                    move = is_trying
                else:
                    move = _find_winning_move(player_board_value, state, checked, node, term_nodes, is_trying=is_trying)
            elif chek_node_value == 0 and is_trying is None:
                # print("hop v")
                try_hops.append(node)

            if move is not None:
                return move

    for try_hop in reversed(try_hops):
        move = _find_winning_move(player_board_value, state, checked, try_hop, term_nodes, is_trying=try_hop)

        if move is not None:
            return move
    return move


def find_winning_move(
        state: HexBoardGameState,
        team_0: bool):
    """
    Returns the winning move if it finds one, None otherwise.
    """
    checked = set()
    move = None

    term_nodes, start_nodes = set(), set()

    if team_0:
        node_v = 1  # TODO: is BAD, fix
        for n in range(state.board_size_x - state.board_padding):
            term_nodes.add((n, state.board_size_x - 1))
            checked.add((n, 0 + state.board_padding))
        start_nodes = [(n, 0 + state.board_padding) for n in range(state.board_size_x - state.board_padding)]
    else:
        node_v = -1  # TODO: is BAD, fix
        for n in range(state.board_size_x - state.board_padding):
            term_nodes.add((state.board_size_x - state.board_padding - 1, n + state.board_padding))
            checked.add((0, n + state.board_padding))
        start_nodes = [(0, n + state.board_padding) for n in range(state.board_size_x - state.board_padding)]

    # print("is team 0", team_0)
    # print("term", term_nodes)
    # print("init", start_nodes)
    for node in start_nodes:
        if state.get_board_val(node[0], node[1]) == node_v:
            move = _find_winning_move(node_v, state, checked, node, term_nodes)

        if move is not None:
            return move
            break

    return move


def _get_free_positions(state: HexBoardGameState) -> [(int)]:
    """
    Returns the positions on the board that are available.
    """
    ret = []

    for x in range(state.board_size_x):
        for y in range(state.board_size_x):
            if state.get_board_val(x, y) == 0:
                ret.append((x, y))

    return ret


def _slice_active_board_from_internal(board,
                                      slice_size):
    """
    Todo - Figure out exactly what it does.
    """
    sliced_board = []
    for x in range(slice_size):
        sliced_board.append(board[x][-slice_size:])

    return sliced_board


class HexGameEnvironment(BoardGameEnvironment):

    def __init__(self,
                 board_size,
                 internal_board_size=10,
                 player_0_value=1,
                 player_1_value=- 1):
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
        Returns the observation space inversion ap.
        """
        observation_space_size = self.get_observation_space_size()
        map = [0 for _ in range(observation_space_size)]

        for x in range(self.internal_board_size):
            for y in range(self.internal_board_size):
                map[(x * self.internal_board_size) + y] = (x * self.internal_board_size) + y

        return map

    def invert_observation_space_vec(self,
                                     observation_space_vec):
        """
        Inverts the observation space vector.
        """
        out = [0 for _ in range(self.get_observation_space_size())]
        for idx, map_idx in enumerate(self._observation_space_inversion_map):
            out[map_idx] = observation_space_vec[idx]
        return out

    def invert_action_space_vec(self,
                                action_space_vec):
        """
        Inverts the action space vector.
        """
        out = [0 for _ in range(self.get_observation_space_size())]
        for idx, map_idx in enumerate(self._observation_space_inversion_map):
            out[map_idx] = action_space_vec[idx]
        return out

    def game_has_reversible_moves(self) -> bool:
        """
        Checks if the game has reversible moves.
        """
        return True

    def get_state_winning_move(self,
                               state: BoardGameBaseState) -> (bool, (int, int)):
        """
        Returns the state winning move if found and None otherwise.
        """
        return find_winning_move(state, state.current_player_turn() == 0)

    def reverse_move(self,
                     state: HexBoardGameState,
                     action):
        """
        Reverses the moves done.
        """
        state.set_board_val(action[0], action[1], 0)
        state.change_turn()
        state.update_state_hash()

    def act(self,
            state: HexBoardGameState,
            action: (int, int),
            inplace=False) -> (BaseState, int, bool):
        """
        Simulate the transition from the provided state to a new state using the provided action.
        The new state is returned, along with the reward and a bool indicating whether the new state is terminal.
        """

        if inplace:
            next_s = state
        else:
            next_s = HexBoardGameState(None, board_padding=self.internal_board_size - self.board_size)
            next_s.board_size_x = state.board_size_x
            next_s._board_vec = state._board_vec[:]
            next_s.players_turn = state.players_turn

        put_val = self.player_0_value if state.current_player_turn() == 0 else self.player_1_value
        x, y = action[0], action[1]

        if next_s.get_board_val(x, y) != 0:
            print("#" * 10)
            print("INVALID MOVE")
            print("#" * 10)
            print("move: ", action)
            print("from s", next_s.get_as_vec())
            self.display_state(next_s)
            raise Exception("invalid move")

        next_s.set_board_val(x, y, put_val)
        next_s.change_turn()
        next_s.update_state_hash()

        reward = 0
        if state.current_player_turn() == 0:
            won = _is_game_won(next_s, team_0=False)
            if won:
                reward = -1
        else:
            won = _is_game_won(next_s, team_0=True)
            if won:
                reward = 1

        # if _is_game_won(next_s, team_0=False):
        #     reward = -1
        #     won = True
        # elif _is_game_won(next_s, team_0=True):
        #     reward = 1
        #     won = True
        # else:
        #     reward = 0
        #     won = False

        return next_s, reward, won

    def get_winning_player(self,
                           state):
        """
        Returns the number of the player who won.
        """

        player_0_won = _is_game_won(state, team_0=True)
        player_1_won = _is_game_won(state, team_0=False)

        if player_0_won:
            return 0
        elif player_1_won:
            return 1
        else:
            return None

    def get_valid_actions(self,
                          state):
        """
        Returns the valid actions of the current state.
        """
        return _get_free_positions(state)

    def get_initial_state(self) -> HexBoardGameState:
        """
        Returns teh initial state of teh game board.
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

        return HexBoardGameState(board, board_padding=self.internal_board_size - self.board_size)

    def is_state_won(self,
                     state) -> bool:
        """
        Checks if the state is won.
        """
        return _is_game_won(state, True)

    def get_winning_player_id(self,
                              state):
        """
        If the game is won, it returns the id of the player who won. Else returns None.
        """
        if _is_game_won(state, True):
            return 0
        elif _is_game_won(state, False):
            return 1

        return None

    def display_state(self,
                      state: HexBoardGameState,
                      display_internal=False):
        """
        Displays the current state if the board and prints the status of the game.
        """
        if display_internal:
            board = state.get_as_board()
        else:
            board = _slice_active_board_from_internal(
                board=state.get_as_board(),
                slice_size=self.board_size
            )
        _terminal_display_hex_board(board)

        red_w = _is_game_won(state, False)
        green_w = _is_game_won(state, True)
        print("Red won: ", red_w)
        print("Green won: ", green_w)
        # print("stat vec: ", state._board_vec)
        pass

    def get_observation_space_size(self) -> int:
        """
        Returns the observation space of the board.
        """
        # return self.board_size ** 2
        return self.internal_board_size ** 2

    def get_action_space_size(self) -> int:
        """
        Returns the action space of the board.
        """
        # return self.board_size ** 2
        return self.internal_board_size ** 2

    def get_action_space(self) -> []:
        """
        Returns the action space.
        """
        return self._action_space_list

    def get_valid_action_space_list(self,
                                    state: BaseState) -> [bool]:
        """
        Returns the valid actions space list.
        """
        state_act = self.get_valid_actions(state)
        all_act = self.get_action_space()
        return [act in state_act for act in all_act]
