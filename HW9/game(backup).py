import copy
import math
import random
import time


class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth = 3



    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = self.detect_drop_phase(state)   # TODO: detect drop phase
        succs = self.succ(state, drop_phase, self.my_piece)
        winner = None
        min = []
        for succ in succs:
            if self.game_value(succ) == 1:
                winner = succ
                break
            curr_min = self.min_value(float("-inf"), float("inf"), state, 0)
            min.append((succ, curr_min))

        curr_succ = None
        temp = None
        if winner is not None:
            curr_succ = winner
        else:
            temp = min[0][1]
            curr_succ = min[0][0]
            for i in min:
                if temp < i[1]:
                    temp = i[1]
                    curr_succ = i[0]

        move = []

        for row in range(5):
            for col in range(5):
                # compare succ and state
                if drop_phase:
                    if curr_succ[row][col] != state[row][col]:
                        move.append((row, col))
                        return move

                else:
                    # return a list of two tuples, 0 = new position 1 = old position
                    if curr_succ[row][col] != state[row][col]:
                        # new position
                        if state[row][col] == " ":
                            move.insert(0, (row, col))
                        # old position
                        else:
                            move.append((row, col))

        return move


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and 3x3 square corners wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        ## TODO: check \ diagonal wins
        for row in range(1):
            for col in range(1):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == \
                        state[row + 3][col + 3]:
                    return 1 if state[i][col] == self.my_piece else -1

        # TODO: check / diagonal wins
        for row in range(1):
            for col in range(3, 5):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col - 1] == state[row + 2][col - 2] == \
                        state[row + 3][col - 3]:
                    return 1 if state[i][col] == self.my_piece else -1
        # TODO: check 3x3 square corners wins
        for row in range(3):
            for col in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row][col+2] == state[row + 2][col] == \
                        state[row + 2][col + 2]:
                    return 1 if state[i][col] == self.my_piece else -1

        return 0 # no winner yet


    def detect_drop_phase(self, state):
        counter = 0
        for row in range(5):
            for col in range(5):
                if state[row][col] != ' ':
                    counter += 1
        if counter < 8:
            return True
        return False

    def succ(self, state, is_drop, player_color):
        succs = []

        if is_drop:
            for list in range(5):
                for pos in range(5):
                    if state[list][pos] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[list][pos] = player_color
                        succs.append(new_state)

        else:
            for list in range(5):
                for pos in range(5):
                    if state[list][pos] == player_color:
                        if list + 1 <= 4 and state[list + 1][pos] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list + 1][pos] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list - 1 >= 0 and state[list - 1][pos] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list - 1][pos] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if pos + 1 <= 4 and state[list][pos + 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list][pos + 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list - 1 >= 0 and state[list][pos - 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list][pos - 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list + 1 <= 4 and pos + 1 <= 4 and state[list + 1][pos + 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list + 1][pos + 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list + 1 <= 4 and pos - 1 >= 0 and state[list + 1][pos - 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list + 1][pos - 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list - 1 >= 0 and pos + 1 <= 4 and state[list - 1][pos + 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list - 1][pos + 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
                        if list - 1 >= 0 and pos - 1 >= 0 and state[list - 1][pos - 1] == ' ':
                            new_state = copy.deepcopy(state)
                            new_state[list - 1][pos - 1] = player_color
                            new_state[list][pos] = ' '
                            succs.append(new_state)
        return succs

    def max_value(self, alpha, beta, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= self.depth:
            return self.heuristic_game_value(state)

        for successor in self.succ(state, self.detect_drop_phase(state), self.my_piece):
            alpha = max(alpha, self.min_value(alpha, beta, successor, depth + 1))
            if alpha >= beta:
                return beta
        return alpha

    def min_value(self, alpha, beta, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= self.depth:
            return self.heuristic_game_value(state)

        for successor in self.succ(state, self.detect_drop_phase(state), self.my_piece):
            beta = min(beta, self.max_value(alpha, beta, successor, depth + 1))
            if alpha >= beta:
                return beta
        return alpha

    def heuristic_game_value(self, state):
        val = self.game_value(state)
        if val != 0:
            return val

        max = []
        min = []
        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                    max.append([row, col])
                if state[row][col] == self.opp:
                    min.append([row, col])
        max_vals = []
        for i in range(len(max)):
            for j in range(len(max)):
                if j <= i:
                    continue
                x = math.pow((max[j][0] - max[i][0]), 2)
                y = math.pow((max[j][1] - max[i][1]), 2)
                max_vals.append(math.sqrt(x+y))
        max_total = 0
        if len(max_vals) < 1:
            total = 2
        else:
            for k in max_vals:
                max_total += k
            total = max_total / max_vals.__len__()

        min_vals = []
        for i in range(len(min)):
            for j in range(len(min)):
                if j <= i:
                    continue
                x = math.pow((min[j][0] - min[i][0]), 2)
                y = math.pow((min[j][1] - min[i][1]), 2)
                min_vals.append(math.sqrt(x + y))
        min_total = 0
        if len(min_vals) < 1:
            min_total = 2
        else:
            for k in min_vals:
                min_total = min_total + k
            min_total= min_total / min_vals.__len__()

        # get approx normalization of avg distances
        max_total = 1 / max_total
        min_total = 1 / min_total
        if max_total >= 1:
            max_total = 0.99
        if min_total >= 1:
            min_total = 0.99

        if max_total >= min_total:
            return max_total
        else:
            return -1 * min_total

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    state = [[' ', ' ', 'b', 'r', ' '], [' ', ' ', 'b', 'r', ' '], [' ', 'b', ' ', ' ', ' '], [' ', ' ', ' ', 'r', ' '], [' ', 'r', 'b', ' ', ' ']]
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    alpha = float('-inf')
    beta = float('inf')

    print(ai.min_value(alpha, beta, state, 1))

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
