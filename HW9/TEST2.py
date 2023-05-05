# Name: Jiwon Song
# Project: p6
# Class: Spring 2020 CS 540

import random
import copy
import math
from collections import defaultdict
import time


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']
    alpha = float("inf")
    beta = float("-inf")
    initial_depth = 0
    max_depth = 3

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
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
        # check whether it is in drop phase or not
        drop_phase = False
        count = 0
        for row in state:
            for element in row:
                if element != ' ':
                    count += 1
        if count < 8:
            drop_phase = True
        if not drop_phase:

            successors = self.succ(state, drop_phase)
            heuristics = []
            for s in successors:
                heuristics += [(s, self.heuristic_game_value(s, self.initial_depth, drop_phase))]

            best = 100
            best_one = []
            for h in heuristics:
                if h[1] < best:
                    best = h[1]
                    best_one = h[0]
            orig = []
            new = []
            for i in range(len(state)):
                for j in range(len(state)):
                    if state[i][j] == self.my_piece:
                        orig += [(i, j)]
                    if best_one[i][j] == self.my_piece:
                        new += [(i, j)]

            move = []
            for n in new:
                if n not in orig:
                    move.insert(0, n)
            for e in orig:
                if e not in new:
                    move.insert(1, e)
            return move

        if count == 0:
            move = []
            (row, col) = (random.randint(0, 4), random.randint(0, 4))
            move.insert(0, (row, col))
            return move

        successors = self.succ(state, drop_phase)
        heuristics = []
        for s in successors:
            heuristics += [(s, self.heuristic_game_value(s, self.initial_depth, drop_phase))]

        best = 100
        best_one = []
        for h in heuristics:
            if h[1] < best:
                best = h[1]
                best_one = h[0]
        orig = []
        new = []
        for i in range(len(state)):
            for j in range(len(state)):
                if state[i][j] == self.my_piece:
                    orig += [(i, j)]
                if best_one[i][j] == self.my_piece:
                    new += [(i, j)]

        move = []
        for n in new:
            if n not in orig:
                move.insert(0, n)
        return move

    def Max_Value(self, state, depth, drop_phase):

        # if state is a terminal state, return the state
        if self.game_value(state) == 1:
            return 1
        elif depth >= self.max_depth:
            return 0
        else:
            successors = self.succ(state, drop_phase)
            for s in successors:
                self.alpha = max(self.alpha, self.Min_Value(s, depth + 1, drop_phase))
            return self.alpha

    def Min_Value(self, state, depth, drop_phase):

        # if state is a terminal state, return the state
        if self.game_value(state) == -1:
            return -1
        elif depth >= self.max_depth:
            return 0
        else:
            successors = self.succ(state, drop_phase)
            for s in successors:
                self.beta = min(self.beta, self.Max_Value(s, depth + 1, drop_phase))
            return self.beta

    def succ(self, state, drop_phase):
        '''
        :param state:
        :param drop_phase:
        :return:
        '''
        # successor list to return
        succs = []
        if not drop_phase:
            for row in range(len(state)):
                for col in range(len(state)):
                    if state[row][col] == self.my_piece:
                        # check upper
                        if not row - 1 < 0:
                            if state[row - 1][col] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row - 1][col] = self.my_piece
                                succs += [s]
                        # check left
                        if not col - 1 < 0:
                            if state[row][col - 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row][col - 1] = self.my_piece
                                succs += [s]
                        # check down
                        if row + 1 < len(state):
                            if state[row + 1][col] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row + 1][col] = self.my_piece
                        # check right
                        if col + 1 < len(state):
                            if state[row][col + 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row][col + 1] = self.my_piece
                        # check upper left
                        if not row - 1 < 0 and not col - 1 < 0:
                            if state[row - 1][col - 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row - 1][col - 1] = self.my_piece
                                succs += [s]
                        # check upper right
                        if not row - 1 < 0 and col + 1 < len(state):
                            if state[row - 1][col + 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row - 1][col + 1] = self.my_piece
                                succs += [s]
                        # check down left
                        if row + 1 < len(state) and not col - 1 < 0:
                            if state[row + 1][col - 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row + 1][col - 1] = self.my_piece
                                succs += [s]
                        # check down right
                        if row + 1 < len(state) and col + 1 < len(state):
                            if state[row + 1][col + 1] == ' ':
                                s = copy.deepcopy(state)
                                s[row][col] = ' '
                                s[row + 1][col + 1] = self.my_piece
                                succs += [s]
            return succs
        else:
            for row in range(len(state)):
                for col in range(len(state)):
                    if state[row][col] == ' ':
                        new = copy.deepcopy(state)
                        # mark the empty space
                        new[row][col] = self.my_piece
                        succs += [new]
            return succs

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
                raise Exception("You don't have a piece there!")
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
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.
        Returns:
        int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        completed: complete checks for diagonal and 2x2 box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][
                    col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins

        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] \
                        == state[row + 3][col + 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # check / diagonal wins
        for row in range(2):
            for col in range(4, 2, -1):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col - 1] == state[row + 2][col - 2] \
                        == state[row + 3][col - 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # check 2x2 box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 1][col] \
                        == state[row][col + 1]:
                    return 1 if state[row][col] == self.my_piece else -1
        return 0  # no winner yet

    def heuristic_game_value(self, state, depth, drop_phase):

        if not drop_phase:
            is_terminal = self.Max_Value(state, depth, drop_phase)
            if is_terminal == 1:
                return 1
            elif is_terminal == -1:
                return -1
            else:
                # for non-terminal states
                total_score = 0
                score = 0.2

                pieces = []
                opp = []
                for i in range(len(state)):
                    for j in range(len(state)):
                        if state[i][j] == self.my_piece:
                            pieces += [(i, j)]
                        if state[i][j] == self.opp:
                            opp += [(i, j)]

                for p in pieces:
                    for q in pieces:
                        if p != q:
                            distance = math.sqrt(((p[1] - q[1]) ** 2) + ((p[0] - q[0]) ** 2))
                            if distance == 1:
                                total_score -= score
                            else:
                                total_score += distance

                total_score = total_score / 4

                return total_score
        else:
            total_score = 0
            score = 0.2
            counts = defaultdict(int)
            for row in state:
                for col in row:
                    if col != ' ':
                        counts[col] += 1

            if counts[self.my_piece] == 1:
                pieces = []
                opp = []
                for i in range(len(state)):
                    for j in range(len(state)):
                        if state[i][j] == self.my_piece:
                            pieces += [(i, j)]
                        if state[i][j] == self.opp:
                            opp += [(i, j)]
                for p in pieces:
                    for o in opp:
                        distance = math.sqrt(((p[1] - o[1]) ** 2) + ((p[0] - o[0]) ** 2))
                        if distance == 1:
                            total_score -= score
                        else:
                            total_score += distance
                total_score = total_score / 4
                return total_score
            else:
                pieces = []
                for i in range(len(state)):
                    for j in range(len(state)):
                        if state[i][j] == self.my_piece:
                            pieces += [(i, j)]

                for p in pieces:
                    for q in pieces:
                        if p != q:
                            distance = math.sqrt(((p[1] - q[1]) ** 2) + ((p[0] - q[0]) ** 2))
                            if distance == 1:
                                total_score -= score
                            else:
                                total_score += distance

                total_score = total_score / 4

                return total_score


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################

def main():
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0
    state = [[' ', ' ', 'b', 'r', ' '], [' ', ' ', 'b', 'r', ' '], [' ', 'b', ' ', ' ', ' '], [' ', ' ', ' ', 'r', ' '],
             [' ', 'r', 'b', ' ', ' ']]
    alpha = float('-inf')
    beta = float('inf')

    print(ai.Min_Value(state, 1, False))

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
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
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
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