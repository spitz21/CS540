import copy
import heapq
import time


def heuristic (puzzle):

    """
    call succ(state), and print each element with its heuristic value
    """

    row = 0
    col = 0
    h = 0

    for element in puzzle:
        # print(element)
        expect_corrd = get_expect_coord(element)

        if(element == 0):
            col += 1
            if col % 3 == 0 and col != 0:
                row += 1
            if col % 3 == 0 and col != 0:
                col = 0

            continue

        # calculate the manhattan distance of the element location and expected location
        h += abs(expect_corrd[0] - row) + abs(expect_corrd[1] - col)

        col += 1

        # update the row
        if col %3 == 0 and col != 0:
            row += 1
        if col % 3 == 0 and col != 0:
            col = 0
    return h


def print_succ(state):

    """
    # call succ(state), and print each element with its heuristic value
    """

    succ_list = succ(state)

    for element in succ_list:
        print(element, 'h=' + str(heuristic(element)))

# return all the possible successors of the state
def succ(state):
    row = 0
    succ_list = []
    for i in range(9):
        if state[i] == 0:
            # for the first row
            if row == 0:
                # first row tiles can always switch with the tile under them
                succ1 = swap_pos(state, i, i + 3)
                succ_list.append(succ1)

                # determine to swith left of right or both for col is middle
                # same for the latter two rows
                succs = help_print(state, i)
                if i % 3 == 1:
                    succ_list.append(succs[0])
                    succ_list.append(succs[1])
                else:
                    succ_list.append(succs)

            elif row == 1:
                # second row can always swith up and down
                succ1 = swap_pos(state, i, i + 3)
                succ3 = swap_pos(state, i, i - 3)

                succ_list.append(succ1)
                succ_list.append(succ3)

                succs = help_print(state, i)

                if i % 3 == 1:
                    succ_list.append(succs[0])
                    succ_list.append(succs[1])
                else:
                    succ_list.append(succs)

            elif row == 2:
                # third row can always swithc up
                succ1 = swap_pos(state, i, i - 3)
                succ_list.append(succ1)

                succs = help_print(state, i)

                if i % 3 == 1:
                    succ_list.append(succs[0])
                    succ_list.append(succs[1])
                else:
                    succ_list.append(succs)
        # update row
        if i % 3 == 2:
            row += 1

    # sorted(succ_list)


    return sorted(succ_list)

# helper function to determine the possible successors by switching horizontally
# return the list of horizontally swtich successors if i in middle col, else return successor
def help_print(state, i):
    succ_list = []

    # col in the left
    if i % 3 == 0:
        # col 0
        succ2 = swap_pos(state, i, i + 1)
        succ_list = succ2

    # col in the middle, return two possible successors
    if i % 3 == 1:
        # col 1
        succ2 = swap_pos(state, i, i + 1)
        succ3 = swap_pos(state, i, i - 1)

        succ_list.append(succ2)
        succ_list.append(succ3)

    # col in the right
    if i % 3 == 2:
        # col 2
        succ2 = swap_pos(state, i, i - 1)
        succ_list = succ2

    return succ_list


def swap_pos (state, pos1, pos2):
    state_copy = copy.deepcopy(state)
    a = state_copy[pos1]
    state_copy[pos1] = state_copy[pos2]
    state_copy[pos2] = a
    # print(state_copy)

    return state_copy


def get_expect_coord(value):
    coord = []
    if value == 1:
        coord = [0,0]
    elif value == 2:
        coord = [0,1]
    elif value == 3:
        coord = [0,2]
    elif value == 4:
        coord = [1,0]
    elif value == 5:
        coord = [1,1]
    elif value == 6:
        coord = [1,2]
    elif value == 7:
        coord = [2,0]
    elif value == 8:
        coord = [2,1]
    return coord

def solve(state):
    goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

    a = A_star(state, goal_state)
    moves = 0

    i = len(a) - 1

    while i >= 0:
        # print(a[i])
        print(a[i], 'h='+ str(heuristic(a[i])), 'moves: ' + str(moves))

        i = i - 1
        moves += 1

    pq = []
# perform A star algorithm
# return the reconstruct path
def A_star(start, goal_state):
    open = []
    closed = []
    open_heap = []

    a = -1 # parent index

    current = start
    open.append(current)

    # dict with parents as key and children as value
    parents = dict()


    heapq.heappush(open_heap, (heuristic(start), start, (0, heuristic(start), a)))

    # used for reconstruct_path1 but ends with higher time complexity
    # static_heap.append((heuristic(start), start, (0, heuristic(start), a)))

    while len(open_heap) > 0:

        f_score, current, info = heapq.heappop(open_heap)

        open.remove(current)
        closed.append(current)

        if current == goal_state:
            return reconstruct_path(parents, current)

        for node in succ(current):

            # update the g score and f score
            temp_g_score = info[0] + 1
            temp_f_score = temp_g_score + heuristic(node)

            # parent_index = static_heap.index((f_score, current, info))

            if node not in closed:
                parents[str(node)] = current

                open.append(node)
                heapq.heappush(open_heap,(temp_f_score, node, (temp_g_score, heuristic(node), a + 1)))

                # used for reconstruct_path1 but ends with higher time complexity
                # static_heap.append((temp_f_score, node, (temp_g_score, heuristic(node), parent_index)))




"""
def reconstruct_path1(current, static_heap, info):
    total_path = [(info[0] + info[1], current, info)]
    while info[2] != -1:
        current = static_heap[info[2]]
        info = static_heap[info[2]][2]
        total_path.append(current)
    return total_path
"""


def reconstruct_path(parents, current):

    total_path = [current]

    while str(current) in parents.keys():
        current = parents[str(current)]
        total_path.append(current)

    return total_path


if __name__ == "__main__":
    puzzle = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    print_succ([1, 2, 3, 4, 5, 0, 6, 7, 8])
    print_succ([8, 7, 6, 5, 4, 3, 2, 1, 0])

    solve(puzzle)