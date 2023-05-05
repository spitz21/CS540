import heapq

def manhattan(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def heuristic(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    sum = 0
    curr_position = 0

    #iterate through given state
    for num1 in state:
        if num1 != 0:
            num1_coord = (int(curr_position / 3), curr_position % 3)
            goal_position = 0
            #find goal position of current number
            for num2 in goal:
                if num2 == num1:
                    num2_coord = (int(goal_position / 3), goal_position % 3)
                else:
                    goal_position += 1
            #find and update total manhattan distance
            sum += manhattan(num1_coord, num2_coord)
        curr_position += 1

    return sum


def find_succ(state):
    #Find 0 coordinate and numerical position
    i = 0
    for x in state:
        if x == 0:
            break
        else:
            i += 1
    position = i
    coord = ((int)(i / 3), i % 3)

    #Find coordinates of all possible moves
    col = coord[0]
    row = coord[1]
    possible = []
    if col - 1 >= 0:
        possible.append((coord[0] - 1, coord[1]))
    if row - 1 >= 0:
        possible.append((coord[0], coord[1] - 1))
    if row + 1 <= 2:
        possible.append((coord[0], coord[1] + 1))
    if col + 1 <= 2:
        possible.append((coord[0] + 1, coord[1]))

    #Build all possible successor lists
    succ_list = []
    for poss in possible:
        succ = state.copy()
        num = succ[poss[0] * 3 + poss[1]]
        succ[poss[0] * 3 + poss[1]] = 0
        succ[position] = num
        succ_list.append(succ)
    return succ_list


def print_succ(state):
    succs = find_succ(state)
    for succ in succs:
        print(str(succ) + " h=" + str(heuristic(succ)))
    return


def solve(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    open = []
    closed = []
    tree = {}
    parent = -1
    root = state
    heapq.heappush(open, (heuristic(root), root, (0, heuristic(root), parent)))
    #loop until goal state found
    while len(open) != 0:
        curr_sum, current, curr_data = heapq.heappop(open)
        closed.append(current)
        #if goal state found, use parent dict to build path
        if current == goal:
            path = [current]
            while str(current) in tree.keys():
                current = tree[str(current)]
                path.insert(0, current)
            move = 0
            #print path
            for succ in path:
                print(str(succ) + " h=" + str(heuristic(succ)) + " moves: " + str(move))
                move += 1
            return
        #goal state no yet found, add successors states to heap and parent dict
        for succ in find_succ(current):
            g = curr_data[0] + 1
            sum = g + heuristic(succ)
            #add successor to heap if not already present
            if succ not in closed:
                tree[str(succ)] = current
                heapq.heappush(open, (sum, succ, (g, heuristic(succ), parent + 1)))









