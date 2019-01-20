import numpy as np
import random
import os
import webbrowser
from IPython.display import HTML, display, Image

# Global

tau0 = 1 / 81  # 1/d^2
epsilon = 0.1

test_sudoku2 = np.array([[5, 0, 0, 9, 0, 7, 4, 0, 3],
                       [0, 4, 0, 0, 0, 0, 6, 0, 7],
                       [8, 0, 0, 0, 0, 2, 0, 1, 0],
                       [0, 0, 8, 3, 0, 0, 0, 7, 0],
                       [0, 0, 0, 0, 7, 0, 0, 0, 0],
                       [0, 3, 0, 0, 0, 4, 2, 0, 0],
                       [0, 8, 0, 2, 0, 0, 0, 0, 1],
                       [7, 0, 3, 0, 0, 0, 0, 6, 0],
                       [6, 0, 1, 7, 0, 3, 0, 0, 5]])

test_sudoku = np.array([[0, 3, 0, 2, 0, 0, 0, 0, 6],
                        [0, 0, 0, 0, 0, 9, 0, 0, 4],
                        [7, 6, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 5, 0, 7, 0, 0],
                        [0, 0, 0, 0, 0, 1, 8, 6, 0],
                        [0, 5, 0, 4, 8, 0, 0, 9, 0],
                        [8, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 7, 6, 0, 0, 0],
                        [0, 7, 5, 0, 0, 8, 1, 0, 0]])


# UPDATE: using ONE-HOT vectors of shape(1,10) for value_sets. First value is 1 if cell has fixed value
# also the corrsponding index is set to 1:
# Example: if cell has a fxed value of 5
# [1,0,0,0,0,1,0,0,0,0]
#  0 1 2 3 4 5 6 7 8 9 <- indexes: index equals value of cell. 0 indicates if fixed

def initialize(sudoku):
    h, w = sudoku.shape  # using w as 9
    if h != w:
        print("Not square sudoku")
        # return
    new_sudoku = np.empty((h * w, w + 1))
    # new_sudoku = np.empty((h*w, w+1+20)) # if peers are added at initialization
    for i in range(w):  # ith row
        for j in range(w):  # jth column

            if sudoku[i][j] == 0:
                value_set = np.ones((1, w + 1))
                value_set[0][0] = 0
            else:
                value_set = np.zeros((1, w + 1))
                value_set[0][0] = 1
                value_set[0][sudoku[i][j]] = 1
            new_sudoku[i * w + j] = value_set
            # new_sudoku[i*w+j] = np.concatanate(value_set, find_peers(i*w+j, w)) # if peers are added at initialization
    return new_sudoku


# Every cell has its peers - cells 
def find_peers(index, range):
    h = index // range
    w = index % range
    r = np.sqrt(range)  # for 3x3 matrix range=9, r=3
    wb = w // r
    hb = h // r  # Block index - shows in which 3x3 block the cell is

    hx = np.full((range, 1), h)  # vertical 9x1 array
    hy = np.arange(0, range).reshape(-1, 1)  # make it vertical
    row_peers = np.hstack((hx, hy))  # concatenate to vertical arrays
    # print(row_peers)

    wx = hy
    wy = np.full((range, 1), w)
    column_peers = np.hstack((wx, wy))  # concatenate to vertical arrays
    # column_peers2 = column_peers[:,0]*range + column_peers[:,1]
    # print(column_peers2)

    a = np.array((0, 1, 2))
    bx = np.repeat([a], r, axis=0).reshape(range, 1) + hb * r
    by = np.repeat(a, r).reshape(-1, 1) + wb * r
    block_peers = np.hstack((bx, by))  # concatenate to vertical arrays
    # print(block_peers)

    peers = np.unique(np.vstack((row_peers, column_peers, block_peers)), axis=0)
    peers = np.delete(peers, np.where((peers[:, 0] == h) * (peers[:, 1] == w))[0][0], axis=0)
    peers_1D = np.int8(peers[:, 0] * range + peers[:, 1])

    return peers_1D


def constraint_propagation2(sudoku):
    c, r = sudoku.shape  # cell index, range of values for each sell
    r = r - 1  # First column refers to fixed value. IMPORTANT: if the initialize funtion is changed so that the
    # sudoku matrix stores peers, this value need to be redefined
    change = 1
    while change == 1:  # go through the loop until there has been zero changes:
        change = 0
        for i in range(c):
            peers = find_peers(i, r)  # Always same output, would be okay to store in a variable. Maybe it should be
            # added in the sudoku initialization.
            if sudoku[i][0] == 1 and np.sum(sudoku[i]) == 2:  # Cell value is fixed
                value = np.where(sudoku[i][1:1 + r] == 1)[0][0] + 1
                for p in peers:
                    if np.sum(sudoku[p][1:1 + r]) == 0:
                        return sudoku, False
                    if sudoku[p][0] == 0 and sudoku[p][value] == 1:  # if peer is not fixed, remove cell's value from its value_set
                        sudoku[p][value] = 0
                        if np.sum(sudoku[p][1:1 + r]) == 1:
                            sudoku[p][0] = 1  # if peers value_set has decreased to 1 value, declare peer fixed
                        change = 1
                    elif sudoku[p][0] == 1:  # if peer is fixed
                        # print(sudoku[p][1:1+r])
                        peer_value = np.where(sudoku[p][1:1 + r] == 1)[0][0] + 1
                        if sudoku[i][peer_value] == 1:  # if peers value is in cells value_set
                            # print(value, peer_value)
                            sudoku[i][peer_value] = 0  # remove peers value from cells value_set
                            if np.sum(sudoku[i][1:1 + r]) == 1:
                                sudoku[i][0] = 1  # if cells value_set has decreased to 1 value, declare cell fixed
                            change = 1
    return sudoku, True


def possible_values(cell):  # If a cel lis fixed, it returns the cells value
    values = np.where(cell[
                      1:] == 1) + 1  # find all indexes where value equals 1, leave out the first value that
    # indicates if cell is fixed or not
    return values


def get_cell_value(cell):
    if cell[0] == 1:  # Check if cell is fixed
        values = np.where(cell[
                          1:] == 1)  # find all indexes where value equals 1, leave out the first value that
        # indicates if cell is fixed or not
        if values[0].shape[
            0] == 1:  # If there are more (or less) than one indeces returned, no one value can be returned
            return values[0][0] + 1  # add one to get real value of cell
    return 0  # Means not fixed, retrun 0


def check_1D(row):  # can be used for row, column or block
    r = row.shape[0]  # For 3x3 sudoku r=9, -> max value
    values = np.unique(
        [get_cell_value(cell) for cell in row])  # Use get_cell_value for each cell in row, keep unique values
    # Check if there are n (9) different values, if largest = r (9) and if smallest = 1
    if values.shape[0] == r and np.amax(values) == r and np.amin(values) == 1:
        return True  # If all correct, return True
    return False  # If some condition was not satisified, return False


def check_if_solved(sudoku):
    checksum = np.unique(np.arange(1, 10))
    c, r = sudoku.shape  # cell index, range of values for each sell
    r = r - 1  # range of values, for 3x3 sudoku r=9
    R = np.int(np.sqrt(r))  # Rank of sudoku, for 9x9 sudoku R=3

    for i in range(r):
        row = i * r + np.arange(0, r,
                                1)  # every row starts with the index that is multiple of r, contains the next r values
        column = i + np.arange(0, r * r,
                               r)  # every column starts with an index from 0 to r, the indexes for next values differ by r
        # determine which block, multiply by Rank to get the start index of that block. Inside block, first row is
        # always [0:R-1] (0,1,2 for Rank 3), add smae value to other rows. Add r for each next row start
        block = (i // R * R * r + i % R * R) + (
                    np.arange(0, R, 1, dtype=np.int_) * np.ones((R, R), dtype=np.int_)
                    + np.arange(0, R * r, r, dtype=np.int_).reshape(
                -1, 1)).flatten()  # 0, 1, 2 for 9x9 sudoku * 3x3 ones
        if (check_1D(sudoku[row]) and check_1D(sudoku[column]) and check_1D(sudoku[block])) == 0:
            return False
    return True


# Modified, should be more correct
def constraint_propagation(sudoku):
    c, r = sudoku.shape  # cell index, range of values for each sell
    r = r - 1  # First column refers to fixed value. IMPORTANT: if the initialize funtion is changed so that the
    # sudoku matrix stores peers, this value need to be redefined
    change = 1
    while change == 1:  # go through the loop until there has been zero changes:
        change = 0
        for i in range(c):  # for every cell
            peers = find_peers(i, r)  # Always same output, would be okay to store in a variable. Maybe it should be
            # added in the sudoku initialization.

            if sudoku[i][0] == 1 and np.sum(sudoku[i]) == 2:  # Cell value is fixed
                value = np.where(sudoku[i][1:1 + r] == 1)[0][0] + 1
                for p in peers:
                    if np.sum(sudoku[p][1:1 + r]) == 0:  # wrong solution
                        return sudoku, False
                    elif sudoku[p][0] == 0 and sudoku[p][value] == 1:  # if peer is not fixed, remove cell's value from its value_set
                        sudoku[p][value] = 0
                        if np.sum(sudoku[p][1:1 + r]) == 1:
                            sudoku[p][0] = 1  # if peers value_set has decreased to 1 value, declare peer fixed
                        change = 1
            elif sudoku[i][0] == 0:  # cell value is not fixed
                for p in peers:
                    if np.sum(sudoku[p][1:1 + r]) == 0:  # wrong solution
                        return sudoku, False
                    elif sudoku[p][0] == 1:  # if peer is fixed
                        peer_value = np.where(sudoku[p][1:1 + r] == 1)[0][0] + 1
                        if sudoku[i][peer_value] == 1:  # if peers value is in cells value_set
                            sudoku[i][peer_value] = 0  # remove peers value from cells value_set
                            if np.sum(sudoku[i][1:1 + r]) == 1:
                                sudoku[i][0] = 1  # if cells value_set has decreased to 1 value, declare cell fixed
                            change = 1

    return sudoku, True


def output_initial_shape(sudoku):  # Convert the sudoku to original rxr shape
    c, r = sudoku.shape  # cell index, range of values for each sell
    r = r - 1  # remove the fixed value indicator, for 9x9 sudoku r=9
    out_sudoku = np.empty(c, dtype=np.int_)  # c should equal r*r
    for i in range(c):
        out_sudoku[i] = get_cell_value(sudoku[i])
    return out_sudoku.reshape((r, r))


def roulette(pheromones, cp_values):
    cp_values = np.copy(cp_values)[-9:]
    pheromones = np.copy(pheromones) * cp_values
    r = random.random() * np.sum(pheromones)
    for i in range(len(pheromones)):
        r -= pheromones[i]
        if r < 0:
            return i + 1


def generate_pheromones(cp_sudoku):
    return np.full((81, 9), tau0)*cp_sudoku[:, 1:]


# Different local pheromone changer
class Ant:
    def __init__(self, cp_sudoku, pheromones, start):
        self.in_sudoku = np.copy(cp_sudoku)  # for comparisson
        self.cp_sudoku = np.copy(cp_sudoku)  # matrix 81x10
        self.pheromones = np.copy(pheromones)  # matrix 81x9
        self.start = start  # integer 0...81
        self.cells, self.r = self.cp_sudoku.shape  # cells =81

    def run(self):
        # Run over all cells
        for i in range(self.cells):
            # if cell value not set
            if self.cp_sudoku[self.start - i, 0] == 0 and np.sum(self.cp_sudoku[self.start - i]) != 0:
                # Choose weighted random value
                cell_random = roulette(self.pheromones[self.start - i], self.cp_sudoku[self.start - i])
                old_cp_sudoku = np.copy(self.cp_sudoku)
                # Set that value in the sudoku
                self.cp_sudoku[self.start - i] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.cp_sudoku[self.start - i, cell_random] = 1

                # Run CP
                constraint_propagation(self.cp_sudoku)
                new_cp_sudoku = np.copy(self.cp_sudoku)
                for j in range(len(new_cp_sudoku)):
                    if np.sum(new_cp_sudoku[j]) == 1:
                        new_cp_sudoku[j] = np.zeros(10)

                    # Update pheromone table
                    if old_cp_sudoku[j, 0] != new_cp_sudoku[j, 0] and np.sum(new_cp_sudoku) == 2:
                        index = np.nonzero(new_cp_sudoku[j, 1:])[0][0]
                        self.pheromones[self.start - i, index] = (1 - epsilon) * self.pheromones[
                            self.start - i, index] + epsilon * tau0

        for i in range(len(self.cp_sudoku)):
            if np.sum(self.cp_sudoku[i]) == 1:
                self.cp_sudoku[i] = np.zeros(10)

        changed_cells = np.count_nonzero(self.cp_sudoku[:, 0]) - np.count_nonzero(self.in_sudoku[:, 0])
        self.result = 81 / (81 - changed_cells)
        return


class Ant2:
    def __init__(self, cp_sudoku, pheromones, start):
        self.in_sudoku = np.copy(cp_sudoku)  # for comparisson
        self.cp_sudoku = np.copy(cp_sudoku)  # matrix 81x10
        self.pheromones = np.copy(pheromones)  # matrix 81x9
        self.start = start  # integer 0...81
        self.cells, self.r = self.cp_sudoku.shape  # cells =81
        self.result = 1

    def run(self):
        # Run over all cells
        for i in range(self.cells):
            # if cell value not set
            if self.cp_sudoku[self.start - i, 0] == 0 and np.sum(self.cp_sudoku[self.start - i]) != 0:
                # Choose weighted random value
                cell_random = roulette(self.pheromones[self.start - i], self.cp_sudoku[self.start - i])

                # Set that value in the sudoku
                self.cp_sudoku[self.start - i] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.cp_sudoku[self.start - i, cell_random] = 1

                # Run CP
                constraint_propagation(self.cp_sudoku)

                # Update pheromone table
                self.pheromones[self.start - i, cell_random - 1] = (1 - epsilon) * self.pheromones[
                    self.start - i, cell_random - 1] + epsilon * tau0

        for i in range(len(self.cp_sudoku)):
            if np.sum(self.cp_sudoku[i]) == 1:
                self.cp_sudoku[i] = np.zeros(10)

        changed_cells = np.count_nonzero(self.cp_sudoku[:, 0]) - np.count_nonzero(self.in_sudoku[:, 0])
        self.result = 81 / (81 - changed_cells)
        return


# Displays 9x9 matrix as a sudoku
def html_sudoku(sudoku, caption):
    ret = """
    <style>
        table {
            margin:1em auto;
            display:table;
            border: 4px solid;
            border-collapse: collapse;
            position: relative;
        }
        table tr {
            display:table-row;
            position: relative;
            z-index:-1;
        }
        table td{
            display:table-cell;
            padding:8px;
            border: 1px solid;
            text-align: center;
        }
        table td:nth-child(3), table td:nth-child(6){border-right: 4px solid; } /*vertical*/
        table tr:nth-child(3) td, table tr:nth-child(6) td{border-bottom: 4px solid;}  /*horizontal*/
    </style>""" + '<table><caption>' + caption + '</caption><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(("" if _ == 0 else str(_)) for _ in row)) for row in sudoku))

    return ret


def solve_sudoku(sudoku, ant_num=10, evaporation_parameter=0.9):
    sudoku = initialize(sudoku)
    sudoku, tf = constraint_propagation(sudoku)
    if not tf:
        print("Error in sudoku")
        return output_initial_shape(sudoku)

    d_tau = 1
    global_pheromones = generate_pheromones(sudoku)

    gen = 0
    while not check_if_solved(sudoku):
        gen += 1
        print(gen)

        ants = [Ant(sudoku, global_pheromones, int(random.random() * 81)) for i in
                range(ant_num)]  # int(random.random()*81)
        bestant = ants[0]
        for ant in ants:
            ant.run()
            if ant.result > bestant.result:
                bestant = ant

        d_tau = max(d_tau, bestant.result)
        global_pheromones = (1 - evaporation_parameter) * bestant.pheromones + d_tau * evaporation_parameter
        d_tau = d_tau * 0.995  # 1-f_{BVE}

        if check_if_solved(bestant.cp_sudoku):
            break

    return output_initial_shape(bestant.cp_sudoku), gen


# Running the solver and displaing in web browser
path1 = os.path.abspath('initial.html')
url1 = 'file://' + path1
with open(path1, 'w') as f:
    f.write(html_sudoku(test_sudoku, "Initial sudoku."))
    f.close()
webbrowser.open(url1)

gens = []
for i in range(10):
    solution = solve_sudoku(np.copy(test_sudoku))
    gens.append(solution[1])
    print(i)

print("Average:" + str(sum(gens)/len(gens)))

path2 = os.path.abspath('solution.html')
url2 = 'file://' + path2
with open(path2, 'w') as f:
    f.write(html_sudoku(solution[0], "Solved sudoku."))
    f.close()
webbrowser.open(url2)

print("Generation count: " + str(solution[1]))
