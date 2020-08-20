def solveSudoku(self, board):
    """
    :type board: List[List[str]]
    :rtype: void Do not return anything, modify board in-place instead.
    """
    # Populate unused values for each constraint
    # 0-8 is row, 9-17 is column, and 18-26 is sub-box
    options = [set(range(1, 10)) for i in range(27)]
    pos = []  # Empty positions

    for i in range(0, 9):
        for j in range(0, 9):
            c = board[i][j]
            if c != '.':
                num = int(c)
                options[i].remove(num)
                options[9 + j].remove(num)
                options[18 + (i // 3) * 3 + j // 3].remove(num)
            else:
                pos.append((i, j))

    self.populateSingleOption(pos, options, board)
    answer = []
    self.backtracking(pos, answer, options)
    for k in range(0, len(pos)):
        i, j = pos[k]
        num = answer[k]
        board[i][j] = str(num)

    return


def populateSingleOption(self, pos, options, board):
    min_options = 1  # Track whether there are single options populated in last iteration
    while min_options == 1:
        min_options = 10  # max value
        # Because items are removed during iteration, do so from the the end
        for k in range(len(pos) - 1, -1, -1):
            i, j = pos[k]
            option = set.intersection(
                options[i], options[9 + j], options[18 + (i // 3) * 3 + j // 3])
            if not option:
                return
            if len(option) < min_options:
                min_options = len(option)
            if len(option) == 1:
                num = option.pop()
                board[i][j] = str(num)
                pos.pop(k)  # This position is populated, remove
                options[i].remove(num)
                options[9 + j].remove(num)
                options[18 + (i // 3) * 3 + j // 3].remove(num)


def backtracking(self, pos, answer, options):
    if len(answer) == len(pos):
        return

    i, j = pos[len(answer)]
    option = set.intersection(
        options[i], options[9 + j], options[18 + (i // 3) * 3 + j // 3])

    if len(option) == 0:
        return
    else:
        for num in option:
            # Try this number
            answer.append(num)
            options[i].remove(num)
            options[9 + j].remove(num)
            options[18 + (i // 3) * 3 + j // 3].remove(num)

            self.backtracking(pos, answer, options)
            if len(answer) == len(pos):
                break

            # This number does not satisfy requirement, restore to previous state
            answer.pop()
            options[i].add(num)
            options[9 + j].add(num)
            options[18 + (i // 3) * 3 + j // 3].add(num)
