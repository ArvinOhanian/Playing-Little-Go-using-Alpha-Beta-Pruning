
from copy import deepcopy
import sys

#4x13
#5x7
max_depth = 4
branching_factor = 7
best_order_list = [[1,3],[1,2],[2,3], [1,1], [3,3], [3,1],  [3,2], [2,1],  [2,2], [0,3], [0,1], [4,3], [4,1], [1,0], [1,4], [3,0], [3,4], [0,2], [2,0],
                   [2,4], [4,2], [4,4], [4,0], [0,0], [0,4]]


class AlphaBeta():
    
    def __init__(self, n=5):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        #self.previous_board = None # Store the previous board
        self.X_move = True # X chess plays first
        self.died_pieces = [] # Intialize died pieces to be empty
        self.n_move = 0 # Trace the number of moves
        self.max_move = n * n - 1 # The max movement of a Go game
        self.komi = n/2 # Komi rule
        self.verbose = False # Verbose only when there is a manual player
        
    
    def Alpha_Max(self, cur_state, alpha, beta, depth, my_piece):
        if my_piece == 1:
            op_piece = 1
        else:
            op_piece = 2
        if depth == max_depth:
            
            return tuple((self.Evaluate(cur_state, my_piece), None, -float('infinity'), float('infinity')))
        next_move = None
        next_state = None
        tot_score = -float('infinity')
        
        pass_score = self.Evaluate(cur_state, my_piece)#evaluate position if I were to pass
        cur_branches=0
        for i,j in best_order_list:
            if cur_state[i][j] == 0 and go.valid_place_check(i, j, my_piece) and cur_branches < branching_factor:
                cur_branches +=1 
                next_state = self.Next_State(cur_state, i, j, my_piece)
                
                min_out = self.Beta_Min(next_state, alpha, beta, depth+1, my_piece)
                play_score = min_out[0]
                alpha = min_out[2]
                beta = min_out[3]
                
                cur_score = max(tot_score, play_score , pass_score)
                
                if cur_score > tot_score and cur_score > pass_score:#if play score is the highest
                    tot_score = cur_score
                    next_move = tuple((i,j))
                    
                    if tot_score >= beta:
                        
                        return tuple((tot_score, next_move, alpha, beta))
                    alpha = max(alpha, tot_score)
                    
                if cur_score > tot_score and cur_score > play_score:#if pass score is the highest
                    tot_score = cur_score
                    next_move = tuple((i,j))
                    
                    if tot_score >= beta:
                        
                        return tuple((tot_score, next_move, alpha, beta))
                    alpha = max(alpha, tot_score)
                        
        return tuple((tot_score, next_move, alpha, beta))
        
        
    
    def Beta_Min(self, cur_state, alpha, beta, depth, my_piece):
        if my_piece == 1:
            op_piece = 2
        else:
            op_piece = 1
        if depth == max_depth:
            
            return tuple((self.Evaluate(cur_state, my_piece), None, -float('infinity'), float('infinity')))
        next_move = None
        next_state = None
        tot_score = float('infinity')
        
        pass_score = self.Evaluate(cur_state, my_piece)#evaluate position if opponent were to pass
        cur_branches = 0
        for i,j in best_order_list:
            if cur_state[i][j] == 0 and go.valid_place_check(i, j, op_piece) and cur_branches < branching_factor:
                cur_branches+=1
                next_state = self.Next_State(cur_state, i, j, op_piece)
                max_out = self.Alpha_Max(next_state, alpha, beta, depth+1, my_piece)
                play_score = max_out[0]
                alpha = max_out[2]
                beta = max_out[3]
                
                cur_score = min(tot_score, play_score , pass_score)
                
                if cur_score < tot_score and cur_score<pass_score:#if play score is lowest
                    tot_score = cur_score
                    next_move = tuple((i,j))
                   
                    if tot_score <= alpha:
                        return tuple((tot_score, next_move, alpha, beta))
                    beta = min(beta, tot_score)
                    
                if cur_score < tot_score and cur_score<play_score:#if pass score is lowest
                    tot_score = pass_score
                    next_move = 'PASS'
                    
                    if tot_score <= alpha:
                        return tuple((tot_score, next_move, alpha, beta))
                    beta = min(beta, tot_score)
                        
        return tuple((tot_score, next_move, alpha, beta))
    
    def Evaluate(self, cur_state, my_piece):#evaluation heuristic for non-leaf nodes(always evaluated for my_piece)
        num_my_pieces = 0
        num_op_pieces = 0
        if my_piece == 1:
            op_piece = 2
            #num_op_pieces+=2.5
        else:
            op_piece = 1
            #num_my_pieces+=2.5
        tot_pat1 = 0
        tot_pat2 = 0
        tot_pat3 = 0
        
        op_tot_pat1 = 0
        op_tot_pat2 = 0
        op_tot_pat3 = 0
        
        my_libs = 0
        op_libs = 0
        
        num_my_edge_pieces = 0
        num_op_edge_pieces = 0
        for row in range(5):    #Q1 Q2
            for col in range(5):#Q3 Q4
                
                
                                
                if row<4 and col < 4:
                    Q1 = cur_state[row][col]
                    Q2 = cur_state[row][col+1]
                    Q3 = cur_state[row+1][col]
                    Q4 = cur_state[row+1][col+1]
                    
                    
                    if self.Pat1_Check(my_piece,Q1, Q2, Q3, Q4) == True:
                        tot_pat1+=1
                       
                    if self.Pat2_Check(my_piece,Q1, Q2, Q3, Q4) == True:
                        tot_pat2+=1
                    
                    if self.Pat3_Check(my_piece,Q1, Q2, Q3, Q4) == True:
                        tot_pat3+=1
                        
                if row == 0 or row == 4 or col==0 or col==4:
                    if cur_state[row][col] == my_piece:
                        num_my_edge_pieces += 1
                    elif cur_state[row][col] == op_piece:
                        num_op_edge_pieces += 1
                    
                        
                if cur_state[row][col] == 0:
                    if row > 0:
                        if cur_state[row-1][col] == my_piece:
                            my_libs+=1
                        elif cur_state[row-1][col] == op_piece:
                            op_libs+=1
                        
                    if row < 4:
                        if cur_state[row+1][col] == my_piece:
                            my_libs+=1
                        elif cur_state[row+1][col] == op_piece:
                            op_libs+=1
                            
                    if col > 0:
                        if cur_state[row][col-1] == my_piece:
                            my_libs+=1
                        elif cur_state[row][col-1] == op_piece:
                            op_libs+=1
                    
                    if col < 4:
                        if cur_state[row][col+1] == my_piece:
                            my_libs+=1
                        elif cur_state[row][col+1] == op_piece:
                            op_libs+=1
                            
                elif cur_state[row][col] == my_piece:
                    num_my_pieces += 1
                elif cur_state[row][col] == op_piece:
                    num_op_pieces += 1
        
        num_edges = num_my_edge_pieces - num_op_edge_pieces
        num_pieces = num_my_pieces - num_op_pieces
        num_liberties = my_libs - op_libs               
        pat_score = (tot_pat1 - tot_pat2 + (2*tot_pat3))/4
        
        
        state_score = min(max(num_liberties, -4), 4) + (-4 * pat_score) +(5*num_pieces) - num_edges
        #state_score = num_liberties +(5*num_pieces) - num_edges
        
        return state_score
    
    #pattern 1: - - or - - or - X or X - 
    #           - X    X -    - -    - -
    def Pat1_Check(self,my_piece, Q1, Q2, Q3, Q4):#check if any of the four cases of pattern 1 exist in the four given coords, return true if any of the four cases are present
        if Q1 == my_piece and Q2==0 and Q3==0 and Q4==0:
            return True
        elif Q1 == 0 and Q2==my_piece and Q3==0 and Q4==0:
            return True
        elif Q1 == 0 and Q2==0 and Q3==my_piece and Q4==0:
            return True
        elif Q1 == 0 and Q2==0 and Q3==0 and Q4==my_piece:
            return True
        return False
        
    
    #pattern 2: - X or X X or X - or X X 
    #           X X    - X    X X    X - 
    def Pat2_Check(self,my_piece, Q1, Q2, Q3, Q4):#check if any of the four cases of pattern 2 exist in the four given coords, return true if any of the four cases are present
        if Q1 == my_piece and Q2==my_piece and Q3==my_piece and Q4==0:
            return True
        elif Q1 == my_piece and Q2==my_piece and Q3==0 and Q4==my_piece:
            return True
        elif Q1 == my_piece and Q2==0 and Q3==my_piece and Q4==my_piece:
            return True
        elif Q1 == 0 and Q2==my_piece and Q3==my_piece and Q4==my_piece:
            return True
        return False
   
    #pattern 3: X - or - X 
    #           - X    X - 
    def Pat3_Check(self,my_piece, Q1, Q2, Q3, Q4):#check if any of the four cases of pattern 2 exist in the four given coords, return true if any of the four cases are present
        if Q1 == my_piece and Q2==0 and Q3==0 and Q4==my_piece:
            return True
        elif Q1 == 0 and Q2==my_piece and Q3==my_piece and Q4==0:
            return True
        
        return False
    #pattern 4: X when one of these pieces is on an edge
    #           X
    def Pat4_Check(self, my_piece, Q1, Q2):
        if Q1 == my_piece and Q2==my_piece:
            return True
        return False
        
    #pattern 5: X X when one of these pieces is on an edge
    
    def Pat5_Check(self, my_piece, Q1, Q2):
        if Q1 == my_piece and Q2==my_piece:
            return True
        return False
    
    def Next_State(self, cur_state, row, col, piece_type):#takes current state and return next state if move (row, col) is played.
        if piece_type == 1:
            dead_piece = 2
        else:
            dead_piece = 1
        next_state = [[0 for i in range(5)] for j in range(5)]
        for i in range(5):
            for j in range(5):
                next_state[i][j] = cur_state[i][j]
        next_state[row][col] = piece_type
        
        for i in range(5):
            for j in range(5):
                # Check if there is a piece at this position:
                if next_state[i][j] == dead_piece:
                    # The piece die if it has no liberty
                    flag = 0
                    ally_members = self.my_ally_dfs(i, j, next_state)
                    for member in ally_members:
                        neighbors = self.my_detect_neighbor(member[0], member[1])
                        for piece in neighbors:
                            # If there is empty space around a piece, it has liberty
                            if board[piece[0]][piece[1]] == 0:
                                flag += 1
                    if flag == 0:
                        next_state[i][j] == 0
       
        
    
        return next_state
    
    ############################ THIS CODE IS FROM HOST.PY, WRITE.PY, AND READ.PY FILES, DONT MARK AS PLAGARISM####################################
    def my_detect_neighbor_ally(self, i, j, state):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = state
        neighbors = self.my_detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def my_ally_dfs(self, i, j, state):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.my_detect_neighbor_ally(piece[0], piece[1], state)
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members
    
    def my_detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < 5 - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < 5 - 1: neighbors.append((i, j+1))
        return neighbors
    
    def init_board(self, n):
        '''
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        '''
        board = [[0 for x in range(n)] for y in range(n)]  # Empty space marked as 0
        # 'X' pieces marked as 1
        # 'O' pieces marked as 2
        self.board = board
        self.previous_board = deepcopy(board)

    def set_board(self, piece_type, previous_board, board):
        '''
        Initialize board status.
        :param previous_board: previous board state.
        :param board: current board state.
        :return: None.
        '''

        # 'X' pieces marked as 1
        # 'O' pieces marked as 2

        for i in range(self.size):
            for j in range(self.size):
                if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                    self.died_pieces.append((i, j))

        # self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def copy_board(self):
        '''
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        '''
        return deepcopy(self)

    def detect_neighbor(self, i, j):
        '''
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0: neighbors.append((i-1, j))
        if i < len(board) - 1: neighbors.append((i+1, j))
        if j > 0: neighbors.append((i, j-1))
        if j < len(board) - 1: neighbors.append((i, j+1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        '''
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        '''
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        '''
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        '''
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        '''
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        '''
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        '''
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        '''
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i,j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        '''
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        '''

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces: return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        '''
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        '''
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        '''
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        '''
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        '''
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        '''   
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
            return False
        
        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print('Invalid placement. There is already a chess in this position.')
            return False
        
        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print('Invalid placement. No liberty found in this position.')
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print('Invalid placement. A repeat move not permitted by the KO rule.')
                return False
        return True
        
    def update_board(self, new_board):
        '''
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        '''   
        self.board = new_board

    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)

    def game_end(self, piece_type, action="MOVE"):
        '''
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        '''

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        '''
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        '''

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt          

    def judge_winner(self):
        '''
        Judge the winner of the game by number of pieces for each player.

        :param: None.
        :return: piece type of winner of the game (0 if it's a tie).
        '''        

        cnt_1 = self.score(1)
        cnt_2 = self.score(2)
        if cnt_1 > cnt_2 + self.komi: return 1
        elif cnt_1 < cnt_2 + self.komi: return 2
        else: return 0
        
    def play(self, player1, player2, verbose=False):
        '''
        The game starts!

        :param player1: Player instance.
        :param player2: Player instance.
        :param verbose: whether print input hint and error information
        :return: piece type of winner of the game (0 if it's a tie).
        '''
        self.init_board(self.size)
        # Print input hints and error message if there is a manual player
        if player1.type == 'manual' or player2.type == 'manual':
            self.verbose = True
            print('----------Input "exit" to exit the program----------')
            print('X stands for black chess, O stands for white chess.')
            self.visualize_board()
        
        verbose = self.verbose
        # Game starts!
        while 1:
            piece_type = 1 if self.X_move else 2

            # Judge if the game should end
            if self.game_end(piece_type):       
                result = self.judge_winner()
                if verbose:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(player + " makes move...")

            # Game continues
            if piece_type == 1: action = player1.get_input(self, piece_type)
            else: action = player2.get_input(self, piece_type)

            if verbose:
                player = "X" if piece_type == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], piece_type):
                    if verbose:
                        self.visualize_board() 
                    continue

                self.died_pieces = self.remove_died_pieces(3 - piece_type) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board() # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move # Players take turn
    
    def judge(n_move, verbose=False):

        N = 5
       
        piece_type, previous_board, board = AlphaBeta.readInput(N)
        go = AlphaBeta(N)
        go.verbose = verbose
        go.set_board(piece_type, previous_board, board)
        go.n_move = n_move
        try:
            action, x, y = go.readOutput()
        except:
            print("output.txt not found or invalid format")
            sys.exit(3-piece_type)
    
        if action == "MOVE":
            if not go.place_chess(x, y, piece_type):
                print('Game end.')
                print('The winner is {}'.format('X' if 3 - piece_type == 1 else 'O'))
                sys.exit(3 - piece_type)
    
            go.died_pieces = go.remove_died_pieces(3 - piece_type)
    
        if verbose:
            go.visualize_board()
            print()
    
        if go.game_end(piece_type, action):       
            result = go.judge_winner()
            if verbose:
                print('Game end.')
                if result == 0: 
                    print('The game is a tie.')
                else: 
                    print('The winner is {}'.format('X' if result == 1 else 'O'))
            sys.exit(result)
    
        piece_type = 2 if piece_type == 1 else 1
    
        if action == "PASS":
            go.previous_board = go.board
        go.writeNextInput(piece_type, go.previous_board, go.board)
    
        sys.exit(0)
        
    
    ############################ END OF HOST.PY CODE####################################
        
    def Move(self, my_piece, cur_board):#final output
        call = self.Alpha_Max(cur_board, -float('infinity'),float('infinity'),0, my_piece)
        my_move = call[1]
        my_score = call[0]
        if my_move == 'PASS':
            return 'PASS'
        out_row = my_move[0]
        out_col = my_move[1]
        
        return tuple((out_row, out_col))
        
###########TAKEN FROM READ AND WRITE.PY################        
def readInput( n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def readOutput( path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y
def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
    	res = "PASS"
    else:
	    res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
	with open(path, 'w') as f:
		f.write("PASS")

def writeNextInput(piece_type, previous_board, board, path="input.txt"):
	res = ""
	res += str(piece_type) + "\n"
	for item in previous_board:
		res += "".join([str(x) for x in item])
		res += "\n"
        
	for item in board:
		res += "".join([str(x) for x in item])
		res += "\n"

	with open(path, 'w') as f:
		f.write(res[:-1]);
###########END OF TAKEN FROM READ AND WRITE.PY################    

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = AlphaBeta(N)
    
    go.set_board(piece_type, previous_board, board)
    
    alg = AlphaBeta()
    action = alg.Move(piece_type, board)
    writeOutput(action)