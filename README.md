# Playing-Little-Go-using-Alpha-Beta-Pruning
This project plays the game of Little Go using the Alpha-Beta Minimax Algorithm. Little Go is the same as the traditional game of GO, but is played on a 5x5 board rather than a 19x19 board.

![image](https://user-images.githubusercontent.com/34993121/146104698-162b0874-1088-468e-87ae-188b12da656f.png)

The game is run by a host program that takes in the move from player 1 and makes changes to the board. This new board is given to player 2 who makes their move and reports back to the host. This is illustrated above.

![image](https://user-images.githubusercontent.com/34993121/146105395-b9e15116-b5f8-46b4-80b2-b76d74eabdec.png)

The board is a 5x5 grid where a 0 represents an empty cell, 1 represents a black stone, and 2 represents a white stone.

#How to Play Little GO

Each player takes their turn by placing a stone on a 5x5 grid. Each stone has four adjacent spots which if empty are called liberties. If all four adjacent spots are filled by other stones, then the stone will be captured since it has run out of liberties. If an adjacent stone is of the same color, then the liberities are shared among the connected stones. The corners and edges of the grid have less liberties since they have are adjacent to the ends of the board.

![image](https://user-images.githubusercontent.com/34993121/146105897-07209ffa-b585-495f-a7cf-dc829b5f7328.png)

Here are some common ways of capturing in Go.
