from .board import Board, Color
from .player import Player

class HumanPlayer(Player):

    def __init__(self, color: Color):
        super().__init__(color)

    def play(self, board):
        valid_move = False
        while valid_move != False:
            indices = input("Position you want to play ROW COL: ").split()
            irow, icol = [int(ix) for ix in indices]
            valid_move = board.play(irow, icol, self.get_color())

