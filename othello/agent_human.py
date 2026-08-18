from .board import Board, Color
from .player import Player

class HumanPlayer(Player):

    def __init__(self, color: Color):
        super().__init__(color)

    def play(self, board: Board) -> bool:
        if not board.has_valid_move(self.get_color()):
            print("No legal moves; passing.")
            return False

        while True:
            try:
                indices = input("Position you want to play ROW COL: ").split()
                irow, icol = [int(ix) for ix in indices]
            except ValueError:
                print("Enter exactly two integer coordinates, e.g. '2 3'.")
                continue
            if board.play(irow, icol, self.get_color()):
                return True
            print("That is not a legal move. Try again.")
