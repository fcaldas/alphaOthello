import random
from .board import Board, Color
from .player import Player

class RandomPlayer(Player):

    def __init__(self, color: Color):
        super().__init__(color)

    def play(self, board: Board) -> bool:
        other_pos = board.valid_moves(self.get_color())
        if not other_pos:
            return False
        # Pick one at random and play it
        choice = random.choice(other_pos)
        return board.play(choice[0], choice[1], self.color)
