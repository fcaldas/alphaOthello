import random
import numpy as np

from .board import Board, Color
from .player import Player

class RandomPlayer(Player):

    def __init__(self, color: Color):
        super().__init__(color)

    def play(self, board):
        # Get all empty blocks
        other_pos = board.get_positions_with(0)
        # Filter to the ones we can play
        other_pos = [pos for pos in other_pos if board.is_valid_move(pos[0], pos[1], self.get_color())]
        # There should always be at least one if the game is valid
        assert len(other_pos) > 0
        # Pick one at random and play it
        choice = random.choice(other_pos)
        assert board.play(choice[0], choice[1], self.color)