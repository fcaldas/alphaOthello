import random
import numpy as np

from .board import Board, Tile, Color
from .player import Player

class OthelloGame:

    def __init__(self) -> None:
        self.board = Board()
        self.players: dict[Color, Player | None] = {
            Color.BLACK: None,
            Color.WHITE: None
        } 

    def add_player(self, player: Player):
        self.players[player.get_color()] = player
    
    def restart(self):
        self.board = Board()
    
    def run(self, trace=False) -> dict[Color, int]:
        """
        Run the game and return a dictionary with the final score for each color
        """
        board = Board()
        move = random.choice([Color.BLACK, Color.WHITE])
        while not self.board.is_game_over():
            if trace:
                print(f"{'O' if move == Color.WHITE else 'X'} moves:")
                board.print()
            
            player = self.players[move]
            if player:
                player.play(board)
            else:
                raise ValueError(f"Missing player for {move}")
            
            move = Color.BLACK if move == Color.WHITE else Color.WHITE
        
        return {
            Color.BLACK: np.where(self.board == int(Color.BLACK))[0].shape[0],
            Color.WHITE: np.where(self.board == int(Color.WHITE))[0].shape[0]
        }
