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
        move = Color.BLACK
        while not self.board.is_game_over():
            if trace:
                print(f"{'O' if move == Color.WHITE else 'X'} moves:")
                self.board.print()
            
            player = self.players[move]
            if player is None:
                raise ValueError(f"Missing player for {move}")
            if self.board.has_valid_move(move):
                previous_board = self.board.get_board().copy()
                if not player.play(self.board) or np.array_equal(
                    previous_board, self.board.get_board()
                ):
                    raise RuntimeError(f"Player for {move} failed to make a legal move")
            elif trace:
                print(f"{'O' if move == Color.WHITE else 'X'} passes.")
            
            move = Color.BLACK if move == Color.WHITE else Color.WHITE
        
        return {
            Color.BLACK: self.board.score(Color.BLACK),
            Color.WHITE: self.board.score(Color.WHITE),
        }
