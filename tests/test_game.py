import numpy as np

from othello.agent_random import RandomPlayer
from othello.board import Color
from othello.game import OthelloGame


def test_game_passes_when_current_player_has_no_legal_move():
    game = OthelloGame()
    game.add_player(RandomPlayer(Color.BLACK))
    game.add_player(RandomPlayer(Color.WHITE))
    game.board.board = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [-1, -1, -1, 1, 1, 1, 1, -1],
            [-1, -1, -1, -1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1, 1, 1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, -1, 1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1],
            [-1, -1, -1, 1, 1, 1, 1, -1],
        ],
        dtype=np.int8,
    )

    score = game.run()

    assert score == {Color.BLACK: 29, Color.WHITE: 35}
    assert game.board.is_game_over()
