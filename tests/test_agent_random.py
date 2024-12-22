import numpy as np

from othello.agent_random import RandomPlayer
from othello.game import Color, OthelloGame

def test_it_picks_valid_move():
    game = OthelloGame()
    p_black = RandomPlayer(Color.BLACK)
    p_white = RandomPlayer(Color.WHITE)
    game.add_player(p_black)
    game.add_player(p_white)
    p_black.play(game.board)
    npboard = game.board.get_board()
    resultb = np.where(npboard == -1)
    resultw = np.where(npboard == 1)
    assert len(resultb[0]) == 4
    assert len(resultw[0]) == 1


def test_it_plays_valid_positions():
    game = OthelloGame()
    p_black = RandomPlayer(Color.BLACK)
    p_white = RandomPlayer(Color.WHITE)
    game.add_player(p_black)
    game.add_player(p_white)
    result = game.run(trace=True)