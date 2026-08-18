import numpy as np
from othello.board import Board, Color


def test_is_valid_move():
    board = Board()
    assert not board.is_valid_move(3, 5, Color.BLACK)


def test_is_valid_move_fixed():
    s = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.int32
    )
    board = Board()
    board.board = s
    assert not board.is_valid_move(3, 6, Color.BLACK)
    for k in range(2,6+1):
        assert board.is_valid_move(2, k, Color.BLACK)

def test_endgame():
    s = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
       [-1., -1., -1.,  1.,  1.,  1.,  1., -1.],
       [-1., -1., -1., -1., -1.,  1.,  1., -1.],
       [-1., -1., -1., -1., -1.,  1.,  1., -1.],
       [ 0., -1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1.,  1.,  1., -1.,  1., -1.],
       [-1., -1., -1.,  1.,  1.,  1.,  1., -1.],
       [-1., -1., -1.,  1.,  1.,  1.,  1., -1.]], dtype=np.int32)
    board = Board()
    board.board = s
    assert not board.is_valid_move(4, 0, Color.BLACK)
    assert board.is_valid_move(4, 0, Color.WHITE)