import numpy as np
import pytest

torch = pytest.importorskip("torch")

from othello.agent_dqn import (
    DQNPlayer,
    OthelloQNetwork,
    action_to_position,
    encode_board,
    legal_action_mask,
    save_checkpoint,
)
from othello.board import Board, Color
from othello.train import ReplayBuffer, collect_self_play_game


def test_board_encoding_is_relative_to_the_player():
    board = Board()

    black_state = encode_board(board, Color.BLACK)
    white_state = encode_board(board, Color.WHITE)

    assert black_state.shape == (2, 8, 8)
    assert black_state.dtype == np.float32
    assert np.array_equal(black_state[0], white_state[1])
    assert np.array_equal(black_state[1], white_state[0])


def test_dqn_player_selects_a_legal_move_and_checkpoint_round_trips(tmp_path):
    board = Board()
    network = OthelloQNetwork()
    player = DQNPlayer(Color.BLACK, network)

    action = player.choose_action(board)
    assert action is not None
    assert legal_action_mask(board, Color.BLACK)[action]
    assert player.play(board)

    checkpoint = tmp_path / "agent.pt"
    save_checkpoint(checkpoint, network, games_completed=3)
    restored = DQNPlayer.from_checkpoint(Color.WHITE, checkpoint)
    assert isinstance(restored.network, OthelloQNetwork)


def test_action_conversion_rejects_invalid_action():
    assert action_to_position(19) == (2, 3)
    with pytest.raises(ValueError):
        action_to_position(64)


def test_self_play_collects_transitions_with_legal_follow_up_masks():
    replay = ReplayBuffer(capacity=128)
    moves = collect_self_play_game(OthelloQNetwork(), epsilon=1.0, replay=replay, device="cpu")

    assert moves > 0
    assert len(replay) == moves
    assert all(item.done or item.next_mask.any() for item in replay.transitions)
