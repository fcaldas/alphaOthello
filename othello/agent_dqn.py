"""A legal-move-masked DQN player for Othello."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .board import Board, Color
from .player import Player


@dataclass(frozen=True)
class NetworkConfig:
    channels: int = 64


class OthelloQNetwork(nn.Module):
    """Maps a player-relative board state to Q-values for its 64 squares."""

    def __init__(self, config: NetworkConfig = NetworkConfig()) -> None:
        super().__init__()
        self.config = config
        channels = config.channels
        self.layers = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def encode_board(board: Board, color: Color) -> np.ndarray:
    """Encode the board from ``color``'s perspective as own/opponent planes."""
    position = board.get_board()
    return np.stack(
        [
            (position == int(color)).astype(np.float32),
            (position == -int(color)).astype(np.float32),
        ]
    )


def legal_action_mask(board: Board, color: Color) -> np.ndarray:
    mask = np.zeros(64, dtype=np.bool_)
    for row, column in board.valid_moves(color):
        mask[row * 8 + column] = True
    return mask


def action_to_position(action: int) -> tuple[int, int]:
    if not 0 <= action < 64:
        raise ValueError(f"Action must be in [0, 63], got {action}")
    return divmod(action, 8)


class DQNPlayer(Player):
    """A Player adapter around a trained :class:`OthelloQNetwork`."""

    def __init__(
        self,
        color: Color,
        network: OthelloQNetwork,
        device: torch.device | str = "cpu",
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(color)
        self.network = network.to(device)
        self.device = torch.device(device)
        self.epsilon = epsilon
        self.network.eval()

    @classmethod
    def from_checkpoint(
        cls, color: Color, checkpoint_path: str | Path, device: torch.device | str = "cpu"
    ) -> "DQNPlayer":
        # Checkpoints are local artifacts produced by ``save_checkpoint``. Do not
        # load a checkpoint from an untrusted source.
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = NetworkConfig(**checkpoint["network_config"])
        network = OthelloQNetwork(config)
        network.load_state_dict(checkpoint["model_state"])
        return cls(color, network, device)

    def choose_action(self, board: Board) -> int | None:
        mask = legal_action_mask(board, self.color)
        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            return None
        if random.random() < self.epsilon:
            return int(random.choice(legal_actions))

        state = torch.from_numpy(encode_board(board, self.color)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.network(state).squeeze(0)
            q_values[~torch.from_numpy(mask).to(self.device)] = -torch.inf
            return int(torch.argmax(q_values).item())

    def play(self, board: Board) -> bool:
        action = self.choose_action(board)
        if action is None:
            return False
        row, column = action_to_position(action)
        return board.play(row, column, self.color)


def save_checkpoint(
    path: str | Path, network: OthelloQNetwork, games_completed: int
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "network_config": asdict(network.config),
            "model_state": network.state_dict(),
            "games_completed": games_completed,
        },
        path,
    )
