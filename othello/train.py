"""Self-play training entry point: ``python -m othello.train --games 200000``."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .agent_dqn import (
    DQNPlayer,
    NetworkConfig,
    OthelloQNetwork,
    action_to_position,
    encode_board,
    legal_action_mask,
    save_checkpoint,
)
from .agent_random import RandomPlayer
from .board import Board, Color
from .game import OthelloGame


@dataclass(frozen=True)
class TrainingConfig:
    games: int = 200_000
    replay_size: int = 100_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    warmup_steps: int = 2_000
    target_update_steps: int = 1_000
    evaluation_every: int = 10_000
    evaluation_games: int = 100
    checkpoint_path: str = "checkpoints/othello_dqn.pt"


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool
    next_sign: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.transitions, batch_size)


def preferred_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def terminal_reward(board: Board, color: Color) -> float:
    difference = board.score(color) - board.score(Color(-int(color)))
    return float(np.sign(difference))


def train_batch(
    network: OthelloQNetwork,
    target_network: OthelloQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    batch = replay.sample(config.batch_size)
    states = torch.from_numpy(np.stack([item.state for item in batch])).to(device)
    actions = torch.tensor([item.action for item in batch], device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([item.done for item in batch], dtype=torch.bool, device=device)
    signs = torch.tensor([item.next_sign for item in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).to(device)
    next_masks = torch.from_numpy(np.stack([item.next_mask for item in batch])).to(device)

    q_values = network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target_values = target_network(next_states).masked_fill(~next_masks, -torch.inf).max(1).values
        target_values = torch.where(dones, torch.zeros_like(target_values), target_values)
        expected_values = rewards + config.gamma * signs * target_values

    loss = nn.functional.smooth_l1_loss(q_values, expected_values)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def collect_self_play_game(
    network: OthelloQNetwork, epsilon: float, replay: ReplayBuffer, device: torch.device
) -> int:
    board = Board()
    color = Color.BLACK
    moves = 0
    network.eval()

    while not board.is_game_over():
        mask = legal_action_mask(board, color)
        if not mask.any():
            color = Color(-int(color))
            continue
        state = encode_board(board, color)
        player = DQNPlayer(color, network, device, epsilon)
        action = player.choose_action(board)
        assert action is not None
        row, column = action_to_position(action)
        assert board.play(row, column, color)
        moves += 1

        if board.is_game_over():
            replay.append(
                Transition(state, action, terminal_reward(board, color), state, mask, True, 0.0)
            )
            break

        next_color = Color(-int(color))
        if not board.has_valid_move(next_color):
            next_color = color
        replay.append(
            Transition(
                state,
                action,
                0.0,
                encode_board(board, next_color),
                legal_action_mask(board, next_color),
                False,
                1.0 if next_color == color else -1.0,
            )
        )
        color = next_color
    return moves


def evaluate(network: OthelloQNetwork, device: torch.device, games: int) -> float:
    """Return win rate versus random, alternating the learned player's colour."""
    wins = 0.0
    for index in range(games):
        learned_color = Color.BLACK if index % 2 == 0 else Color.WHITE
        game = OthelloGame()
        game.add_player(DQNPlayer(learned_color, network, device))
        game.add_player(RandomPlayer(Color(-int(learned_color))))
        score = game.run()
        difference = score[learned_color] - score[Color(-int(learned_color))]
        wins += 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
    return wins / games


def train(config: TrainingConfig, device: torch.device | None = None) -> OthelloQNetwork:
    device = device or preferred_device()
    network = OthelloQNetwork(NetworkConfig()).to(device)
    target_network = OthelloQNetwork(NetworkConfig()).to(device)
    target_network.load_state_dict(network.state_dict())
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_size)
    steps = 0

    print(f"Training on {device.type} for {config.games:,} games")
    for game_number in range(1, config.games + 1):
        epsilon = max(0.05, 1.0 - 0.95 * game_number / max(config.games * 0.6, 1))
        moves = collect_self_play_game(network, epsilon, replay, device)
        network.train()
        loss = None
        for _ in range(moves):
            if len(replay) >= max(config.batch_size, config.warmup_steps):
                loss = train_batch(network, target_network, optimizer, replay, config, device)
                steps += 1
                if steps % config.target_update_steps == 0:
                    target_network.load_state_dict(network.state_dict())

        if game_number % config.evaluation_every == 0 or game_number == config.games:
            network.eval()
            win_rate = evaluate(network, device, config.evaluation_games)
            save_checkpoint(config.checkpoint_path, network, game_number)
            loss_text = f", loss={loss:.4f}" if loss is not None else ""
            print(f"game={game_number:,}, replay={len(replay):,}, win_rate={win_rate:.1%}{loss_text}")
    return network


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train an Othello DQN through self-play.")
    parser.add_argument("--games", type=int, default=TrainingConfig.games)
    parser.add_argument("--checkpoint", default=TrainingConfig.checkpoint_path)
    parser.add_argument("--evaluate-every", type=int, default=TrainingConfig.evaluation_every)
    parser.add_argument("--evaluation-games", type=int, default=TrainingConfig.evaluation_games)
    args = parser.parse_args()
    return TrainingConfig(
        games=args.games,
        checkpoint_path=args.checkpoint,
        evaluation_every=args.evaluate_every,
        evaluation_games=args.evaluation_games,
    )


if __name__ == "__main__":
    train(parse_args())
