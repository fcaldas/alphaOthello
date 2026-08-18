# alphaOthello

`alphaOthello` is a small Othello rules engine and an experiment in training a
neural-network player through self-play. The immediate aim is not to reproduce
a world-class Othello engine: it is to build a complete, inspectable learning
loop that can first beat a random opponent and then improve through evaluation
and iteration.

The game engine enforces Othello's rules, including disc flipping, legal moves,
forced passes, and the rule that the game ends only when neither player has a
legal move. Human, random, and learned players share the same `Player`
interface, so the trained agent can play either colour against a person or
another agent.

## How the agent sees Othello

The board is always encoded from the perspective of the side about to move.
This means the same network can play both Black and White without learning two
separate versions of the game. The state is a `float32` tensor of shape
`(2, 8, 8)`:

| Plane | Value `1` means |
| --- | --- |
| `state[0]` | A disc belonging to the player whose turn it is |
| `state[1]` | A disc belonging to that player's opponent |

Empty squares are zero in both planes. The side to move is implicit in this
representation: the network always treats plane 0 as “mine”. This player-
relative encoding also lets a position with colours swapped share the same
meaning.

An action is one of 64 board squares in row-major order:
`action = row * 8 + column`. Before the agent chooses an action, the rules
engine produces a 64-element legal-action mask. Illegal actions are excluded
both when selecting a move and when computing future Q-values. Consequently,
the neural network cannot intentionally play on an occupied square or make a
non-capturing move. If the mask is empty, the game engine performs a pass.

## What DQN is learning

The network predicts a value for each possible move. In reinforcement-learning
notation, $Q_\theta(s, a)$ is the expected discounted final outcome from
choosing action $a$ in state $s$, then following the current policy; $\theta$
is the neural network's trainable parameter vector.

An episode is one complete self-play game. A transition stores the encoded
state, chosen legal action, reward, next encoded state, next legal-action mask,
and whether the game ended. Intermediate moves receive reward `0`; the player
who makes the final move receives `+1` for a win, `0` for a draw, and `-1` for
a loss.

### Bellman target with alternating players

Let $\mathcal{A}(s)$ be the legal actions in $s$, $r_t$ the observed reward,
and $d_t\in\{0,1\}$ indicate a terminal transition. Let
$\sigma_t=-1$ when the next state belongs to the opponent and
$\sigma_t=+1$ when the opponent must pass and the same player moves again.
The target used by the trainer is:

$$
y_t =
\begin{cases}
r_t, & d_t = 1, \\
r_t + \gamma\,\sigma_t\displaystyle\max_{a' \in \mathcal{A}(s_{t+1})}
Q_{\bar{\theta}}(s_{t+1}, a'), & d_t = 0.
\end{cases}
$$

Here $\gamma=0.99$ is the discount factor and $\bar{\theta}$ are the frozen
target-network parameters. The negative sign in a normal turn change converts
the opponent's best future outcome into the current player's value. If an
opponent pass leaves the turn unchanged, the sign is positive instead. This is
the zero-sum modification of the standard DQN Bellman target and is why
forced-pass positions are learned correctly.

During greedy play, the selected move is the highest-valued legal action:

$$
a_t \in \mathcal{A}(s_t), \qquad
Q_\theta(s_t, a_t) =
\max_{a \in \mathcal{A}(s_t)} Q_\theta(s_t, a).
$$

The legal-action mask is equivalent to restricting the maximisation to
$\mathcal{A}(s)$; invalid squares never contribute to either move selection or
the bootstrap target. During training this policy is made exploratory with
epsilon-greedy selection: with probability $\varepsilon$ choose uniformly from
$\mathcal{A}(s_t)$, otherwise use the equation above.

### Loss and optimisation

For a replay batch $B$, the temporal-difference error is
$\delta_i = y_i - Q_\theta(s_i,a_i)$. The code minimises the Huber (smooth L1)
loss, which is quadratic for small errors and linear for large ones:

$$
\mathcal{L}(\theta) = \frac{1}{|B|}\sum_{i \in B}
\begin{cases}
\tfrac{1}{2}\delta_i^2, & |\delta_i| < 1, \\
|\delta_i| - \tfrac{1}{2}, & |\delta_i| \geq 1.
\end{cases}
$$

AdamW updates $\theta$ to reduce $\mathcal{L}$, and gradients are clipped to
norm 10 before each update. Huber loss and the delayed target network help
avoid a few inaccurate early predictions destabilising the whole run.

The learner samples old transitions at random from a replay buffer rather than
only fitting the most recent game. This reduces correlation between updates.
A second, periodically copied target network supplies the future values, which
makes DQN updates substantially less unstable than using the network currently
being changed.

## Neural network

`OthelloQNetwork` is deliberately small—about 571,000 trainable parameters—so
it is practical to train locally on an Apple-silicon MacBook Air:

```text
(batch, 2, 8, 8)
  → Conv2d(2, 64, 3×3, padding=1) → ReLU
  → Conv2d(64, 64, 3×3, padding=1) → ReLU
  → Flatten                         # 64 × 8 × 8 = 4,096 features
  → Linear(4,096, 128) → ReLU
  → Linear(128, 64)                 # one Q-value per board square
```

The convolution layers let nearby discs and lines of discs interact before the
network makes a value prediction. The output is not a probability distribution:
it is 64 Q-values. The legal-action mask converts it into a valid decision by
choosing the highest-valued legal square. During training, epsilon-greedy
exploration sometimes chooses a random legal square so the agent continues to
discover alternatives.

## Training procedure

The default run trains one shared network as both sides in self-play:

1. Start a legal Othello game with Black to move.
2. Select legal moves with epsilon-greedy exploration; epsilon decays from
   `1.0` to `0.05` over the first 60% of the requested games.
3. Save every move transition in a replay buffer of up to 100,000 transitions.
4. After 2,000 transitions are available, train with random batches of 256
   transitions using Huber loss and AdamW (`learning_rate=3e-4`).
5. Copy the online network to the target network every 1,000 optimisation
   steps; gradients are clipped to a norm of 10.
6. Every 10,000 games, play 100 evaluation games against `RandomPlayer`,
   alternating the learned player's colour, then save a checkpoint.

PyTorch uses the `mps` device automatically when it is available on Apple
silicon; otherwise the trainer uses CPU. A checkpoint contains the network
configuration, learned weights, and number of completed games. Only load local
checkpoints you trust.

## Play a game

For example, a human can play Black against the baseline random player:

```python
from othello.agent_human import HumanPlayer
from othello.agent_random import RandomPlayer
from othello.board import Color
from othello.game import OthelloGame

game = OthelloGame()
game.add_player(HumanPlayer(Color.BLACK))
game.add_player(RandomPlayer(Color.WHITE))
print(game.run(trace=True))
```

Install the locked development environment and run the tests with:

```bash
uv sync
uv run pytest -q
```

## Train and evaluate

```bash
uv run python -m othello.train --games 200000 --checkpoint checkpoints/othello_dqn.pt
```

The default 200,000-game run is a good first experiment. Its periodic output
reports replay-buffer size, evaluation win rate against random, and (once
learning begins) training loss. A short smoke run checks that the environment,
PyTorch device, and checkpoint path work:

```bash
uv run python -m othello.train --games 100 --evaluate-every 100 --evaluation-games 10
```

To use a resulting checkpoint in a game, load it into `DQNPlayer` and add it
as either colour:

```python
from othello.agent_dqn import DQNPlayer

agent = DQNPlayer.from_checkpoint(Color.WHITE, "checkpoints/othello_dqn.pt", "mps")
game.add_player(agent)
```
