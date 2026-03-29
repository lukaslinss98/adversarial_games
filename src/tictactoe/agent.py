import random

import numpy as np
import torch

from tictactoe.environment import TicTacToe
from tictactoe.minimax import minimax_alpha_beta
from tictactoe.model import QNet


class MinimaxAgent:
    def __init__(self, env, marker, max_depth, pruning, deterministic):
        self.env = env
        self.marker = marker
        self.nodes_visited = 0
        self.max_depth = max_depth
        self.pruning = pruning
        self.deterministic = deterministic

    def step(self) -> None:
        move = self._best_move()
        self.env.move(move, self.marker)

    def _best_move(self) -> tuple[int, int]:
        env = self.env.copy()
        score_by_move = {}
        for move in self.env.actions():
            env.move(*move, self.marker)
            result = minimax_alpha_beta(
                env,
                player=self.marker,
                current=env.get_opponent(self.marker),
                max_depth=self.max_depth,
                pruning=self.pruning,
            )
            score_by_move[move] = result.score
            env.clear(*move)
            self.nodes_visited += result.nodes_visited

        if self.deterministic:
            return max(score_by_move, key=lambda move: score_by_move[move])

        max_score = max(score_by_move.values())
        best_moves = [m for m, s in score_by_move.items() if s == max_score]
        return random.choice(best_moves)


class DQNAgent:
    def __init__(self, env: TicTacToe, marker, weights):
        self.env = env
        self.marker = marker
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.net = self._init_net(weights)
        self.nodes_visited = 0

    def step(self) -> None:
        input_vec = self.env.one_hot(self.marker).to(self.device)
        with torch.no_grad():
            q_vals = self.net(input_vec)

        mask = torch.full((9,), float('-inf')).to(self.device)
        actions = self.env.actions()

        for action in actions:
            mask[action[0] * 3 + action[1]] = 0

        q_values = q_vals + mask
        best_index = q_values.argmax().item()
        best_action = next(
            action for action in actions if action[0] * 3 + action[1] == best_index
        )
        self.env.move(best_action, self.marker)

    def _init_net(self, weights):
        net = QNet(27, 9)
        net.load_state_dict(weights)
        net.eval()
        return net.to(self.device)


class QLearningAgent:
    def __init__(self, env, marker, q_table: dict):
        self.env = env
        self.marker = marker
        self.q_vals = q_table
        self.nodes_visited = 0

    def step(self) -> None:
        self.env.move(self._best_move(), self.marker)

    def _best_move(self):
        state = self.env.state_key()
        actions = self.env.actions()
        max_arg = np.argmax([self.q_vals.get((state, action), 0) for action in actions])
        return actions[max_arg]


class RandomAgent:
    def __init__(self, env, marker):
        self.env = env
        self.marker = marker
        self.nodes_visited = 0

    def step(self) -> None:
        self.env.move(random.choice(self.env.actions()), self.marker)


class DefaultAgent:
    def __init__(self, env, marker):
        self.env = env
        self.marker = marker
        self.nodes_visited = 0

    def step(self) -> None:
        best_move = self._best_move()
        self.env.move(best_move, self.marker)

    def _best_move(self) -> tuple[int, int]:
        winning_moves = self.env.winning_moves(self.marker)
        if winning_moves:
            return random.choice(winning_moves)

        opponent = self.env.get_opponent(self.marker)
        losing_moves = self.env.winning_moves(opponent)

        if losing_moves:
            return random.choice(losing_moves)

        return random.choice(self.env.actions())
