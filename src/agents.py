import random
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from environment import Environment


class Agent(ABC):
    def __init__(self, env: Environment, marker):
        self.env = env
        self.marker = marker
        self.nodes_visited = 0

    @abstractmethod
    def step(self): ...


class RandomAgent(Agent):
    def step(self) -> None:
        move = random.choice(self.env.actions())
        self.env.move(move, self.marker)


class DefaultAgent(Agent):
    def step(self) -> None:
        winning_moves = self.env.winning_moves(self.marker)
        if winning_moves:
            move = random.choice(winning_moves)
            self.env.move(move, self.marker)
            return

        opponent = self.env.get_opponent(self.marker)
        losing_moves = self.env.winning_moves(opponent)

        if losing_moves:
            move = random.choice(losing_moves)
            self.env.move(move, self.marker)
            return

        move = random.choice(self.env.actions())
        self.env.move(move, self.marker)


class QLearningAgent(Agent):
    def __init__(self, env: Environment, marker, q_table: dict):
        super().__init__(env, marker)
        self.q_vals = q_table

    def step(self) -> None:
        state = self.env.state_key()
        actions = self.env.actions()
        max_arg = np.argmax([self.q_vals.get((state, action), 0) for action in actions])
        move = actions[max_arg]

        self.env.move(move, self.marker)


class MinimaxAgent(Agent):
    def __init__(
        self,
        env: Environment,
        marker,
        minimax_fn: Callable,
        max_depth: int | None = None,
        pruning: bool = True,
        deterministic: bool = False,
    ):
        super().__init__(env, marker)
        self.minimax_fn = minimax_fn
        self.max_depth = max_depth
        self.pruning = pruning
        self.deterministic = deterministic

    def step(self) -> None:
        env = self.env.copy()
        score_by_move = {}
        for move in self.env.actions():
            env.move(move, self.marker)
            result = self.minimax_fn(
                env,
                player=self.marker,
                current=env.get_opponent(self.marker),
                max_depth=self.max_depth,
                pruning=self.pruning,
            )
            score_by_move[move] = result.score
            env.clear(move)
            self.nodes_visited += result.nodes_visited

        if self.deterministic:
            move = max(score_by_move, key=lambda m: score_by_move[m])

        max_score = max(score_by_move.values())
        best_moves = [m for m, s in score_by_move.items() if s == max_score]
        move = random.choice(best_moves)

        self.env.move(move, self.marker)


class DQNAgent(Agent):
    def __init__(
        self,
        env: Environment,
        marker,
        weights,
        net: type[nn.Module],
        input_dims: int,
        output_dims: int,
        action_to_index: Callable | None = None,
    ):
        super().__init__(env, marker)
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.action_to_index = action_to_index or (lambda a: a)
        self.output_dims = output_dims
        self.net = self._init_net(net, input_dims, output_dims, weights)

    def step(self) -> None:
        input_vec = self.env.one_hot(self.marker).to(self.device)
        with torch.no_grad():
            q_vals = self.net(input_vec)

        mask = torch.full((self.output_dims,), float('-inf')).to(self.device)
        actions = self.env.actions()

        for action in actions:
            mask[self.action_to_index(action)] = 0

        q_values = q_vals + mask
        best_index = q_values.argmax().item()
        best_action = next(a for a in actions if self.action_to_index(a) == best_index)
        self.env.move(best_action, self.marker)

    def _init_net(self, net_cls, input_dims, output_dims, weights):
        net = net_cls(input_dims, output_dims)
        net.load_state_dict(weights)
        net.eval()
        return net.to(self.device)
