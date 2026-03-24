import random
from typing import Callable

from algorithms.minimax import minimax
from environments.tiktaktoe import TikTakToe


class Agent:
    def __init__(self, env: TikTakToe, marker: str) -> None:
        self.env = env
        self.marker = marker

    def step(self) -> None:
        move = self._best_move()
        self.env.move(*move, self.marker)

    def _best_move(self) -> tuple[int, int]:
        env = self.env.copy()
        score_by_move = {}
        for move in self.env.possible_moves():
            env.move(*move, self.marker)
            score = minimax(
                env, player=self.marker, current=env.get_opponent(self.marker)
            )
            score_by_move[move] = score
            env.clear(*move)

        print(score_by_move)

        return max(score_by_move, key=score_by_move.get)


class RandomAgent:
    def __init__(self, env: TikTakToe, marker) -> None:
        self.env = env
        self.marker = marker

    def step(self) -> None:
        move = self._best_move()
        self.env.move(*move, self.marker)

    def _best_move(self) -> tuple[int, int]:
        return random.choice(self.env.possible_moves())
