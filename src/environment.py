from abc import ABC, abstractmethod
from typing import Any

import torch


class Environment(ABC):
    current_player: Any

    @abstractmethod
    def move(self, action, player) -> None: ...

    @abstractmethod
    def clear(self, action) -> None: ...

    @abstractmethod
    def actions(self) -> list: ...

    @abstractmethod
    def is_winner(self, player) -> bool: ...

    @abstractmethod
    def is_draw(self) -> bool: ...

    @abstractmethod
    def is_game_over(self) -> bool: ...

    @abstractmethod
    def winning_moves(self, player) -> list: ...

    @abstractmethod
    def get_opponent(self, player) -> Any: ...

    @abstractmethod
    def state_key(self) -> tuple: ...

    @abstractmethod
    def one_hot(self, player) -> torch.Tensor: ...

    @abstractmethod
    def copy(self) -> 'Environment': ...

    @abstractmethod
    def reset(self) -> None: ...
