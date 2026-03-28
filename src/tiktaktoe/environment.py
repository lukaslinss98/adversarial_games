import numpy as np
import torch

from util import BORDER, CELL_SIZE, GREEN, WHITE, WINDOW_SIZE, get_font

WIN_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


class TikTakToe:
    def __init__(self):
        self.state: list[list[str | None]] = [[None, None, None] for _ in range(3)]
        self.current_player: str = 'X'

    def move(self, x, y, player):
        self.state[x][y] = player
        self.current_player = self.get_opponent(player)

    def clear(self, x: int, y: int):
        self.state[x][y] = None
        self.current_player = self.get_opponent(self.current_player)

    def actions(self) -> list[tuple[int, int]]:
        moves = []
        for x, row in enumerate(self.state):
            for y, cell in enumerate(row):
                if not cell:
                    moves.append((x, y))
        return moves

    def is_winner(self, player: str):
        return any(
            all(self.state[x][y] == player for x, y in line) for line in WIN_LINES
        )

    def is_draw(self) -> bool:
        return not self.actions() and not any(
            self.is_winner(marker) for marker in ('X', 'O')
        )

    def get_opponent(self, player):
        if player == 'X':
            return 'O'
        return 'X'

    def is_game_over(self) -> bool:
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def get_winning_line(self, player: str) -> list[tuple[int, int]] | None:
        for line in WIN_LINES:
            if all(self.state[x][y] == player for x, y in line):
                return line
        return None

    def winning_moves(self, player):
        moves = []
        for move in self.actions():
            self.move(*move, player)
            if self.is_winner(player):
                moves.append(move)
            self.clear(*move)
        return moves

    def draw(self, screen, winning_line: list[tuple[int, int]] | None = None) -> None:
        import pygame

        for i in range(1, 3):
            pygame.draw.line(
                screen,
                WHITE,
                (i * CELL_SIZE + BORDER, 0 + BORDER),
                (i * CELL_SIZE + BORDER, WINDOW_SIZE + BORDER),
                3,
            )
            pygame.draw.line(
                screen,
                WHITE,
                (BORDER, i * CELL_SIZE + BORDER),
                (WINDOW_SIZE + BORDER, i * CELL_SIZE + BORDER),
                3,
            )

        for x in range(3):
            for y in range(3):
                marker = self.state[x][y]
                if marker:
                    color = GREEN if winning_line and (x, y) in winning_line else WHITE
                    text = get_font().render(marker, True, color)
                    text_rect = text.get_rect(
                        center=(
                            y * CELL_SIZE + BORDER + CELL_SIZE // 2,
                            x * CELL_SIZE + BORDER + CELL_SIZE // 2,
                        )
                    )
                    screen.blit(text, text_rect)

    def reset(self):
        self.state = [[None, None, None] for _ in range(3)]
        self.current_player = 'X'

    def copy(self):
        new_game = TikTakToe()
        new_game.state = [row[:] for row in self.state]
        new_game.current_player = self.current_player
        return new_game

    def state_key(self):
        return tuple(cell for row in self.state for cell in row) + (
            self.current_player,
        )

    def one_hot(self, player: str):
        opponent = self.get_opponent(player)
        one_hot_vec = []
        for row in self.state:
            for cell in row:
                if cell == player:
                    one_hot_vec.extend([1, 0, 0])
                elif cell == opponent:
                    one_hot_vec.extend([0, 1, 0])
                else:
                    one_hot_vec.extend([0, 0, 1])
        return torch.tensor(one_hot_vec, dtype=torch.float32)

    def __str__(self) -> str:
        return ''.join(f'{row}\n' for row in self.state)
