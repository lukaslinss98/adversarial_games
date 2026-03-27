from enum import Enum

import pygame

from util import (BLUE, BORDER, CONNECT_FOUR_CELL_SIZE, CONNECT_FOUR_COLS,
                  CONNECT_FOUR_ROWS, RED, WHITE)

MAX_CAP = 6
COLUMNS = 7


class Token(Enum):
    RED = 0
    BLUE = 1

    def __str__(self) -> str:
        if self == Token.RED:
            return 'red'
        return 'blue'


class ConnectFour:
    def __init__(self):
        self.state: list[list[Token | None]] = [
            [None] * COLUMNS for _ in range(MAX_CAP)
        ]

    def move(self, col: int, player: Token):
        for row in range(MAX_CAP - 1, -1, -1):
            if self.state[row][col] is None:
                self.state[row][col] = player
                return
        raise Exception(f'Column {col} is full')

    def clear(self, col: int):
        for row in range(MAX_CAP):
            if self.state[row][col] is not None:
                self.state[row][col] = None
                return

    def actions(self):
        return [[col] for col in range(COLUMNS) if self.state[0][col] is None]

    def copy(self):
        new = ConnectFour()
        new.state = [row[:] for row in self.state]
        return new

    def _get_cell(self, row: int, col: int) -> Token | None:
        if row < 0 or row >= MAX_CAP or col < 0 or col >= COLUMNS:
            return None
        return self.state[row][col]

    def winning_moves(self, player: Token) -> list[list[int]]:
        wins = []
        for move in self.actions():
            col = move[0]
            self.move(col, player)
            if self.check_winner(player):
                wins.append([col])
            self.clear(col)
        return wins

    def is_game_over(self) -> bool:
        return (
            self.check_winner(Token.RED)
            or self.check_winner(Token.BLUE)
            or self.is_draw()
        )

    def is_draw(self) -> bool:
        return len(self.actions()) == 0

    def draw(self, screen, winning_line=None):
        for col in range(CONNECT_FOUR_COLS + 1):
            pygame.draw.line(
                screen,
                WHITE,
                (col * CONNECT_FOUR_CELL_SIZE + BORDER, BORDER),
                (
                    col * CONNECT_FOUR_CELL_SIZE + BORDER,
                    CONNECT_FOUR_ROWS * CONNECT_FOUR_CELL_SIZE + BORDER,
                ),
                3,
            )
        for row in range(CONNECT_FOUR_ROWS + 1):
            pygame.draw.line(
                screen,
                WHITE,
                (BORDER, row * CONNECT_FOUR_CELL_SIZE + BORDER),
                (
                    CONNECT_FOUR_COLS * CONNECT_FOUR_CELL_SIZE + BORDER,
                    row * CONNECT_FOUR_CELL_SIZE + BORDER,
                ),
                3,
            )
        for row in range(MAX_CAP):
            for col in range(COLUMNS):
                token = self.state[row][col]
                if token:
                    center_x = (
                        col * CONNECT_FOUR_CELL_SIZE
                        + BORDER
                        + CONNECT_FOUR_CELL_SIZE // 2
                    )
                    center_y = (
                        row * CONNECT_FOUR_CELL_SIZE
                        + BORDER
                        + CONNECT_FOUR_CELL_SIZE // 2
                    )
                    radius = CONNECT_FOUR_CELL_SIZE // 2 - 5
                    color = RED if token == Token.RED else BLUE
                    pygame.draw.circle(screen, color, (center_x, center_y), radius)

    def get_windows(self):
        windows = []

        for r in range(MAX_CAP):
            for c in range(COLUMNS - 3):
                windows.append([self._get_cell(r, c + i) for i in range(4)])

        for r in range(MAX_CAP - 3):
            for c in range(COLUMNS):
                windows.append([self._get_cell(r + i, c) for i in range(4)])

        for r in range(MAX_CAP - 3):
            for c in range(COLUMNS - 3):
                windows.append([self._get_cell(r + i, c + i) for i in range(4)])

        for r in range(MAX_CAP - 3):
            for c in range(3, COLUMNS):
                windows.append([self._get_cell(r + i, c - i) for i in range(4)])

        return windows

    def check_winner(self, token: Token) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(CONNECT_FOUR_ROWS):
            for col in range(CONNECT_FOUR_COLS):
                if self._get_cell(row, col) == token:
                    for dr, dc in directions:
                        if all(
                            self._get_cell(row + dr * i, col + dc * i) == token
                            for i in range(1, 4)
                        ):
                            return True
        return False

    def get_opponent(self, player: Token):
        if player == Token.RED:
            return Token.BLUE
        return Token.RED

    def __str__(self) -> str:
        return ''.join(f'{row}\n' for row in self.state)
