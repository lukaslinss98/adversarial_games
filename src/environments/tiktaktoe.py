import pygame

from util import BORDER, CELL_SIZE, WHITE, WINDOW_SIZE, get_font

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
    def __init__(self) -> None:
        self.grid: list[list[str | None]] = [
            [None] * 3,
            [None] * 3,
            [None] * 3,
        ]

    def move(self, x: int, y: int, marker: str) -> None:
        self.grid[x][y] = marker

    def clear(self, x: int, y: int) -> None:
        self.grid[x][y] = None

    def possible_moves(self) -> list[tuple[int, int]]:
        moves = []
        for x, row in enumerate(self.grid):
            for y, cell in enumerate(row):
                if not cell:
                    moves.append((x, y))
        return moves

    def check_winner(self, marker: str) -> bool:
        return any(
            all(self.grid[x][y] == marker for x, y in line) for line in WIN_LINES
        )

    def is_draw(self) -> bool:
        return not self.possible_moves() and not any(
            self.check_winner(marker) for marker in ('X', 'O')
        )

    def get_opponent(self, player):
        if player == 'X':
            return 'O'
        return 'X'

    def is_game_over(self) -> bool:
        return self.check_winner('X') or self.check_winner('O') or self.is_draw()

    def draw(self, screen) -> None:
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
            for y in range(len(self.grid)):
                marker = self.grid[x][y]
                if marker:
                    text = get_font().render(marker, True, WHITE)
                    text_rect = text.get_rect(
                        center=(
                            y * CELL_SIZE + BORDER + CELL_SIZE // 2,
                            x * CELL_SIZE + BORDER + CELL_SIZE // 2,
                        )
                    )
                    screen.blit(text, text_rect)

    def copy(self) -> 'TikTakToe':
        new_game = TikTakToe()
        new_game.grid = [row[:] for row in self.grid]
        return new_game

    def __str__(self) -> str:
        return ''.join(f'{row}\n' for row in self.grid)
