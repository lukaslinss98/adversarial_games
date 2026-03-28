from pathlib import Path

import pygame
import torch

_WEIGHTS_DIR = Path(__file__).parent.parent.parent / 'weights'

from tiktaktoe.agent import DefaultAgent, DQNAgent, MinimaxAgent, RandomAgent
from tiktaktoe.environment import TikTakToe
from util import BLACK, BORDER, GREEN, PANEL_WIDTH, WHITE, WINDOW_SIZE


def tiktaktoe():
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2 + PANEL_WIDTH, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Tic Tac Toe - AI vs AI')
    weights = torch.load(_WEIGHTS_DIR / 'tiktaktoe_dqn.pth', weights_only=True)

    game = TikTakToe()
    agents = {
        'X': DQNAgent(game, marker='X', weights=weights),
        'O': RandomAgent(game, marker='O'),
    }
    winner: str | None = None
    winning_line: list[tuple[int, int]] | None = None
    draw: bool = False
    running: bool = True
    move_delay = 1000
    game_over_printed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        game.draw(screen, winning_line)

        panel_x = WINDOW_SIZE + BORDER * 2
        pygame.draw.line(
            screen, WHITE, (panel_x, 0), (panel_x, WINDOW_SIZE + BORDER * 2), 1
        )
        panel_font = pygame.font.SysFont('courier', 18)
        total = sum(a.nodes_visited for a in agents.values())
        status = (
            f'Winner: {winner}'
            if winner
            else ('Draw!' if draw else f'Turn:  {game.current_player}')
        )
        panel_lines = [
            'STATS',
            '',
            status,
            '',
            f'X:     {agents["X"].nodes_visited:,}',
            f'O:     {agents["O"].nodes_visited:,}',
            f'Total: {total:,}',
        ]
        for i, line in enumerate(panel_lines):
            color = GREEN if i == 0 else WHITE
            surf = panel_font.render(line, True, color)
            screen.blit(surf, (panel_x + 15, BORDER + i * 28))

        if not game.is_game_over() and game.actions():
            current = game.current_player
            agents[current].step()

            if game.is_winner(current):
                winner = current
                winning_line = game.get_winning_line(current)
            elif game.is_draw():
                draw = True

        if (winner or draw) and not game_over_printed:
            print(f'States explored: {sum(a.nodes_visited for a in agents.values()):,}')
            game_over_printed = True

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()
