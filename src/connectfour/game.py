from pathlib import Path

import torch

from tiktaktoe.agent import MinimaxAgent

_WEIGHTS_DIR = Path(__file__).parent.parent.parent / 'weights'

from connectfour.agent import DefaultAgent, DQNAgent, RandomAgent
from connectfour.environment import ConnectFour, Token
from util import BLACK, BORDER, GREEN, PANEL_WIDTH, WHITE, WINDOW_SIZE


def connect_four():
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2 + PANEL_WIDTH, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Connect Four')

    weights = torch.load(_WEIGHTS_DIR / 'connectfour_dqn.pth', weights_only=True)

    game = ConnectFour()
    agents = {
        Token.RED: DQNAgent(game, marker=Token.RED, weights=weights),
        Token.BLUE: MinimaxAgent(game, marker=Token.BLUE, max_depth=5),
    }
    winner: Token | None = None
    draw = False
    running = True
    move_delay = 300
    game_over_printed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        game.draw(screen, None)

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
            f'red:   {agents[Token.RED].nodes_visited:,}',
            f'blue:  {agents[Token.BLUE].nodes_visited:,}',
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
            elif game.is_draw():
                draw = True

        if (winner or draw) and not game_over_printed:
            print(f'States explored: {sum(a.nodes_visited for a in agents.values()):,}')
            game_over_printed = True

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()
