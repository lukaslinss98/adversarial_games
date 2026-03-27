import pygame

from connectfour.agent import Agent as ConnectFourAgent
from connectfour.environment import ConnectFour, Token
from util import BLACK, BORDER, GREEN, PANEL_WIDTH, WHITE, WINDOW_SIZE


def connect_four():
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2 + PANEL_WIDTH, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Connect Four')

    game = ConnectFour()
    RED = Token.RED
    BLUE = Token.BLUE
    agents = {
        BLUE: ConnectFourAgent(game, marker=BLUE, max_depth=10),
        RED: ConnectFourAgent(game, marker=RED, max_depth=10),
    }
    turn = BLUE
    winner: Token | None = None
    draw = False
    running = True
    move_delay = 100
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
            f'Winner: {winner}' if winner else ('Draw!' if draw else f'Turn:  {turn}')
        )
        panel_lines = [
            'STATS',
            '',
            status,
            '',
            f'red:   {agents[RED].nodes_visited:,}',
            f'blue:  {agents[BLUE].nodes_visited:,}',
            f'Total: {total:,}',
        ]
        for i, line in enumerate(panel_lines):
            color = GREEN if i == 0 else WHITE
            surf = panel_font.render(line, True, color)
            screen.blit(surf, (panel_x + 15, BORDER + i * 28))

        if not game.is_game_over() and game.possible_moves():
            agents[turn].step()

            if game.check_winner(turn):
                winner = turn
            elif game.is_draw():
                draw = True
            else:
                turn = BLUE if turn == RED else RED

        if (winner or draw) and not game_over_printed:
            print(f'States explored: {sum(a.nodes_visited for a in agents.values()):,}')
            game_over_printed = True

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()
