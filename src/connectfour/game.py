from connectfour.environment import ConnectFour, Token
from connectfour.evaluate import _load_dqn_weights, _load_q_table, _make_agent
from util import BLACK, BORDER, GREEN, PANEL_WIDTH, WHITE, WINDOW_SIZE


def connect_four(agent1_type: str = 'dqn', agent2_type: str = 'minimax', move_delay: int = 300, minimax_depth=5, pruning=True):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2 + PANEL_WIDTH, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Connect Four')

    game = ConnectFour()
    q_table = _load_q_table(agent1_type, agent2_type)
    dqn_weights = _load_dqn_weights(agent1_type, agent2_type)
    agents = {
        Token.RED: _make_agent(agent1_type, game, Token.RED, q_table, dqn_weights, minimax_depth, pruning),
        Token.BLUE: _make_agent(agent2_type, game, Token.BLUE, q_table, dqn_weights, minimax_depth, pruning),
    }
    winner: Token | None = None
    draw = False
    running = True
    game_over_printed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        game.draw(screen)

        panel_x = WINDOW_SIZE + BORDER * 2
        pygame.draw.line(
            screen, WHITE, (panel_x, 0), (panel_x, WINDOW_SIZE + BORDER * 2), 1
        )
        panel_font = pygame.font.SysFont('courier', 18)
        red_nodes = sum(agents[Token.RED].nodes_visited)
        blue_nodes = sum(agents[Token.BLUE].nodes_visited)
        total = red_nodes + blue_nodes
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
            f'red ({agent1_type}): {red_nodes:,}',
            f'blue ({agent2_type}): {blue_nodes:,}',
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
            print(f'States explored: {sum(sum(a.nodes_visited) for a in agents.values()):,}')
            game_over_printed = True

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()
