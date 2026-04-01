from tictactoe.environment import TicTacToe
from tictactoe.evaluate import create_agent, _load_dqn_weights, _load_q_table
from util import BLACK, BORDER, GREEN, PANEL_WIDTH, WHITE, WINDOW_SIZE


def tictactoe(agent1_type: str = 'dqn', agent2_type: str = 'minimax', move_delay: int = 1000, minimax_depth=None, pruning=True):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2 + PANEL_WIDTH, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Tic Tac Toe - AI vs AI')

    game = TicTacToe()
    q_table = _load_q_table(agent1_type, agent2_type)
    dqn_weights = _load_dqn_weights(agent1_type, agent2_type)
    agents = {
        'X': create_agent(agent1_type, game, 'X', q_table, dqn_weights, minimax_depth, pruning),
        'O': create_agent(agent2_type, game, 'O', q_table, dqn_weights, minimax_depth, pruning),
    }
    winner: str | None = None
    winning_line: list[tuple[int, int]] | None = None
    draw: bool = False
    running: bool = True
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
        x_nodes = sum(agents['X'].nodes_visited)
        o_nodes = sum(agents['O'].nodes_visited)
        total = x_nodes + o_nodes
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
            f'X ({agent1_type}): {x_nodes:,}',
            f'O ({agent2_type}): {o_nodes:,}',
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
            print(f'States explored: {sum(sum(a.nodes_visited) for a in agents.values()):,}')
            game_over_printed = True

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()
