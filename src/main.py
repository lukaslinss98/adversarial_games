import pygame

from agent import Agent, RandomAgent
from environments.tiktaktoe import TikTakToe
from util import BLACK, BORDER, GREEN, WINDOW_SIZE


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_SIZE + BORDER * 2, WINDOW_SIZE + BORDER * 2)
    )
    pygame.display.set_caption('Tic Tac Toe - AI vs AI')

    game = TikTakToe()
    agents = {
        'X': Agent(game, marker='X'),
        'O': Agent(game, marker='O'),
    }
    turn: str = 'X'
    winner: str | None = None
    draw: bool = False
    running: bool = True
    move_delay = 100

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        game.draw(screen)

        if not game.is_game_over() and game.possible_moves():
            agents[turn].step()
            print(game.__str__())

            if game.check_winner(turn):
                winner = turn
            elif game.is_draw():
                draw = True
            else:
                turn = 'O' if turn == 'X' else 'X'

        if winner:
            message = f'Winner: {winner}'
        elif draw:
            message = 'Draw!'
        else:
            message = None

        if message:
            result_font = pygame.font.SysFont('courier', 20)
            text = result_font.render(message, True, GREEN)
            text_rect = text.get_rect(center=(BORDER + 10, BORDER))
            screen.blit(text, text_rect)

        pygame.display.flip()
        pygame.time.wait(move_delay)

    pygame.quit()


if __name__ == '__main__':
    main()
