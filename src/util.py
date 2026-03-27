import pygame

_font = None


def get_font():
    global _font
    if _font is None:
        _font = pygame.font.SysFont('courier', 50)
    return _font


BORDER = 50
WINDOW_SIZE = 600
CELL_SIZE = WINDOW_SIZE // 3

BLACK = (0, 0, 0)
WHITE = (250, 250, 250)
GREEN = (0, 240, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

CONNECT_FOUR_COLS = 7
CONNECT_FOUR_ROWS = 6
CONNECT_FOUR_CELL_SIZE = WINDOW_SIZE // CONNECT_FOUR_COLS

PANEL_WIDTH = 250
