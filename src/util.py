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
