from environments.tiktaktoe import TikTakToe


def minimax(state: TikTakToe, player: str, oponent: str, maximize=True, depth=0):
    if state.check_winner(player):
        return 10 - depth - 1

    if state.check_winner(oponent):
        return -10 + depth - 1

    if state.is_draw():
        return 0

    if maximize:
        scores = []
        for move in state.possible_moves():
            state.move(*move, marker=player)
            score = minimax(state, player, oponent, maximize=False, depth=depth + 1)
            scores.append(score)
            state.clear(*move)

        return max(scores)

    else:
        scores = []
        for move in state.possible_moves():
            state.move(*move, marker=oponent)
            score = minimax(state, player, oponent, maximize=True, depth=depth + 1)
            scores.append(score)
            state.clear(*move)

        return min(scores)
