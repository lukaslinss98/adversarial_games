from environments.tiktaktoe import TikTakToe


def minimax(state: TikTakToe, player: str, current: str, depth=0):
    opponent = state.get_opponent(player)
    if state.check_winner(player):
        return 10 - depth - 1

    if state.check_winner(opponent):
        return -10 + depth - 1

    if state.is_draw():
        return 0

    maximize = player == current

    if maximize:
        scores = []
        for move in state.possible_moves():
            state.move(*move, marker=current)
            score = minimax(state, player, state.get_opponent(current), depth=depth + 1)
            scores.append(score)
            state.clear(*move)

        return max(scores)

    else:
        scores = []
        for move in state.possible_moves():
            state.move(*move, marker=current)
            score = minimax(state, player, state.get_opponent(current), depth=depth + 1)
            scores.append(score)
            state.clear(*move)

        return min(scores)
