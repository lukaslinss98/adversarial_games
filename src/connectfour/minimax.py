from dataclasses import dataclass


@dataclass
class MinimaxResult:
    score: int | float
    nodes_visited: int


def minimax(
    state,
    player,
    current,
    depth=0,
    max_depth=None,
    alpha=float('-inf'),
    beta=float('inf'),
    pruning=True,
) -> MinimaxResult:
    if max_depth and depth >= max_depth:
        return MinimaxResult(non_terminal_score(state, player), 1)

    nodes_visited = 0
    opponent = state.get_opponent(player)
    if state.check_winner(player):
        return MinimaxResult(10 - depth - 1, 1)

    if state.check_winner(opponent):
        return MinimaxResult(-10 + depth - 1, 1)

    if state.is_draw():
        return MinimaxResult(0, 1)

    maximize = player == current

    if maximize:
        for move in state.possible_moves():
            state.move(*move, player=current)
            result = minimax(
                state,
                player,
                state.get_opponent(current),
                depth + 1,
                max_depth,
                alpha,
                beta,
                pruning,
            )
            nodes_visited += result.nodes_visited + 1
            alpha = max(alpha, result.score)
            state.clear(*move)
            if alpha >= beta and pruning:
                break

        return MinimaxResult(alpha, nodes_visited)

    else:
        for move in state.possible_moves():
            state.move(*move, player=current)
            result = minimax(
                state,
                player,
                state.get_opponent(current),
                depth + 1,
                max_depth,
                alpha,
                beta,
                pruning,
            )
            nodes_visited += result.nodes_visited + 1
            beta = min(beta, result.score)
            state.clear(*move)
            if alpha >= beta:
                break

        return MinimaxResult(beta, nodes_visited)


def non_terminal_score(state, player):
    opponent = state.get_opponent(player)
    score = 0
    windows = state.get_windows()

    for window in windows:
        player_count = window.count(player)
        opponent_count = window.count(opponent)

        if opponent_count == 0:
            if player_count == 3:
                score += 3
            elif player_count == 2:
                score += 1
        if player_count == 0:
            if opponent_count == 3:
                score -= 3
            elif opponent_count == 2:
                score -= 1

    return score
