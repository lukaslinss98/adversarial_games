from dataclasses import dataclass


@dataclass
class MinimaxResult:
    score: int | float
    nodes_visited: int


def minimax(state, player: str, current: str, depth=0):
    opponent = state.get_opponent(player)
    if state.is_winner(player):
        return 10 - depth - 1

    if state.is_winner(opponent):
        return -10 + depth - 1

    if state.is_draw():
        return 0

    maximize = player == current

    if maximize:
        scores = []
        for move in state.actions():
            state.move(move, player=current)
            score = minimax(state, player, state.get_opponent(current), depth=depth + 1)
            scores.append(score)
            state.clear(move)

        return max(scores)

    else:
        scores = []
        for move in state.actions():
            state.move(move, player=current)
            score = minimax(state, player, state.get_opponent(current), depth=depth + 1)
            scores.append(score)
            state.clear(move)

        return min(scores)


def minimax_alpha_beta(
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
        return MinimaxResult(0, 1)

    nodes_visited = 0
    opponent = state.get_opponent(player)
    if state.is_winner(player):
        return MinimaxResult(10 - depth - 1, 1)

    if state.is_winner(opponent):
        return MinimaxResult(-10 + depth - 1, 1)

    if state.is_draw():
        return MinimaxResult(0, 1)

    maximize = player == current

    if maximize:
        for move in state.actions():
            state.move(move, player=current)
            result = minimax_alpha_beta(
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
            state.clear(move)
            if alpha >= beta and pruning:
                break

        return MinimaxResult(alpha, nodes_visited)

    else:
        for move in state.actions():
            state.move(move, player=current)
            result = minimax_alpha_beta(
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
            state.clear(move)
            if alpha >= beta:
                break

        return MinimaxResult(beta, nodes_visited)
