from dataclasses import dataclass

from environment import Environment


@dataclass
class MinimaxResult:
    score: int | float
    nodes_visited: int


def minimax(
    state: Environment,
    player,
    current,
    depth=0,
    max_depth=None,
    alpha=float('-inf'),
    beta=float('inf'),
    pruning=True,
    counter=None,
) -> MinimaxResult:
    if max_depth and depth >= max_depth:
        score = non_terminal_score(state, player)
        return MinimaxResult(score, 1)

    nodes_visited = 0
    opponent = state.get_opponent(player)
    if state.is_winner(player):
        return MinimaxResult(1000 - depth, 1)

    if state.is_winner(opponent):
        return MinimaxResult(-1000 + depth, 1)

    if state.is_draw():
        return MinimaxResult(0, 1)

    maximize = player == current

    for move in state.actions():
        state.move(move, player=current)
        result = minimax(
            state,
            player,
            state.get_opponent(current),
            depth + 1,
            max_depth,
            alpha,
            beta,
            pruning,
            counter,
        )
        nodes_visited += result.nodes_visited + 1
        if counter is not None:
            counter[0] += 1
        state.clear(move)

        if maximize:
            alpha = max(alpha, result.score)
        else:
            beta = min(beta, result.score)

        if pruning and alpha >= beta:
            break

    return MinimaxResult(alpha if maximize else beta, nodes_visited)


def non_terminal_score(state, player):
    opponent = state.get_opponent(player)
    score = 0
    windows = state.get_windows()

    for window in windows:
        player_count = window.count(player)
        opponent_count = window.count(opponent)

        if opponent_count == 0:
            if player_count == 3:
                score += 2
            elif player_count == 2:
                score += 1
        if player_count == 0:
            if opponent_count == 3:
                score -= 2
            elif opponent_count == 2:
                score -= 1

    return score
