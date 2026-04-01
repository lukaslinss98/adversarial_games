import argparse
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectfour.environment import ConnectFour, Token
from minimax import minimax

DURATION = 30 * 60

_live_nodes: list[int] = [0]


class TimeUp(Exception):
    pass


def _handler(signum, frame):
    raise TimeUp()


def _best_move(env, marker, pruning: bool):
    opponent = env.get_opponent(marker)
    score_by_move = {}
    total_nodes = 0
    alpha = float('-inf')
    beta = float('inf')
    for move in env.actions():
        env.move(move, marker)
        result = minimax(
            env,
            player=marker,
            current=opponent,
            max_depth=None,
            pruning=pruning,
            counter=_live_nodes,
            alpha=alpha,
            beta=beta,
        )
        score_by_move[move] = result.score
        total_nodes += result.nodes_visited
        env.clear(move)
    best = max(score_by_move, key=lambda m: score_by_move[m])
    return best, total_nodes


def _ticker(start: float, stop: threading.Event) -> None:
    while not stop.wait(timeout=1):
        elapsed = time.perf_counter() - start
        remaining = max(0, DURATION - elapsed)
        nodes = _live_nodes[0]
        print(
            f'\r  [elapsed {elapsed:6.0f}s  remaining {remaining:6.0f}s  nodes so far {nodes:>15,}]',
            end='',
            flush=True,
        )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pruning',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable alpha-beta pruning (default: on)',
    )
    args = parser.parse_args()
    pruning = args.pruning

    env = ConnectFour()

    total_nodes = 0
    moves_completed = 0
    games_completed = 0
    move_log: list[tuple[int, int]] = []

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(DURATION)
    start = time.perf_counter()

    pruning_label = 'ON' if pruning else 'OFF'
    print(f'Running unlimited minimax on Connect Four for {DURATION // 60} minutes …')
    print(f'(alpha-beta pruning {pruning_label}, max_depth=None)\n')

    try:
        while True:
            env.reset()
            current = Token.RED
            move_in_game = 0

            while not env.is_game_over():
                stop_ticker = threading.Event()
                ticker = threading.Thread(
                    target=_ticker, args=(start, stop_ticker), daemon=True
                )
                ticker.start()

                t0 = time.perf_counter()
                move, nodes = _best_move(env, current, pruning)
                elapsed_move = time.perf_counter() - t0

                stop_ticker.set()
                ticker.join()
                print('\r' + ' ' * 80 + '\r', end='')

                env.move(move, current)
                total_nodes += nodes
                moves_completed += 1
                move_in_game += 1
                move_log.append((move_in_game, nodes))

                elapsed_total = time.perf_counter() - start
                print(
                    f'  game {games_completed + 1}  move {move_in_game:>2}  '
                    f'col={move}  nodes={nodes:>12,}  '
                    f'move_time={elapsed_move:.2f}s  '
                    f'total_elapsed={elapsed_total:.1f}s'
                )

                current = env.get_opponent(current)

            games_completed += 1

    except TimeUp:
        pass

    elapsed = time.perf_counter() - start
    nodes_at_interrupt = _live_nodes[0]
    print(f'\n{"=" * 60}')
    print(f'Time elapsed      : {elapsed:.1f}s  ({elapsed / 60:.2f} min)')
    print(f'Games completed   : {games_completed}')
    print(f'Moves completed   : {moves_completed}')
    print(f'Total nodes       : {total_nodes:,}')
    if moves_completed == 0:
        print(f'Nodes visited (partial, first move): {nodes_at_interrupt:,}')
        print('(search did not complete a single move — tree is too large)')
    else:
        print(f'Nodes / move (avg): {total_nodes / moves_completed:,.0f}')
        nodes_list = [n for _, n in move_log]
        print(f'Nodes / move (max): {max(nodes_list):,}')
        print(f'Nodes / move (min): {min(nodes_list):,}')


if __name__ == '__main__':
    run()
