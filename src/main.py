import argparse
from itertools import combinations_with_replacement
from pathlib import Path

from connectfour.evaluate import VALID_AGENTS as CF_AGENTS
from connectfour.evaluate import evaluate_connectfour
from connectfour.game import connect_four
from dqn_training import train_dqn
from q_learning_training import train_ql
from tictactoe.evaluate import VALID_AGENTS as TTT_AGENTS
from tictactoe.evaluate import evaluate_tictactoe
from tictactoe.game import tictactoe

VALID_GAMES = ('tictactoe', 'connectfour')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial games')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    play_parser = subparsers.add_parser('play', help='Play a game')
    play_parser.add_argument(
        '--game',
        type=str,
        required=True,
        choices=VALID_GAMES,
        help=f'Game to play. Valid options: {", ".join(VALID_GAMES)}',
    )
    play_parser.add_argument(
        '--agent1', type=str, default='dqn', choices=TTT_AGENTS,
        help='Agent 1 type (default: dqn)',
    )
    play_parser.add_argument(
        '--agent2', type=str, default='minimax', choices=TTT_AGENTS,
        help='Agent 2 type (default: minimax)',
    )
    play_parser.add_argument(
        '--move-delay', type=int, default=None,
        help='Milliseconds between moves (default: 1000 tictactoe, 300 connectfour)',
    )
    play_parser.add_argument(
        '--minimax-depth', type=int, default=None,
        help='Minimax search depth (default: unlimited tictactoe, 5 connectfour)',
    )
    play_parser.add_argument(
        '--pruning',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable alpha-beta pruning (default: on)',
    )

    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument(
        '--game',
        type=str,
        required=True,
        choices=VALID_GAMES,
        help=f'Game to train. Valid options: {", ".join(VALID_GAMES)}',
    )
    train_parser.add_argument(
        '--episodes',
        type=int,
        default=10000,
        help='Number of training episodes (default: 10000)',
    )
    train_parser.add_argument(
        '--save',
        action='store_true',
        help='Save the q-table after training',
    )
    train_parser.add_argument(
        '--algo',
        type=str,
        default='ql',
        choices=('ql', 'dqn'),
        help='Training algorithm: ql or dqn (default: ql)',
    )
    train_parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (default: 0.1 ql, 0.00001 dqn)',
    )
    train_parser.add_argument(
        '--gamma', type=float, default=None,
        help='Discount factor (default: 0.9)',
    )
    train_parser.add_argument(
        '--min-eps', type=float, default=None,
        help='Minimum epsilon for exploration (default: 0.15 ql, 0.01 dqn)',
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size, DQN only (default: 128)',
    )
    train_parser.add_argument(
        '--buffer-cap', type=int, default=None,
        help='Replay buffer capacity, DQN only (default: 10000)',
    )
    train_parser.add_argument(
        '--tau', type=float, default=None,
        help='Soft update rate, DQN only (default: 0.005)',
    )
    train_parser.add_argument(
        '--eval-interval', type=int, default=None,
        help='Episodes between evaluations (default: 500 ql, 100 dqn)',
    )
    train_parser.add_argument(
        '--eval-games', type=int, default=None,
        help='Games per evaluation (default: 100)',
    )

    eval_parser = subparsers.add_parser('eval', help='Evaluate two agents headlessly')
    eval_parser.add_argument(
        '--game',
        type=str,
        required=True,
        choices=VALID_GAMES,
        help=f'Game to evaluate. Valid options: {", ".join(VALID_GAMES)}',
    )
    eval_parser.add_argument(
        '--runs', type=int, default=100, help='Number of games (default: 100)'
    )
    eval_parser.add_argument(
        '--agent1', type=str, help='Agent 1 (minimax/ql/default/random)'
    )
    eval_parser.add_argument(
        '--agent2', type=str, help='Agent 2 (minimax/ql/default/random)'
    )
    eval_parser.add_argument(
        '--all', action='store_true', help='Test all agent combinations'
    )
    eval_parser.add_argument(
        '--minimax-depth', type=int, default=None,
        help='Minimax search depth (default: unlimited tictactoe, 5 connectfour)',
    )
    eval_parser.add_argument(
        '--pruning',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable alpha-beta pruning (default: on)',
    )

    args = parser.parse_args()

    if args.mode == 'play':
        play_kwargs = {'pruning': args.pruning}
        if args.minimax_depth is not None:
            play_kwargs['minimax_depth'] = args.minimax_depth
        if args.move_delay is not None:
            play_kwargs['move_delay'] = args.move_delay
        if args.game == 'tictactoe':
            tictactoe(args.agent1, args.agent2, **play_kwargs)
        elif args.game == 'connectfour':
            connect_four(args.agent1, args.agent2, **play_kwargs)
    elif args.mode == 'train':
        train_kwargs = {k: v for k, v in {
            'lr': args.lr,
            'gamma': args.gamma,
            'min_eps': args.min_eps,
            'eval_interval': args.eval_interval,
            'eval_games': args.eval_games,
        }.items() if v is not None}
        dqn_kwargs = {k: v for k, v in {
            'batch_size': args.batch_size,
            'buffer_cap': args.buffer_cap,
            'tau': args.tau,
        }.items() if v is not None}

        _WEIGHTS_DIR = Path(__file__).parent.parent / 'weights'
        if args.game == 'tictactoe':
            from tictactoe.environment import TicTacToe
            from tictactoe.model import QNet as TTTQNet

            if args.algo == 'dqn':
                train_dqn(
                    env=TicTacToe(),
                    markers=['X', 'O'],
                    net_cls=TTTQNet,
                    input_dims=27,
                    output_dims=9,
                    episodes=args.episodes,
                    game_name='Tic Tac Toe',
                    save_path=_WEIGHTS_DIR / 'tictactoe_dqn_v2.pth'
                    if args.save
                    else None,
                    action_to_index=lambda a: a[0] * 3 + a[1],
                    **train_kwargs,
                    **dqn_kwargs,
                )
            else:
                train_ql(
                    env=TicTacToe(),
                    markers=['X', 'O'],
                    episodes=args.episodes,
                    save_path=_WEIGHTS_DIR / 'tictactoe_ql.pkl' if args.save else None,
                    **train_kwargs,
                )
        elif args.game == 'connectfour':
            from connectfour.environment import ConnectFour, Token
            from connectfour.model import QNet as CFQNet

            if args.algo == 'dqn':
                train_dqn(
                    env=ConnectFour(),
                    markers=[Token.RED, Token.BLUE],
                    net_cls=CFQNet,
                    input_dims=6 * 7 * 3,
                    output_dims=7,
                    episodes=args.episodes,
                    game_name='Connect Four',
                    save_path=_WEIGHTS_DIR / 'connectfour_dqn_v2.pth'
                    if args.save
                    else None,
                    **train_kwargs,
                    **dqn_kwargs,
                )
            else:
                train_ql(
                    env=ConnectFour(),
                    markers=[Token.RED, Token.BLUE],
                    episodes=args.episodes,
                    save_path=_WEIGHTS_DIR / 'connectfour_ql.pkl'
                    if args.save
                    else None,
                    **train_kwargs,
                )
    elif args.mode == 'eval':
        eval_kwargs = {'pruning': args.pruning}
        if args.minimax_depth is not None:
            eval_kwargs['minimax_depth'] = args.minimax_depth
        if getattr(args, 'all', False):
            agents = TTT_AGENTS if args.game == 'tictactoe' else CF_AGENTS
            for a1, a2 in combinations_with_replacement(agents, 2):
                print(f'\n{"=" * 40}')
                print(f'{a1} vs {a2}')
                print(f'{"=" * 40}')
                if args.game == 'tictactoe':
                    evaluate_tictactoe(args.runs, a1, a2, **eval_kwargs)
                elif args.game == 'connectfour':
                    evaluate_connectfour(args.runs, a1, a2, **eval_kwargs)
        else:
            if not args.agent1 or not args.agent2:
                parser.error('--agent1 and --agent2 are required unless --all is set')
            if args.game == 'tictactoe':
                evaluate_tictactoe(args.runs, args.agent1, args.agent2, **eval_kwargs)
            elif args.game == 'connectfour':
                evaluate_connectfour(args.runs, args.agent1, args.agent2, **eval_kwargs)
