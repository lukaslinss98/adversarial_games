import argparse
from itertools import combinations_with_replacement
from pathlib import Path

from connectfour.evaluate import VALID_AGENTS as CF_AGENTS, evaluate_connectfour
from connectfour.game import connect_four
from dqn_training import train_dqn
from q_learning_training import train_ql
from tictactoe.evaluate import VALID_AGENTS as TTT_AGENTS, evaluate_tictactoe
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

    args = parser.parse_args()

    if args.mode == 'play':
        if args.game == 'tictactoe':
            tictactoe()
        elif args.game == 'connectfour':
            connect_four()
    elif args.mode == 'train':
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
                    save_path=_WEIGHTS_DIR / 'tictactoe_dqn.pth' if args.save else None,
                    action_to_index=lambda a: a[0] * 3 + a[1],
                )
            else:
                train_ql(
                    env=TicTacToe(),
                    markers=['X', 'O'],
                    episodes=args.episodes,
                    save_path=_WEIGHTS_DIR / 'tictactoe_ql.pkl' if args.save else None,
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
                    save_path=_WEIGHTS_DIR / 'connectfour_dqn.pth' if args.save else None,
                )
            else:
                train_ql(
                    env=ConnectFour(),
                    markers=[Token.RED, Token.BLUE],
                    episodes=args.episodes,
                    save_path=_WEIGHTS_DIR / 'connectfour_ql.pkl' if args.save else None,
                )
    elif args.mode == 'eval':
        if getattr(args, 'all', False):
            agents = TTT_AGENTS if args.game == 'tictactoe' else CF_AGENTS
            for a1, a2 in combinations_with_replacement(agents, 2):
                print(f'\n{"="*40}')
                print(f'{a1} vs {a2}')
                print(f'{"="*40}')
                if args.game == 'tictactoe':
                    evaluate_tictactoe(args.runs, a1, a2)
                elif args.game == 'connectfour':
                    evaluate_connectfour(args.runs, a1, a2)
        else:
            if not args.agent1 or not args.agent2:
                parser.error('--agent1 and --agent2 are required unless --all is set')
            if args.game == 'tictactoe':
                evaluate_tictactoe(args.runs, args.agent1, args.agent2)
            elif args.game == 'connectfour':
                evaluate_connectfour(args.runs, args.agent1, args.agent2)
