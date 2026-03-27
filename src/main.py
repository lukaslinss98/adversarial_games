import argparse

from connectfour.evaluate import evaluate_connectfour
from connectfour.game import connect_four
from connectfour.q_learning_training import train_connectfour
from tiktaktoe.evaluate import evaluate_tiktaktoe
from tiktaktoe.game import tiktaktoe
from tiktaktoe.dqn_training import train_tiktaktoe_dqn
from tiktaktoe.q_learning_training import train_tiktaktoe

VALID_GAMES = ('tiktaktoe', 'connectfour')


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
    eval_parser.add_argument('--runs', type=int, default=100, help='Number of games (default: 100)')
    eval_parser.add_argument('--agent1', type=str, required=True, help='Agent 1 (minimax/ql/default/random)')
    eval_parser.add_argument('--agent2', type=str, required=True, help='Agent 2 (minimax/ql/default/random)')

    args = parser.parse_args()

    if args.mode == 'play':
        if args.game == 'tiktaktoe':
            tiktaktoe()
        elif args.game == 'connectfour':
            connect_four()
    elif args.mode == 'train':
        if args.game == 'tiktaktoe':
            if args.algo == 'dqn':
                train_tiktaktoe_dqn(episodes=args.episodes, save=args.save)
            else:
                train_tiktaktoe(episodes=args.episodes, save=args.save)
        if args.game == 'connectfour':
            train_connectfour(episodes=args.episodes, save=args.save)
    elif args.mode == 'eval':
        if args.game == 'tiktaktoe':
            evaluate_tiktaktoe(args.runs, args.agent1, args.agent2)
        elif args.game == 'connectfour':
            evaluate_connectfour(args.runs, args.agent1, args.agent2)
