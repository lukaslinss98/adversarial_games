import argparse

from connectfour.game import connect_four
from connectfour.q_learning_training import train_connectfour
from tiktaktoe.game import tiktaktoe
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

    args = parser.parse_args()

    if args.mode == 'play':
        if args.game == 'tiktaktoe':
            tiktaktoe()
        elif args.game == 'connectfour':
            connect_four()
    elif args.mode == 'train':
        if args.game == 'tiktaktoe':
            train_tiktaktoe(episodes=args.episodes, save=args.save)
        if args.game == 'connectfour':
            train_connectfour(episodes=args.episodes, save=args.save)
