import pickle
import random

import numpy as np

from connectfour.agent import DefaultAgent
from connectfour.environment import ConnectFour, Token


def get_reward(env, player: Token):
    if env.is_winner(player):
        return 1

    if env.is_winner(env.get_opponent(player)):
        return -1

    if env.is_draw():
        return 0

    return 0


def get_action(state, actions, q_vals, eps) -> int:
    if random.random() > eps:
        argmax = np.argmax([q_vals.get((state, action), 0) for action in actions])
        return actions[argmax]

    return random.choice(actions)


def train_connectfour(episodes: int, save: bool = False):
    EPISODES = episodes
    LR = 0.1
    GAMMA = 0.9
    EPSILON = 1.0
    MIN_EPSILON = 0.3
    EPSILON_DECAY = 0.9999

    env = ConnectFour()
    q_vals = {}

    makers = [Token.BLUE, Token.RED]
    for ep in range(EPISODES + 1):
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        agent = random.choice(makers)
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))

        print(f'\r{ep} / {EPISODES} | {EPSILON=}', end='', flush=True)

        if agent == Token.BLUE and not env.is_game_over():
            opponent.step()

        while not env.is_game_over():
            state = env.state_key()
            actions = env.actions()
            action = get_action(state, actions, q_vals, EPSILON)
            current_q = q_vals.get((state, action), 0)

            env.move(action, agent)

            if not env.is_game_over():
                opponent.step()

            reward = get_reward(env, agent)
            new_state = env.state_key()
            new_actions = env.actions()
            next_max_q = max(
                [q_vals.get((new_state, a), 0) for a in new_actions],
                default=0,
            )

            current_q = current_q + LR * (reward + GAMMA * next_max_q - current_q)
            q_vals[(state, action)] = current_q

        env.reset()

    print()
    print(len(q_vals))

    if save:
        filename = 'q_table_connectfour.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(q_vals, f)
