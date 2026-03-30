import pickle
import random
from pathlib import Path

import numpy as np

from agents import DefaultAgent
from environment import Environment


def get_reward(env: Environment, player):
    if env.is_winner(player):
        return 1
    if env.is_winner(env.get_opponent(player)):
        return -1
    return 0


def get_action(state, actions, q_vals, eps):
    if random.random() > eps:
        argmax = np.argmax([q_vals.get((state, action), 0) for action in actions])
        return actions[argmax]
    return random.choice(actions)


def train_ql(
    env: Environment,
    markers: list,
    episodes: int,
    save_path: Path | None = None,
    lr: float = 0.1,
    gamma: float = 0.9,
    min_eps: float = 0.1,
):
    eps = 1.0
    eps_decay = (min_eps / eps) ** (1 / episodes)
    q_vals = {}

    for ep in range(episodes + 1):
        eps = eps * eps_decay
        agent = random.choice(markers)
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))

        print(f'\r{ep} / {episodes} | epsilon={eps:.4f}', end='', flush=True)

        if agent == markers[1] and not env.is_game_over():
            opponent.step()

        while not env.is_game_over():
            state = env.state_key()
            actions = env.actions()

            if random.random() > eps:
                argmax = np.argmax(
                    [q_vals.get((state, action), 0) for action in actions]
                )
                action = actions[argmax]
            else:
                action = random.choice(actions)

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

            current_q = current_q + lr * (reward + gamma * next_max_q - current_q)
            q_vals[(state, action)] = current_q

        env.reset()

    print()
    print(f'{len(q_vals)} states')

    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(q_vals, f)
