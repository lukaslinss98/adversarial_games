import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.nn.modules import loss

from agents import DefaultAgent
from environment import Environment


def eval_agent(
    q_vals: dict,
    env: Environment,
    markers: list,
    games: int = 100,
) -> tuple[float, float, float]:
    wins = draws = opponent_wins = 0
    for _ in range(games):
        env.reset()
        agent = random.choice(markers)
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))

        if agent == markers[1] and not env.is_game_over():
            opponent.step()

        while not env.is_game_over():
            state = env.state_key()
            actions = env.actions()
            argmax = np.argmax([q_vals.get((state, a), 0) for a in actions])
            env.move(actions[argmax], agent)
            if not env.is_game_over():
                opponent.step()

        if env.is_winner(agent):
            wins += 1
        elif env.is_draw():
            draws += 1
        else:
            opponent_wins += 1
    return wins / games, draws / games, opponent_wins / games


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
    min_eps: float = 0.15,
    eval_interval: int = 500,
    eval_games: int = 100,
):
    eps = 1.0
    eps_decay = (min_eps / eps) ** (1 / episodes)
    q_vals = {}
    win_rates: list[tuple[int, float, float, float]] = []

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

        if ep > 0 and ep % eval_interval == 0:
            win_rate, draw_rate, loss_rate = eval_agent(
                q_vals, env, markers, eval_games
            )
            win_rates.append((ep, win_rate, draw_rate, loss_rate))

    print()
    print(f'{len(q_vals)} states')

    if win_rates:
        eps_nums, win_r, draw_r, loss_r = zip(*win_rates)
        plt.plot(eps_nums, win_r, marker='o', markersize=3, label='win')
        plt.plot(eps_nums, draw_r, marker='o', markersize=3, label='draw')
        plt.plot(eps_nums, loss_r, marker='o', markersize=3, label='loss')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.legend()
        plt.title(
            f'Results vs DefaultAgent (every {eval_interval} eps, {eval_games} games)'
        )
        plt.show()

    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(q_vals, f)
