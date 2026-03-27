import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn

from tiktaktoe.agent import DefaultAgent
from tiktaktoe.environment import TikTakToe


class QNet(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.linear_1 = nn.Linear(input_dims, 128)
        self.activation_1 = nn.ReLU()
        self.linear_2 = nn.Linear(128, 64)
        self.activation_2 = nn.ReLU()
        self.out = nn.Linear(64, output_dims)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        return self.out(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Tensor,
        action: int,
        reward: int,
        new_state: Tensor,
        is_over: bool,
    ):
        self.buffer.append((state, action, reward, new_state, is_over))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen


def get_reward(env: TikTakToe, player: str):
    if env.is_winner(player):
        return 1

    if env.is_winner(env.get_opponent(player)):
        return -1

    if env.is_draw():
        return 0

    return 0


def get_action(state, actions, q_vals, eps):
    if random.random() > eps:
        argmax = np.argmax([q_vals.get((state, action), 0) for action in actions])
        return actions[argmax]

    return random.choice(actions)


action_to_index = lambda action: sum(action)


def train_tiktaktoe_dqn(episodes: int, save: bool = False):
    EPISODES = episodes
    MOVES = 9
    GAMMA = 0.9
    INPUT_DIMS = 29
    BUFFER_CAP = 256
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    EPSILON = 1.0
    MIN_EPSILON = 0.1
    EPSILON_DECAY = 0.9995
    TARGET_UPDATE = 20

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    makers = ['X', 'O']
    q_net = QNet(INPUT_DIMS, MOVES).to(device)
    target_net = QNet(INPUT_DIMS, MOVES).to(device)
    buffer = ReplayBuffer(capacity=BUFFER_CAP)
    losses = []

    optimizer = optim.Adam(q_net.parameters(), LEARNING_RATE)

    target_net.load_state_dict(q_net.state_dict())

    env = TikTakToe()

    for ep in range(EPISODES + 1):
        env.reset()

        agent = random.choice(makers)
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))

        if agent == 'O' and not env.is_game_over():
            opponent.step()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        while not env.is_game_over():
            print(f'\r{ep} / {EPISODES} | {EPSILON=}', end='', flush=True)

            state = env.state_one_hot()
            state = torch.tensor(state, dtype=torch.float32).to(device)

            q_values: Tensor = q_net(state)

            mask = torch.full((9,), float('-inf')).to(device)
            actions = env.actions()

            for action in actions:
                mask[action_to_index(action)] = 0

            q_values = q_values + mask

            if random.random() > EPSILON:
                best_index = q_values.argmax().item()
                best_action = next(
                    action
                    for action in actions
                    if action_to_index(action) == best_index
                )
            else:
                best_action = random.choice(actions)

            env.move(*best_action, agent)
            new_state = torch.tensor(env.state_one_hot(), dtype=torch.float32).to(
                device
            )

            buffer.push(
                state,
                action_to_index(best_action),
                get_reward(env, agent),
                new_state,
                env.is_game_over(),
            )

            if buffer.is_full():
                optimizer.zero_grad()

                batch = buffer.sample(BATCH_SIZE)
                state, action, reward, new_state, is_done = zip(*batch)

                state = torch.stack(state)
                new_state = torch.stack(new_state)
                action = torch.tensor(action, dtype=torch.long).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                is_done = torch.tensor(is_done, dtype=torch.float32).to(device)

                q_vals = q_net(state)  # (64,9)
                pred = q_vals.gather(1, action.unsqueeze(1)).squeeze()  # (64,)

                target_q_vals = target_net(new_state)

                next_q = target_q_vals.max(dim=1).values
                target = reward + GAMMA * next_q * (1 - is_done)  # (64,)

                loss = F.mse_loss(pred, target)
                losses.append(loss.item())

                loss.backward()

                optimizer.step()

            if not env.is_game_over():
                opponent.step()

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

    smoothed = pd.Series(losses).rolling(window=100).mean()
    plt.plot(smoothed)
    plt.xlabel('Update step')
    plt.ylabel('MSE Loss')
    plt.title('DQN Training Loss')
    plt.show()
    if save:
        torch.save(q_net.state_dict(), 'dqn_tiktaktoe.pth')
