import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from connectfour.agent import RandomAgent
from connectfour.environment import ConnectFour, Token
from dqn_model import QNet

_WEIGHTS_DIR = Path(__file__).parent.parent.parent / 'weights'


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

    def sample(self, batch_size: int) -> list[tuple[Tensor, int, int, Tensor, bool]]:
        return random.sample(self.buffer, batch_size)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen


def get_reward(env: ConnectFour, player: Token):
    if env.is_winner(player):
        return 1

    if env.is_winner(env.get_opponent(player)):
        return -1

    if env.is_draw():
        return 0

    return 0


def train_connectfour_dqn(episodes: int, save: bool = False):
    EPISODES = episodes
    MOVES = 7
    GAMMA = 0.9
    INPUT_DIMS = 6 * 7 * 3 + 2
    BUFFER_CAP = 10000
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001

    EPSILON = 1.0
    MIN_EPSILON = 0.01
    EPSILON_DECAY = 0.9999
    TARGET_UPDATE = 2000

    device = torch.device('cpu')

    makers = [Token.RED, Token.BLUE]
    q_net = QNet(INPUT_DIMS, MOVES).to(device)
    target_net = QNet(INPUT_DIMS, MOVES).to(device)
    buffer = ReplayBuffer(capacity=BUFFER_CAP)
    losses = []

    optimizer = optim.Adam(q_net.parameters(), LEARNING_RATE)

    target_net.load_state_dict(q_net.state_dict())

    env = ConnectFour()

    for ep in range(EPISODES + 1):
        env.reset()

        agent = random.choice(makers)
        opponent = RandomAgent(env, marker=env.get_opponent(agent))

        if agent == Token.BLUE and not env.is_game_over():
            opponent.step()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

        while not env.is_game_over():
            print(
                f'\r{ep} / {EPISODES} | Eps={EPSILON:.4f} | {device=}',
                end='',
                flush=True,
            )

            state = env.one_hot().to(device)

            with torch.no_grad():
                q_values: Tensor = q_net(state)

            mask = torch.full((MOVES,), float('-inf')).to(device)
            actions = env.actions()

            for action in actions:
                mask[action] = 0

            q_values = q_values + mask

            if random.random() > EPSILON:
                best_idx = q_values.argmax().item()
                best_action = next(action for action in actions if action == best_idx)
            else:
                best_action = random.choice(actions)

            env.move(best_action, agent)

            if not env.is_game_over():
                opponent.step()

            reward = get_reward(env, agent)
            new_state = env.one_hot().to(device)
            is_over = env.is_game_over()

            buffer.push(
                state=state,
                action=best_action,
                reward=reward,
                new_state=new_state,
                is_over=is_over,
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

                q_vals = q_net(state)
                pred = q_vals.gather(1, action.unsqueeze(1)).squeeze(1)

                target_q_vals = target_net(new_state)

                next_q = target_q_vals.max(dim=1).values
                target = reward + GAMMA * next_q * (1 - is_done)

                loss = F.mse_loss(pred, target)
                losses.append(loss.item())

                loss.backward()

                optimizer.step()

        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

    smoothed = pd.Series(losses).rolling(window=100).mean()
    plt.plot(smoothed)
    plt.xlabel('Update step')
    plt.ylabel('MSE Loss')
    plt.title('DQN Training Loss')
    plt.show()
    if save:
        _WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save(q_net.state_dict(), _WEIGHTS_DIR / 'connectfour_dqn.pth')
        print('saved weights')
