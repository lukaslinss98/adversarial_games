import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

_WEIGHTS_DIR = Path(__file__).parent.parent.parent / 'weights'

from agents import DefaultAgent
from tictactoe.environment import TicTacToe
from tictactoe.model import QNet


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

    def __len__(self):
        return len(self.buffer)


def get_reward(env: TicTacToe, player: str):
    if env.is_winner(player):
        return 1

    if env.is_winner(env.get_opponent(player)):
        return -1

    if env.is_draw():
        return 0

    return 0


action_to_index = lambda action: action[0] * 3 + action[1]


def backward_pass(q_net, target_net, optimizer, buffer, batch_size, gamma, losses):
    optimizer.zero_grad()

    batch = buffer.sample(batch_size)
    state, action, reward, new_state, is_done = zip(*batch)

    state = torch.stack(state)
    new_state = torch.stack(new_state)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    is_done = torch.tensor(is_done, dtype=torch.float32)

    pred = q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(new_state).max(dim=1).values
        next_q = torch.where(is_done.bool(), torch.tensor(0.0), next_q)
        target = reward + gamma * next_q

    loss = F.mse_loss(pred, target)
    losses.append(loss.item())
    loss.backward()
    clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()


def train_tictactoe_dqn(episodes: int, save: bool = False):
    EPISODES = episodes
    MOVES = 9
    GAMMA = 0.9
    INPUT_DIMS = 27
    BUFFER_CAP = 10_000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001

    EPSILON = 1.0
    MIN_EPSILON = 0.01
    EPSILON_DECAY = (MIN_EPSILON / EPSILON) ** (1 / EPISODES)
    TAU = 0.005

    device = torch.device('cpu')

    q_net = QNet(INPUT_DIMS, MOVES).to(device)
    target_net = QNet(INPUT_DIMS, MOVES).to(device)
    buffer = ReplayBuffer(capacity=BUFFER_CAP)
    losses = []

    optimizer = optim.Adam(q_net.parameters(), LEARNING_RATE)

    target_net.load_state_dict(q_net.state_dict())

    env = TicTacToe()

    for ep in range(EPISODES + 1):
        env.reset()

        agent = random.choice(['X', 'O'])
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))
        EPSILON = EPSILON * EPSILON_DECAY

        if agent == 'O' and not env.is_game_over():
            opponent.step()

        while not env.is_game_over():
            print(
                f'\r{ep} / {EPISODES} | Eps={EPSILON:.4f} | {device=}',
                end='',
                flush=True,
            )

            state = env.one_hot(agent).to(device)

            with torch.no_grad():
                q_values: Tensor = q_net(state)

            mask = torch.full((9,), float('-inf')).to(device)
            actions = env.actions()

            for action in actions:
                mask[action_to_index(action)] = 0

            q_values = q_values + mask

            if random.random() > EPSILON:
                best_idx = q_values.argmax().item()
                best_action = next(
                    action for action in actions if action_to_index(action) == best_idx
                )
            else:
                best_action = random.choice(actions)

            env.move(best_action, agent)

            if not env.is_game_over():
                opponent.step()

            reward = get_reward(env, agent)
            new_state = env.one_hot(agent).to(device)
            is_over = env.is_game_over()

            buffer.push(
                state=state,
                action=action_to_index(best_action),
                reward=reward,
                new_state=new_state,
                is_over=is_over,
            )

            if len(buffer) >= BATCH_SIZE:
                backward_pass(
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    buffer=buffer,
                    batch_size=BATCH_SIZE,
                    gamma=GAMMA,
                    losses=losses,
                )
                for target_param, param in zip(
                    target_net.parameters(), q_net.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )

    smoothed = pd.Series(losses).rolling(window=100).mean()
    plt.plot(smoothed)
    plt.xlabel('Update step')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('DQN Training Loss')
    plt.show()
    if save:
        _WEIGHTS_DIR.mkdir(exist_ok=True)
        torch.save(q_net.state_dict(), _WEIGHTS_DIR / 'tictactoe_dqn.pth')
        print('saved weights')
