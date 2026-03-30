import random
from collections import deque
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from agents import DefaultAgent
from environment import Environment


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Tensor,
        action: int,
        valid_actions: list[int],
        reward: int,
        new_state: Tensor,
        is_over: bool,
    ):
        self.buffer.append((state, action, valid_actions, reward, new_state, is_over))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def get_reward(env: Environment, player):
    if env.is_winner(player):
        return 1
    if env.is_winner(env.get_opponent(player)):
        return -1
    return 0


def create_mask(
    moves: int, valid_actions: list, batch_size: int | None = None
) -> Tensor:
    if batch_size is None:
        mask = torch.full((moves,), float('-inf'))
        for action in valid_actions:
            mask[action] = 0
    else:
        mask = torch.full((batch_size, moves), float('-inf'))
        for i, actions in enumerate(valid_actions):
            for action in actions:
                mask[i, action] = 0
    return mask


def backward_pass(
    q_net, target_net, optimizer, buffer, batch_size, gamma, moves, losses
):
    optimizer.zero_grad()

    batch = buffer.sample(batch_size)
    state, action, valid_actions, reward, new_state, is_done = zip(*batch)

    state = torch.stack(state)
    new_state = torch.stack(new_state)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    is_done = torch.tensor(is_done, dtype=torch.float32)

    pred = q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        target_q_vals = target_net(new_state)
        mask = create_mask(moves, list(valid_actions), batch_size)
        next_q = (target_q_vals + mask).max(dim=1).values
        next_q = torch.where(is_done.bool(), torch.tensor(0.0), next_q)
        target = reward + gamma * next_q

    loss = F.mse_loss(pred, target)
    losses.append(loss.item())
    loss.backward()
    clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()


def train_dqn(
    env: Environment,
    markers: list,
    net_cls: type[nn.Module],
    input_dims: int,
    output_dims: int,
    episodes: int,
    save_path: Path | None = None,
    action_to_index: Callable | None = None,
    gamma: float = 0.9,
    batch_size: int = 64,
    buffer_cap: int = 10_000,
    lr: float = 0.0001,
    tau: float = 0.005,
    min_eps: float = 0.01,
):
    action_to_index = action_to_index or (lambda a: a)
    moves = output_dims

    eps = 1.0
    eps_decay = (min_eps / eps) ** (1 / episodes)

    device = torch.device('cpu')

    q_net = net_cls(input_dims, output_dims).to(device)
    target_net = net_cls(input_dims, output_dims).to(device)
    target_net.load_state_dict(q_net.state_dict())

    buffer = ReplayBuffer(capacity=buffer_cap)
    losses = []
    optimizer = optim.Adam(q_net.parameters(), lr)

    for ep in range(episodes):
        env.reset()

        agent = random.choice(markers)
        opponent = DefaultAgent(env, marker=env.get_opponent(agent))
        eps = eps * eps_decay

        if agent == markers[1] and not env.is_game_over():
            opponent.step()

        while not env.is_game_over():
            curr_loss = losses[-1] if losses else 0
            print(
                f'\r{ep + 1} / {episodes} | eps={eps:.4f} | loss: {curr_loss:.5f} | {device=}',
                end='',
                flush=True,
            )

            state = env.one_hot(agent).to(device)

            with torch.no_grad():
                q_values: Tensor = q_net(state).squeeze()

            actions = env.actions()
            action_indices = [action_to_index(a) for a in actions]
            mask = create_mask(moves, action_indices)
            q_values = q_values + mask

            if random.random() > eps:
                best_idx = int(q_values.argmax().item())
                best_action = next(a for a in actions if action_to_index(a) == best_idx)
            else:
                best_action = random.choice(actions)

            env.move(best_action, agent)

            if not env.is_game_over():
                opponent.step()

            reward = get_reward(env, agent)
            new_state = env.one_hot(agent).to(device)
            new_action_indices = [action_to_index(a) for a in env.actions()]

            buffer.push(
                state=state,
                action=action_to_index(best_action),
                valid_actions=new_action_indices,
                reward=reward,
                new_state=new_state,
                is_over=env.is_game_over(),
            )

            if len(buffer) >= batch_size:
                backward_pass(
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    buffer=buffer,
                    batch_size=batch_size,
                    gamma=gamma,
                    moves=moves,
                    losses=losses,
                )
                for target_param, param in zip(
                    target_net.parameters(), q_net.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

    print()

    smoothed = pd.Series(losses).rolling(window=100).mean()
    plt.plot(smoothed)
    plt.xlabel('Update step')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('DQN Training Loss')
    plt.show()

    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        torch.save(q_net.state_dict(), save_path)
        print(f'saved weights to {save_path}')
