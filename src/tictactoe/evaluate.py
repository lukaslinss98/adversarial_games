import pickle
import random
from pathlib import Path

import torch

from agents import (DefaultAgent, DQNAgent, MinimaxAgent, QLearningAgent,
                    RandomAgent)
from tictactoe.environment import TicTacToe
from minimax import minimax
from tictactoe.model import QNet

VALID_AGENTS = ('minimax', 'ql', 'dqn', 'default', 'random')
_MARKERS = ('X', 'O')
_WEIGHTS_DIR = Path(__file__).parent.parent.parent / 'weights'


def _load_q_table(agent1: str, agent2: str) -> dict | None:
    if 'ql' not in (agent1, agent2):
        return None
    with open(_WEIGHTS_DIR / 'tictactoe_ql.pkl', 'rb') as f:
        return pickle.load(f)


def _load_dqn_weights(agent1: str, agent2: str):
    if 'dqn' not in (agent1, agent2):
        return None
    return torch.load(_WEIGHTS_DIR / 'tictactoe_dqn_v2.pth', weights_only=True)


def create_agent(name: str, env, marker: str, q_table: dict | None, dqn_weights, minimax_depth=None, pruning=True):
    match name:
        case 'minimax':
            return MinimaxAgent(
                env,
                marker,
                minimax_fn=minimax,
                max_depth=minimax_depth,
                pruning=pruning,
            )
        case 'ql':
            return QLearningAgent(env, marker, q_table)
        case 'dqn':
            return DQNAgent(
                env,
                marker,
                dqn_weights,
                net=QNet,
                input_dims=27,
                output_dims=9,
                action_to_index=lambda a: a[0] * 3 + a[1],
            )
        case 'default':
            return DefaultAgent(env, marker)
        case 'random':
            return RandomAgent(env, marker)
        case _:
            raise ValueError(f'Unknown agent "{name}". Valid: {VALID_AGENTS}')


def evaluate_tictactoe(runs: int, agent1_type: str, agent2_type: str, minimax_depth=None, pruning=True) -> None:
    env = TicTacToe()
    q_table = _load_q_table(agent1_type, agent2_type)
    dqn_weights = _load_dqn_weights(agent1_type, agent2_type)

    agent1_wins = agent2_wins = draws = 0
    agent1_nodes: list[int] = []
    agent2_nodes: list[int] = []
    agent1_times: list[float] = []
    agent2_times: list[float] = []

    a1m, a2m = _MARKERS[0], _MARKERS[1]
    for i in range(runs):
        print(f'\rGame {i + 1}/{runs}', end='', flush=True)
        env.reset()
        a1m, a2m = a2m, a1m
        agent_map = {
            a1m: create_agent(agent1_type, env, a1m, q_table, dqn_weights, minimax_depth, pruning),
            a2m: create_agent(agent2_type, env, a2m, q_table, dqn_weights, minimax_depth, pruning),
        }

        while not env.is_game_over():
            agent_map[env.current_player].step()

        if env.is_draw():
            draws += 1
        elif env.is_winner(a1m):
            agent1_wins += 1
        else:
            agent2_wins += 1

        agent1_nodes.extend(agent_map[a1m].nodes_visited)
        agent2_nodes.extend(agent_map[a2m].nodes_visited)
        agent1_times.extend(agent_map[a1m].decision_times)
        agent2_times.extend(agent_map[a2m].decision_times)

    print()
    total = agent1_wins + agent2_wins + draws
    print(f'Results over {total} games:')
    print(f'  {agent1_type:<10} {agent1_wins:>5}  ({agent1_wins / total:.1%})')
    print(f'  {agent2_type:<10} {agent2_wins:>5}  ({agent2_wins / total:.1%})')
    print(f'  {"draws":<10} {draws:>5}  ({draws / total:.1%})')

    for name, nodes, times in (
        (agent1_type, agent1_nodes, agent1_times),
        (agent2_type, agent2_nodes, agent2_times),
    ):
        if nodes:
            mean_nodes = sum(nodes) / len(nodes)
            print(f'  {name} nodes — mean/move: {mean_nodes:.1f}, total: {sum(nodes)}')
        if times:
            mean_ms = sum(times) / len(times) * 1000
            print(
                f'  {name} time  — mean/move: {mean_ms:.2f}ms, total: {sum(times):.2f}s'
            )
