import pickle

from connectfour.agent import DefaultAgent, MinimaxAgent, QLearningAgent, RandomAgent
from connectfour.environment import ConnectFour, Token

VALID_AGENTS = ('minimax', 'ql', 'default', 'random')
_MARKERS = (Token.RED, Token.BLUE)


def _load_q_table(agent1: str, agent2: str) -> dict | None:
    if 'ql' not in (agent1, agent2):
        return None
    with open('q_table_connectfour.pkl', 'rb') as f:
        return pickle.load(f)


def _make_agent(name: str, env, marker: Token, q_table: dict | None):
    match name:
        case 'minimax':
            return MinimaxAgent(env, marker, max_depth=5)
        case 'ql':
            return QLearningAgent(env, marker, q_table)
        case 'default':
            return DefaultAgent(env, marker)
        case 'random':
            return RandomAgent(env, marker)
        case _:
            raise ValueError(f'Unknown agent "{name}". Valid: {VALID_AGENTS}')


def evaluate_connectfour(runs: int, agent1_type: str, agent2_type: str) -> None:
    env = ConnectFour()
    q_table = _load_q_table(agent1_type, agent2_type)

    agent1_wins = agent2_wins = draws = 0

    for i in range(runs):
        print(f'\rGame {i + 1}/{runs}', end='', flush=True)
        env.reset()
        a1m, a2m = (_MARKERS[0], _MARKERS[1]) if i % 2 == 0 else (_MARKERS[1], _MARKERS[0])
        agent_map = {
            a1m: _make_agent(agent1_type, env, a1m, q_table),
            a2m: _make_agent(agent2_type, env, a2m, q_table),
        }

        while not env.is_game_over():
            agent_map[env.current_player].step()

        if env.is_draw():
            draws += 1
        elif env.is_winner(a1m):
            agent1_wins += 1
        else:
            agent2_wins += 1

    print()
    total = agent1_wins + agent2_wins + draws
    print(f'Results over {total} games:')
    print(f'  {agent1_type:<10} {agent1_wins:>5}  ({agent1_wins / total:.1%})')
    print(f'  {agent2_type:<10} {agent2_wins:>5}  ({agent2_wins / total:.1%})')
    print(f'  {"draws":<10} {draws:>5}  ({draws / total:.1%})')
