RUNNING THE APP:

With uv
--------------------
uv sync
cd src && uv run python src/main.py <command>

Without uv
----------
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src && python main.py <command>

Requires Python 3.13+.

COMMANDS:

play: watch agents play in a pygame window
  python main.py play --game tictactoe
  python main.py play --game connectfour
  python main.py play --game tictactoe --agent1 minimax --agent2 random
  python main.py play --game connectfour --agent1 minimax --agent2 dqn --minimax-depth 5

eval: headless evaluation
  python main.py eval --game tictactoe --agent1 minimax --agent2 random --runs 100
  python main.py eval --game connectfour --agent1 minimax --agent2 default --runs 100
  python main.py eval --game tictactoe --all --runs 200

train — train a Q-learning or DQN agent
  python main.py train --game tictactoe --episodes 50000 --save
  python main.py train --game connectfour --episodes 50000 --algo dqn --save

Valid agents: minimax, ql, dqn, default, random

Run python main.py <command> --help for all options.
