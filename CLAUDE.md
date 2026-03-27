# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Play a game (pygame window)
cd src && uv run python main.py play --game tiktaktoe
cd src && uv run python main.py play --game connectfour

# Train Q-learning agent
cd src && uv run python main.py train --game tiktaktoe --episodes 50000 --save
cd src && uv run python main.py train --game connectfour --episodes 50000 --save

# Headless evaluation (no pygame)
cd src && uv run python main.py eval --game tiktaktoe --agent1 ql --agent2 minimax --runs 500
cd src && uv run python main.py eval --game connectfour --agent1 minimax --agent2 default --runs 100

# Lint / format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

Two fully separate game packages (`tiktaktoe/`, `connectfour/`) with intentional code duplication. All source lives in `src/`.

**CLI** (`main.py`): Three subcommands — `play`, `train`, `eval`.

**Each package has the same five modules:**

- `environment.py` — game state. Common interface: `move()`, `clear()`, `actions()`, `is_winner()`, `is_draw()`, `is_game_over()`, `winning_moves()`, `state_key()`, `copy()`, `reset()`, `draw()`. Each environment tracks `current_player` internally; `move()` advances it and `clear()` reverses it for minimax backtracking.

- `agent.py` — four agent types: `MinimaxAgent`, `QLearningAgent`, `DefaultAgent` (win > block > random), `RandomAgent`. All expose `.step()` and `.nodes_visited`.

- `minimax.py` — alpha-beta minimax. TikTakToe uses `minimax_alpha_beta` (returns a `MinimaxResult`). ConnectFour uses `minimax` with `non_terminal_score()` heuristic that scores 4-cell sliding windows.

- `q_learning_training.py` — trains a Q-table (dict) via epsilon-greedy against `DefaultAgent`. Saves to `q_table.pkl` / `q_table_connectfour.pkl` in the working directory.

- `evaluate.py` — headless evaluation loop: runs N games, alternates which agent goes first, prints win/draw report.

**Key asymmetry between games:**
- `TikTakToe.move(x, y, player)` — takes explicit row/col coordinates; `actions()` returns `[(x, y)]` tuples; agent code must unpack with `*move`.
- `ConnectFour.move(col, player)` — takes column only (gravity places the row); `actions()` returns plain ints.

**Q-table key format:** `(state_key_tuple, action)` where `state_key()` is a flat tuple of all board cells plus `current_player`. Including the current player in the state allows a single shared Q-table to encode both players' perspectives correctly.

**`play` mode agents** (hardcoded in `game.py`):
- TikTakToe: X = `QLearningAgent`, O = `MinimaxAgent`
- ConnectFour: RED = `MinimaxAgent` (depth 5), BLUE = `MinimaxAgent` (depth 5)

**Minimax depth defaults** (in `evaluate.py`):
- TikTakToe: `max_depth=None` (full tree, ~9 levels)
- ConnectFour: `max_depth=5`
