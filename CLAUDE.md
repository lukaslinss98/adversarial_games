# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run a game (from src/ directory)
cd src && uv run python main.py --game tiktaktoe
cd src && uv run python main.py --game connectfour

# Run tests manually (no test framework — tests are standalone scripts)
cd src && uv run python test_connectfour.py

# Lint / format
uv run ruff check src/
uv run ruff format src/
```

## Architecture

This project implements AI-vs-AI adversarial board games using minimax with alpha-beta pruning. All source code lives in `src/`.

**Entry point:** `main.py` parses `--game` and delegates to `game.py`, which runs the pygame event loop.

**Layers:**

- `environments/` — game state classes (`TikTakToe`, `ConnectFour`). Each environment implements a common duck-typed interface: `move()`, `clear()`, `possible_moves()`, `check_winner()`, `is_draw()`, `is_game_over()`, `get_opponent()`, `copy()`, `winning_moves()`, `get_windows()` (for heuristic evaluation), and `draw()` (pygame rendering).

- `algorithms/minimax.py` — contains both a basic `minimax` (TikTakToe-only, unbounded depth) and `minimax_alpha_beta` (generic, supports `max_depth` cutoff and a `evaluate()` heuristic for non-terminal nodes). The heuristic scores sliding windows of 4 cells by counting player/opponent pieces.

- `agents.py` — `Agent` uses `minimax_alpha_beta` at `max_depth=8`, evaluating each candidate move from a copied board state. `SlighlySmarterAgent` is a rule-based fallback (win > block > random) — currently unused in `game.py`.

- `util.py` — shared pygame constants (colors, sizes, font singleton).

**Key design detail:** `ConnectFour.move(col, player)` takes only a column (gravity handles the row), while `TikTakToe.move(x, y, player)` takes explicit coordinates. `Agent._best_move` handles this asymmetry by unpacking moves with `*` — ConnectFour `possible_moves()` returns `[[col]]` (list of single-element lists) while TikTakToe returns `[(x, y)]` tuples.
