# Agent Guidelines for Adversarial Games

## Project Overview

This is a Tic Tac Toe game with AI agents using minimax algorithm. The project uses Python with pygame for the game display and follows a modular structure with environments, agents, and algorithms.

## Directory Structure

```
adversarial_games/
├── src/
│   ├── main.py           # Entry point, pygame game loop
│   ├── agent.py          # Agent class for making moves
│   ├── util.py           # Constants and utilities
│   ├── environments/
│   │   └── tiktaktoe.py  # TikTakToe game environment
│   └── algorithms/
│       └── minimax.py     # Minimax AI algorithm
├── pyproject.toml        # Project configuration
└── AGENTS.md            # This file
```

## Commands

### Running the Application

```bash
# Run the game
uv run src/main.py
```

### Linting and Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Install ruff if needed
uv add --dev ruff

# Lint all files
uv run ruff check .

# Format all files
uv run ruff format .

# Lint a specific file
uv run ruff check src/main.py

# Lint and fix auto-fixable issues
uv run ruff check --fix .
```

### Running Tests

```bash
# Run all tests with pytest
uv run pytest

# Run a single test file
uv run pytest tests/test_file.py::TestClass::test_method

# Run tests matching a pattern
uv run pytest -k "test_pattern"
```

## Code Style Guidelines

### Imports

- Use absolute imports from the project root (`src/`)
- Group imports: standard library, third-party, local
- Example:

```python
# Standard library
import random
from typing import Optional

# Third-party
import pygame

# Local (absolute from src)
from agent import Agent
from environments.tiktaktoe import TikTakToe
```

### Formatting

- Line length: 88 characters (ruff default)
- Use Black-compatible formatting
- Use single quotes (`'`) for strings, not double quotes
- Use trailing commas in multi-line constructs
- One blank line between top-level definitions

### Type Hints

- Add type hints to function signatures and return types
- Keep type hints minimal but functional
- Use `Optional[T]` instead of `T | None` for compatibility
- Example:

```python
def minimax(state: TikTakToe) -> tuple[int, int] | None:
    ...
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `TikTakToe`, `Agent`)
- **Functions/variables**: snake_case (e.g., `possible_moves`, `game_board`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `WINDOW_SIZE`, `WIN_LINES`)
- **Modules**: snake_case (e.g., `tiktaktoe.py`, `minimax.py`)

### Error Handling

- Use try/except for operations that may fail
- Keep error messages descriptive
- Example:

```python
try:
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
except pygame.error as e:
    print(f"Failed to initialize display: {e}")
    raise
```

### Pygame Specific

- Initialize pygame at the start of main(): `pygame.init()`
- Create fonts lazily (after pygame.init()) to avoid "font not initialized" errors
- Handle QUIT event properly in game loops:

```python
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        running = False
```

- Always call `pygame.quit()` at the end to clean up

### General Best Practices

- Keep functions focused and single-purpose
- Add docstrings for public APIs and complex logic
- Avoid printing in production code (use logging instead)
- Use immutable collections (tuples) for fixed data
- Make defensive copies when returning internal data

## Configuration

### pyproject.toml

The project uses uv for dependency management. Current dependencies:

```toml
[project]
requires-python = ">=3.13"
dependencies = ["pygame>=2.6.1"]
```

### Ruff Configuration (optional)

Add to pyproject.toml for custom linting rules:

```toml
[tool.ruff]
line-length = 88
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]
```

## Notes for Agents

- This is a small project, keep changes focused and minimal
- The minimax algorithm in `algorithms/minimax.py` currently returns random moves
- The game loop in `main.py` runs an AI vs AI match by default
- When modifying the game state, ensure proper copy semantics (use `.copy()` method)
- Test any changes by running `uv run src/main.py`