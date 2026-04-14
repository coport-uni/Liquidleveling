# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repo adopts the conventions from [CommonClaude](https://github.com/coport-uni/CommonClaude). Sections 1–5 below reproduce those project-wide standards; the "Project Overview" section adds Liquidleveling-specific architecture notes; Section 6 documents the hooks installed in this repo.

---

## Project Overview

Liquid-level control system for a physical tank. A USB camera + YOLO model measures liquid height in real time, and a control algorithm drives a pump through an Arduino Uno R4 (via Telemetrix) to track a setpoint. Several controllers are implemented side-by-side as alternatives: P, PI, improved P, MPC (adaptive, with RLS online system ID), and DQN.

### Running

Each controller is a standalone script — pick one and run it directly:

```bash
python P_control.py              # proportional controller
python PI_control.py             # PI controller
python P_control_improved.py     # P with additional logic
python MPC_control_improved.py   # adaptive MPC + RLS online identification
python DQN_control.py            # runs trained DQN policy (needs dqn_liquid_level_model.pth)
python DQN_training.py           # trains the DQN and writes the .pth model
python "sensing(YOLO).py"        # vision-only: shows YOLO detections + level estimate
python Main.py                   # minimal Arduino digital-write example
python check_camera.py           # list available cameras
```

No build step and no test suite. Lint with `ruff` (see §5).

### Environment

Runs inside a Docker container (`--privileged`, Ubuntu 24.04 Noble) with Claude Code (CLI / VS Code extension) as the primary dev tool.

### Hardware assumptions

- **Arduino Uno R4** (Minima or WiFi) flashed with Telemetrix firmware. Selected by string `"minima"` or `"wifi"` passed to `PyArduino(...)`.
- **MCP4728** DAC over I2C drives pump voltage. `PyArduino.run_pump_speed` clamps to 0–20 and applies a `5/3.3` voltage factor plus a small `voltage_percentage/50` error compensation term — this is real hardware calibration, not cruft.
- **Camera index** is hard-coded per script (`camera_index = 0` or `1`), using `cv2.CAP_DSHOW` (Windows DirectShow backend).
- **YOLO weights**: `20251223nano.pt` (tracked). Classes `0 = liquid`, `1 = tank`; tank height 8.0 cm; liquid height is linear-interpolated between the detected tank top/bottom Y pixels and the liquid Y.

### Architecture

Control scripts share the same threaded pattern (duplicated, not factored):

1. **Sensing thread** — grabs frames, runs YOLO, computes liquid height, updates a lock-protected `SharedLevel`.
2. **Control thread** — every `control_period_s` (typically 3.0 s), reads the latest level, computes a pump command, calls `PyArduino.run_pump_speed(...)`.
3. **Logging** — a `SharedLog` buffer accumulates samples; on exit matplotlib plots the run.

When modifying one controller, check whether the same scaffolding needs updating in the others.

`PyArduino.get_digital_state` / `get_analog_state` use a module-global set by the Telemetrix async callback — intentional for the callback lifecycle, don't refactor to instance state without understanding it. `DQN_control.py` imports `DQNAgent` and hyperparameters from `DQN_training.py`; state dims and action count must match. MPC uses a first-order model `h[k+1] = a·h[k] + b·u[k] + c` identified online via RLS and solved with `scipy.optimize.minimize` (prediction horizon 5).

---

## 1. MIT Code Convention

All code follows the [MIT CommLab Coding and Comment Style](https://mitcommlab.mit.edu/broad/commkit/coding-and-comment-style/).

### Naming

- **Variables and classes** are nouns; **functions and methods** are verbs.
- Names must be pronounceable and straightforward.
- Name length is proportional to scope: short for local, descriptive for broad.
- Avoid abbreviations unless self-explanatory. If unavoidable, define them in a comment block.
- Python conventions:

| Element    | Style        | Example               |
|------------|--------------|-----------------------|
| Variable   | `lower_case` | `joint_angle`         |
| Function   | `lower_case` | `send_action`         |
| Class      | `CamelCase`  | `FairinoFollower`     |
| Constant   | `lower_case` | `_settle_mid_s`       |
| Module     | `lowercase`  | `fairino_follower`    |

### Structure

- **80-column limit** for all new code.
- One statement per line.
- Indent with **4 spaces** (never tabs).
- Place operators on the **left side** of continuation lines so the reader can see at a glance that a line continues.
- Group related items visually with alignment.

### Spacing

- One space after commas, none before: `foo(a, b, c)`.
- One space on each side of `=`, `==`, `<`, `>`, etc.
- Be consistent with arithmetic operators within a file.

### Comments

- Use **complete sentences**.
- Only comment for **context** or **non-obvious choices**. Never restate what the code already says.
- Outdated comments are worse than none. Keep them current or delete them.
- TODO format:
  ```python
  # TODO: (@owner) Implement 2-step predictor-corrector
  # for stability. Adams-Bashforth causes shocks.
  ```

### Language

- All code comments, docstrings, commit messages, documentation files (including README), **GitHub issues, and pull requests** must be written in **English**.

### Documentation

- All public functions and classes must have **docstrings** (PEP 257 / Google style).
- A docstring states **what** and **why**, not **how**.
- Include `Args:`, `Returns:`, and `Raises:` sections when applicable.

---

## 2. Debug File Management

All debug, exploratory, and throwaway test scripts must be saved in `claude_test/`, **not** in `tests/`.

### Rules

| Location        | What goes there                                      |
|-----------------|------------------------------------------------------|
| `tests/`        | Production-quality tests that are part of CI/CD.     |
| `claude_test/`  | Debug scripts, one-off experiments, diagnostic code. |

### When writing debug code

1. Create the file directly in `claude_test/` (e.g., `claude_test/debug_servo_timing.py`).
2. Add a one-line docstring at the top explaining the purpose.
3. If the debug script leads to a real fix, move the relevant parts into a proper test under `tests/` and delete or archive the debug version.

### README

`claude_test/README.md` is the index. When adding a new debug file, add a row to the table in that README describing what the file does and what was learned.

---

## 3. Task Management

> **MANDATORY**: This workflow applies to **every task without exception**, regardless of size or complexity. No task may begin without writing `ToDo.md` and creating a GitHub issue via `gh`. Skipping any step is not allowed.

### Rules

1. **Write ToDo.md**: For every task requested by the user, create a `ToDo.md` file and confirm the contents with the user before starting work.
2. **Accumulate ToDo.md**: Do not overwrite previous entries in `ToDo.md`. Always **append** new tasks below existing ones so that the file serves as a cumulative command history for Claude's actions.
3. **Register GitHub issues**: When possible, use the `gh` CLI to register the Todo list and details as a GitHub issue.

### Command Input Validation

Before writing ToDo.md, the following two checks must be performed:

1. **Is the command explicit?**: If the request is ambiguous or open to interpretation, do not start work. Instead, ask the user for specifics:
   - What is being changed? (target)
   - How is it being changed? (method)
   - Why is it being changed? (purpose)
2. **Are there reference materials?**: Check whether related PDFs, websites, or documents exist. If so, review them before incorporating into the work.

> Do not proceed if either check is not satisfied.

### Workflow

1. Receive the user's task request and **validate the command input**.
2. Once validated, organize the task list in `ToDo.md`.
3. Get the user's confirmation on the `ToDo.md` contents.
4. Once confirmed, create a GitHub issue via `gh issue create`.
5. Check off completed items in `ToDo.md` as work progresses.
6. Update the GitHub issue via `gh issue edit` for completed items.
7. **Commit and push** changes after every user command is completed.

> **Reminder**: Steps 2 (`ToDo.md`) and 4 (`gh issue create`) are **non-negotiable**. Every task must have a corresponding `ToDo.md` entry and a GitHub issue before any work begins.

---

## 4. Testing Rules

Tests exist to verify the **correctness and quality** of code. Code quality must never be sacrificed just to pass tests.

### Rules

1. **No magic numbers**: Do not use arbitrary numbers or values directly to pass tests. All values must be defined as meaningful constants or variables.
   ```python
   # Bad: passing tests with magic numbers
   def calculate_area(radius):
       return 3.14 * radius * radius  # Why 3.14?

   # Good: use meaningful constants
   import math

   def calculate_area(radius):
       return math.pi * radius * radius
   ```

2. **No hardcoding**: Do not hardcode values to match expected test results. Code must work through correct logic, not through branches or fixed values tailored to specific inputs.
   ```python
   # Bad: hardcoded to match test inputs
   def convert_temperature(celsius):
       if celsius == 100:
           return 212
       if celsius == 0:
           return 32
       return celsius * 1.8 + 32

   # Good: correct logic implementation
   def convert_temperature(celsius):
       return celsius * 1.8 + 32
   ```

3. **Code quality first**: Prioritize readability, maintainability, and correctness over whether tests pass. If a test fails, fix the logic correctly rather than tricking the test.

---

## 5. Linting

All Python code must pass **Ruff** checks before committing.

### Rules

1. **Line length**: 80 columns (`line-length = 80` in `ruff.toml`).
2. **Run on every commit**: Before committing, run:
   ```bash
   ruff check <file>.py
   ruff format --check <file>.py
   ```
3. **Fix before committing**: If either command reports errors, fix them before proceeding. Use `ruff format <file>.py` to auto-format.

---

## 6. Claude Code Hooks

This repo ships hooks under [.claude/hooks/](.claude/hooks/) wired up by [.claude/settings.json](.claude/settings.json):

| Hook                              | Trigger        | Purpose                                                      |
|-----------------------------------|----------------|--------------------------------------------------------------|
| `pre-write-guard.sh`              | PreToolUse (Write/Edit) | Blocks debug-named files (`debug_*`, `scratch_*`, `tmp_*`, `experiment_*`, `test_debug_*`) from being written into `tests/`. |
| `post-write-lint.sh`              | PostToolUse (Write/Edit) | Runs `ruff check` + `ruff format --check` on modified `.py` files and blocks on violations. |
| `post-write-debug-remind.sh`      | PostToolUse (Write/Edit) | Reminds to update `claude_test/README.md` when a file is added under `claude_test/`. |
| Stop prompt                       | Stop           | Verifies `ToDo.md` entry, GitHub issue, ruff pass, and `claude_test/README.md` update before allowing the session to end. |

Hooks require `jq` on PATH, and `ruff` for the lint hook.
