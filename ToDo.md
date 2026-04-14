# ToDo

This file is a cumulative command history for Claude's work on this repo. Append new entries below; never overwrite prior ones.

---

## 2026-04-14 | MIT convention cleanup (PI_control.py, PyArduino.py)

- [x] Rename `PyArduino.py` → `py_arduino.py`, update imports in `Main.py`, `P_control.py`, `PI_control.py`, `P_control_improved.py`, `MPC_control_improved.py`, `DQN_control.py`
- [x] Rename classes per MIT table: `sharedlevel`→`SharedLevel`, `sharedlog`→`SharedLog`, `PI_controller`→`PIController`, `pump_controller`→`PumpController`
- [x] Rename constants to lowercase: `Kp`→`kp`, `Ki`→`ki`, `STALE_SEC`→`stale_sec`
- [x] Translate all Korean comments, print messages, and cv2 overlay strings in `PI_control.py` and `py_arduino.py` to English
- [x] Add Google-style docstrings (What/Why; Args/Returns/Raises) to public API in both files
- [x] Enforce 80-column limit; operators on the left side of continuation lines
- [x] Pass `ruff check` and `ruff format --check` on `PI_control.py` and `py_arduino.py`
- [x] Verify no accidental behavior changes with `git diff` and `python -m py_compile` across all affected files
