"""DQN-based liquid-level controller backed by YOLO-based sensing.

Runs two threads: a sensing thread that reads frames from the camera
and publishes the estimated liquid height, and a control thread that
evaluates the state using a pre-trained DQN agent to compute the 
optimal pump speed. A final matplotlib plot summarises the run.
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from py_arduino import PyArduino

# Ensure these match the values used during DQN_training
from dqn_training import (
    DQNAgent,
    action_dims,
    control_period_s,
    max_pump_speed,
    min_pump_speed,
    setpoint_cm,
    state_dims,
    tank_height_cm,
)

# Configuration
camera_index = 1
model_path = "20251223nano.pt"
dqn_model_path = "dqn_liquid_level_model.pth"

liquid_class_id = 0
tank_class_id = 1
show_display = True

# DQN validation parameters
epsilon = 0.05


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into the closed interval ``[lower, upper]``."""
    return max(lower, min(value, upper))


class SharedLevel:
    """Thread-safe carrier for the latest liquid-height reading."""

    def __init__(self):
        self.lock = threading.Lock()
        self.liquid_height_cm: Optional[float] = None
        self.timestamp: float = 0.0
        self.valid: bool = False

    def update(self, liquid_height_cm: Optional[float]):
        """Publish the latest measurement."""
        with self.lock:
            self.liquid_height_cm = liquid_height_cm
            self.timestamp = time.time()
            self.valid = liquid_height_cm is not None

    def get(self) -> Tuple[Optional[float], float, bool]:
        """Return the latest reading, timestamp, and validity flag."""
        with self.lock:
            return self.liquid_height_cm, self.timestamp, self.valid


class SharedLog:
    """Thread-safe accumulator used for the post-run plots."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t = []
        self.level = []
        self.speed = []
        self.setpoint = []
        self.reward = []
        self.action = []
        self.q_value = []

    def add(self, t, level, speed, setpoint, reward, action, q_value):
        """Append one sample to the log."""
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)
            self.reward.append(reward)
            self.action.append(action)
            self.q_value.append(q_value)

    def snapshot(self):
        """Return copies of the log buffers for plotting."""
        with self.lock:
            return (
                self.t[:],
                self.level[:],
                self.speed[:],
                self.setpoint[:],
                self.reward[:],
                self.action[:],
                self.q_value[:],
            )


class PumpController:
    """Inlet pump plus inlet/outlet valve driver."""

    def __init__(
        self,
        board_type="minima",
        inlet_valve_pin=7,
        outlet_valve_pin=5,
    ):
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin

        self.open_all_valves()
        self.set_pump_speed(0)

    def open_all_valves(self):
        """Open both inlet and outlet valves."""
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print("All valves opened.")

    def close_all_valves(self):
        """Close both inlet and outlet valves."""
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\nAll valves closed.")

    def set_pump_speed(self, speed: int):
        """Send a bounded pump speed command via the DAC."""
        speed = int(np.clip(speed, min_pump_speed, max_pump_speed))
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        """Stop the pump and close all valves on exit."""
        self.set_pump_speed(0)
        self.close_all_valves()


class LiquidLevelDetector:
    """YOLO-based detector for tank and liquid level estimation."""

    def __init__(
        self,
        model_path,
        tank_height,
        tank_class_id=1,
        liquid_class_id=0,
    ):
        self.model = YOLO(model_path)
        self.tank_height_cm = tank_height
        self.tank_class_id = tank_class_id
        self.liquid_class_id = liquid_class_id

    def process_frame(self, frame):
        """Detect tank and liquid and return the derived level."""
        tank_box, liquid_line, inference_time_ms = self.detect(frame)

        liquid_height_cm = None
        if tank_box is not None and liquid_line is not None:
            liquid_height_cm = self.calculate_level(
                liquid_line, tank_box
            )

        return tank_box, liquid_line, liquid_height_cm, inference_time_ms

    def detect(self, frame):
        """Return the highest-confidence tank box and liquid line."""
        inference_start_time = time.perf_counter()
        results = self.model(frame, conf=0.9, verbose=False)[0]
        inference_time_ms = (
            time.perf_counter() - inference_start_time
        ) * 1000.0

        tank_box = None
        best_tank_confidence = -1.0

        liquid_line = None
        best_liquid_confidence = -1.0

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            is_tank = class_id == self.tank_class_id
            is_liquid = class_id == self.liquid_class_id

            if is_tank and conf > best_tank_confidence:
                best_tank_confidence = conf
                tank_box = (x1, y1, x2, y2)
            elif is_liquid and conf > best_liquid_confidence:
                best_liquid_confidence = conf
                liquid_line = y1

        return tank_box, liquid_line, inference_time_ms

    def calculate_level(self, liquid_line, tank_box):
        """Linear-interpolate the liquid height between tank edges."""
        _, tank_top_y, _, tank_bottom_y = tank_box

        if tank_bottom_y <= tank_top_y:
            return None

        tank_pixel_height = float(tank_bottom_y - tank_top_y)
        liquid_pixel_height = float(tank_bottom_y - liquid_line)
        ratio = liquid_pixel_height / tank_pixel_height
        liquid_height_cm = ratio * self.tank_height_cm

        return clamp(liquid_height_cm, 0.0, self.tank_height_cm)


def sensing_thread_fn(
    shared: SharedLevel, stop_event: threading.Event
):
    """Capture frames and publish liquid-height estimates."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    detector = LiquidLevelDetector(
        model_path=model_path,
        tank_height=tank_height_cm,
        tank_class_id=tank_class_id,
        liquid_class_id=liquid_class_id,
    )

    if not camera.isOpened():
        print("Failed to open camera.")
        stop_event.set()
        return

    window_name = "DQN Liquid Level Control"

    try:
        while not stop_event.is_set():
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame.")
                stop_event.set()
                break

            tank_box, liquid_line, liquid_height_cm, _ = (
                detector.process_frame(frame)
            )

            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )

                has_liquid = (
                    liquid_line is not None
                    and liquid_height_cm is not None
                )
                if has_liquid:
                    cv2.line(
                        frame,
                        (x1, liquid_line),
                        (x2, liquid_line),
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Height: {liquid_height_cm:.2f}cm",
                        (30, 40),
                        cv2.FONT_ITALIC,
                        1,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Liquid not detected.",
                        (30, 40),
                        cv2.FONT_ITALIC,
                        1,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "Tank not detected.",
                    (30, 40),
                    cv2.FONT_ITALIC,
                    1,
                    (0, 0, 255),
                    2,
                )

            shared.update(liquid_height_cm)

            if show_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10)

                if key & 0xFF == 27:
                    stop_event.set()
                    break

                window_closed = cv2.getWindowProperty(
                    window_name, cv2.WND_PROP_VISIBLE
                ) < 1
                if window_closed:
                    stop_event.set()
                    break

    finally:
        camera.release()
        if show_display:
            cv2.destroyAllWindows()


def control_thread_fn(
    shared: SharedLevel,
    pump: PumpController,
    stop_event: threading.Event,
    log: SharedLog,
    agent: DQNAgent,
):
    """Tick the DQN agent and drive the pump until stopped."""
    print("\n[Validation Mode]")
    print(f"Action range: [{min_pump_speed} - {max_pump_speed}] (Integer)")
    print(f"State dimensions: {state_dims}\n")

    agent.epsilon = float(epsilon)
    agent.reset_episode()

    next_tick = time.time()
    stale_sec = 3.0

    try:
        while not stop_event.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            tick_start = time.time()

            h, ts, valid = shared.get()
            age = time.time() - ts

            is_stale = (not valid) or (h is None) or (age > stale_sec)
            if is_stale:
                pump.set_pump_speed(0)
                agent.u_prev = 0.0
                log.add(
                    time.time(), h, 0, setpoint_cm, 0.0, -1, 0.0
                )
                next_tick = tick_start + control_period_s
                continue

            state = agent.build_state(h)
            action = agent.select_action(state, training=False)
            u_cmd = int(agent.action_to_u(action))

            with torch.no_grad():
                st = torch.tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                q_values = agent.q_net(st)[0]
                q_selected = float(q_values[action].item())

            pump.set_pump_speed(u_cmd)

            # Reward uses the same definition as training so the
            # logged values are directly comparable.
            reward = agent.compute_reward(h, float(u_cmd))
            agent.u_prev = float(u_cmd)
            agent.error_int = float(np.clip(
                agent.error_int
                + (setpoint_cm - h) * control_period_s,
                -10.0,
                10.0,
            ))

            log.add(
                time.time(),
                h,
                u_cmd,
                setpoint_cm,
                reward,
                action,
                q_selected,
            )

            print(
                f"[h={h:.2f}cm | action={action:2d} -> u={u_cmd:2d}] "
                f"| [Q={q_selected:6.2f} | reward={reward:6.2f}]"
            )

            next_tick = tick_start + control_period_s

    finally:
        print("\nControl terminated.")
        pump.shutdown()


def plot_results(log: SharedLog):
    """Show matplotlib plots for level, pump speed, reward, and Q-value."""
    t, level, speed, sp, reward, action, q_value = log.snapshot()
    if len(t) < 2:
        print("Not enough data to plot.")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    ax1.plot(
        t_level, level_valid, linewidth=2, label="Liquid level"
    )
    ax1.axhline(
        setpoint_cm,
        linestyle="--",
        color="r",
        linewidth=2,
        label="Setpoint",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Level (cm)")
    ax1.set_title("Liquid Level Control (DQN)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        t_rel, speed, linewidth=2, color="orange", label="Pump speed"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed")
    ax2.set_title("Control Input")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(
        t_rel, reward, linewidth=2, color="green", label="Reward"
    )
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Step Reward")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(
        t_rel, q_value, linewidth=2, color="purple", label="Q-value"
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Q-value")
    ax4.set_title("Selected Action Q-value")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Start the sensing and DQN control threads, then plot the final run."""
    print("Initializing DQN Validation Mode...")

    agent = DQNAgent(state_dims, action_dims)

    try:
        agent.load(dqn_model_path)
        print(f"Model loaded successfully: {dqn_model_path}")
    except Exception as e:
        print(f"Model load failed: {dqn_model_path}\nError: {e}")
        return

    shared = SharedLevel()
    log = SharedLog()
    stop_event = threading.Event()
    pump = PumpController(
        board_type="minima",
        inlet_valve_pin=7,
        outlet_valve_pin=5,
    )

    t_sense = threading.Thread(
        target=sensing_thread_fn,
        args=(shared, stop_event),
        daemon=True,
    )
    t_ctrl = threading.Thread(
        target=control_thread_fn,
        args=(shared, pump, stop_event, log, agent),
        daemon=True,
    )

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupt received. Shutting down...")
        stop_event.set()

    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)

    print("Shutdown complete.")
    plot_results(log)


if __name__ == "__main__":
    main()
