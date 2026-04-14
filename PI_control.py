"""PI liquid-level controller backed by YOLO-based sensing.

Runs two threads: a sensing thread that reads frames from the camera
and publishes the estimated liquid height, and a control thread that
ticks every ``control_period_s`` to compute a pump speed via PI plus a
steady-state feed-forward. A final matplotlib plot summarises the run.
"""

import threading
import time

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from py_arduino import PyArduino

camera_index = 1

model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1
tank_height_cm = 8.0

setpoint_cm = 4.00

show_display = True

# PI gains
kp = 4.0
ki = 0.005

# Feed-forward steady-state pump speed that holds the setpoint.
state_steady_speed = 10

level_tolerance_cm = 0.05

# Fixed control period in seconds.
control_period_s = 3.0

# Pump speed limits.
max_pump_speed = 20
min_pump_speed = 0


def clamp(value, lower, upper):
    """Clamp ``value`` into the closed interval ``[lower, upper]``.

    Args:
        value: Input scalar to clamp.
        lower: Lower bound (inclusive).
        upper: Upper bound (inclusive).

    Returns:
        ``value`` bounded into ``[lower, upper]``.
    """
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


class SharedLevel:
    """Thread-safe carrier for the most recent liquid-height reading.

    Published by the sensing thread and consumed by the control thread.
    Includes a validity flag and timestamp so the consumer can detect
    stale data and fail safe.
    """

    def __init__(self):
        """Initialise empty state with the lock ready to use."""
        self.lock = threading.Lock()
        self.liquid_height_cm = None
        self.timestamp = 0.0
        self.valid = False

    def update(self, liquid_height_cm):
        """Publish the latest measurement.

        Args:
            liquid_height_cm: Estimated height in cm, or ``None`` if the
                detector could not produce a reading this frame.
        """
        with self.lock:
            self.liquid_height_cm = liquid_height_cm
            self.timestamp = time.time()
            self.valid = liquid_height_cm is not None

    def get(self):
        """Return the latest reading, its timestamp, and validity flag.

        Returns:
            Tuple ``(liquid_height_cm, timestamp, valid)``.
        """
        with self.lock:
            return self.liquid_height_cm, self.timestamp, self.valid


class SharedLog:
    """Thread-safe accumulator used for the post-run matplotlib plots.

    Stores parallel arrays rather than a list of dicts so that matplotlib
    can consume them directly with no further reshaping.
    """

    def __init__(self):
        """Initialise empty log buffers."""
        self.lock = threading.Lock()
        self.t = []
        self.level = []
        self.speed = []
        self.setpoint = []
        # Separate buffer for the integral term so we can sanity-check
        # anti-windup behaviour after a run.
        self.error_i = []

    def add(self, t, level, speed, setpoint, error_i=0.0):
        """Append one sample to the log.

        Args:
            t: Absolute timestamp (seconds since epoch).
            level: Measured liquid height in cm, or ``None``.
            speed: Pump speed command issued this tick.
            setpoint: Target level in cm.
            error_i: PI integral term recorded at this tick.
        """
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)
            self.error_i.append(error_i)

    def snapshot(self):
        """Return copies of the log buffers for plotting.

        Returns:
            Tuple ``(t, level, speed, setpoint, error_i)``.
        """
        with self.lock:
            return (
                self.t[:],
                self.level[:],
                self.speed[:],
                self.setpoint[:],
                self.error_i[:],
            )


def plot_results(log: SharedLog):
    """Show matplotlib plots for level, pump speed, and integral term.

    Args:
        log: Finalised log buffer from the control run.
    """
    t, level, speed, sp, error_i = log.snapshot()
    if len(t) < 2:
        print("Not enough data to plot.")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # Drop frames where sensing failed so the level line is contiguous.
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    plt.figure()
    plt.plot(t_level, level_valid, label="Liquid level (cm)")
    plt.axhline(setpoint_cm, linestyle="--", label="Setpoint (cm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Level (cm)")
    plt.title("Liquid level")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t_rel, speed, label="Pump speed (%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pump speed (%)")
    plt.title("Pump speed")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t_rel, error_i, label="Integral error")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral error")
    plt.title("Integral error")
    plt.grid(True)
    plt.legend()

    plt.show()


class PIController:
    """Discrete PI controller with feed-forward and clamping anti-windup.

    The feed-forward term ``steady_state_speed`` approximates the pump
    speed needed to hold the setpoint, letting the PI correction stay
    small. Anti-windup uses back-calculation so the integral cannot
    wind up while the output is saturated.
    """

    def __init__(
        self,
        kp,
        ki,
        steady_state_speed,
        max_speed,
        dt,
    ):
        """Store tuning parameters and reset the integral term.

        Args:
            kp: Proportional gain.
            ki: Integral gain (per second).
            steady_state_speed: Feed-forward pump speed for the setpoint.
            max_speed: Upper saturation limit for the pump speed.
            dt: Control period in seconds.
        """
        self.kp = kp
        self.ki = ki
        self.steady_state_speed = steady_state_speed
        self.max_speed = max_speed
        self.dt = dt
        self.integral_error = 0.0

    def update(self, setpoint, measurement):
        """Compute one pump speed command.

        Applies a dead-band to suppress chatter near the setpoint and
        adjusts the integral by back-calculation whenever the raw
        command saturates.

        Args:
            setpoint: Desired liquid height in cm.
            measurement: Current liquid height in cm.

        Returns:
            Tuple ``(speed, integral_error)`` where ``speed`` is an
            integer pump command and ``integral_error`` is the updated
            integral term.
        """
        error = setpoint - measurement

        # Dead-band: treat tiny errors as zero to avoid pump chatter.
        if abs(error) < level_tolerance_cm:
            error = 0.0

        self.integral_error += error * self.dt

        speed_raw = (
            self.steady_state_speed
            + self.kp * error
            + self.ki * self.integral_error
        )

        # Clamping anti-windup: bound the output, then back-calculate
        # the integral so it reflects the actually-delivered command.
        speed = clamp(speed_raw, min_pump_speed, self.max_speed)

        if speed_raw != speed:
            if self.ki != 0:
                self.integral_error = (
                    speed - self.steady_state_speed - self.kp * error
                ) / self.ki
            else:
                self.integral_error = 0.0

        speed = int(speed)

        return speed, self.integral_error

    def reset(self):
        """Zero the integral term.

        Intended for stale-data or startup fallbacks where the
        accumulated error is no longer meaningful.
        """
        self.integral_error = 0.0


class PumpController:
    """Inlet pump plus inlet/outlet valve driver.

    Opens both valves on construction so the rig is ready to flow, and
    guarantees a stopped pump plus closed valves on shutdown.
    """

    def __init__(
        self,
        board_type="minima",
        inlet_valve_pin=7,
        outlet_valve_pin=5,
    ):
        """Connect to the board and leave the rig in a ready state.

        Args:
            board_type: Arduino transport identifier for ``PyArduino``.
            inlet_valve_pin: Digital pin driving the inlet solenoid.
            outlet_valve_pin: Digital pin driving the outlet solenoid.
        """
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin

        self.open_all_valves()
        self.set_pump_speed(0)

    def open_all_valves(self):
        """Open both inlet and outlet valves and log the action."""
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print(
            "All valves opened "
            f"(inlet pin {self.inlet_valve_pin}, "
            f"outlet pin {self.outlet_valve_pin})"
        )

    def close_all_valves(self):
        """Close both inlet and outlet valves."""
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\nAll valves closed")

    def set_pump_speed(self, speed: int):
        """Send a pump speed command via the MCP4728 DAC.

        Args:
            speed: Pump speed in the 0-20 range.
        """
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        """Stop the pump and close all valves on exit."""
        self.set_pump_speed(0)
        self.close_all_valves()


class LiquidLevelDetector:
    """YOLO-based detector for tank and liquid level estimation.

    Encapsulates the model loading, inference, and pixel-to-cm conversion
    logic into a single cohesive unit.
    """

    def __init__(self, model_path, tank_height_cm, tank_class_id=1, liquid_class_id=0):
        """Initialise the YOLO model and physical parameters.

        Args:
            model_path: Path to the YOLO weights file.
            tank_height_cm: Physical height of the tank in cm.
            tank_class_id: Class index for the tank bounding box.
            liquid_class_id: Class index for the liquid level line.
        """
        self.model = YOLO(model_path)
        self.tank_height_cm = tank_height_cm
        self.tank_class_id = tank_class_id
        self.liquid_class_id = liquid_class_id

    def process_frame(self, frame):
        """Run inference on a frame and calculate the liquid height.

        Args:
            frame: BGR image from the camera.

        Returns:
            Tuple ``(tank_box, liquid_line, liquid_height_cm, inference_time_ms)``.
            Values are ``None`` if detection fails.
        """
        tank_box, liquid_line, inference_time_ms = self.detect(frame)

        liquid_height_cm = None
        if tank_box is not None and liquid_line is not None:
            liquid_height_cm = self.calculate_level(liquid_line, tank_box)

        return tank_box, liquid_line, liquid_height_cm, inference_time_ms

    def detect(self, frame):
        """Find the highest-confidence tank and liquid detections."""
        inference_start_time = time.perf_counter()
        results = self.model(frame, conf=0.9, verbose=False)[0]
        inference_time_ms = (time.perf_counter() - inference_start_time) * 1000.0

        tank_box = None
        best_tank_confidence = -1.0

        liquid_line = None
        best_liquid_confidence = -1.0

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if class_id == self.tank_class_id and conf > best_tank_confidence:
                best_tank_confidence = conf
                tank_box = (x1, y1, x2, y2)
            elif class_id == self.liquid_class_id and conf > best_liquid_confidence:
                best_liquid_confidence = conf
                liquid_line = y1

        return tank_box, liquid_line, inference_time_ms

    def calculate_level(self, liquid_line, tank_box):
        """Convert pixel coordinates to a physical height in cm.

        Uses the detected tank bounding box as the per-frame pixel-to-cm
        reference so changes in camera distance do not bias the estimate.
        """
        _, tank_top_y, _, tank_bottom_y = tank_box

        if tank_bottom_y <= tank_top_y:
            return None

        ratio = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y)
        liquid_height_cm = ratio * self.tank_height_cm

        return clamp(liquid_height_cm, 0.0, self.tank_height_cm)


def sensing_thread_fn(
    shared: SharedLevel,
    stop_event: threading.Event,
):
    """Capture frames and publish liquid-height estimates.

    Args:
        shared: Destination for the latest level reading.
        stop_event: Event that signals the thread to exit. The thread
            also sets this event on fatal errors so the control thread
            can shut down safely.
    """
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    detector = LiquidLevelDetector(
        model_path=model_path,
        tank_height_cm=tank_height_cm,
        tank_class_id=tank_class_id,
        liquid_class_id=liquid_class_id,
    )

    if not camera.isOpened():
        print("Failed to open camera.")
        stop_event.set()
        return

    window_name = "Liquidleveling"
    try:
        while not stop_event.is_set():
            ret, frame = camera.read()
            loop_start_time = time.perf_counter()

            if not ret:
                print("Failed to read frame.")
                stop_event.set()
                break

            tank_box, liquid_line, liquid_height_cm, inference_time_ms = detector.process_frame(frame)

            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                if liquid_line is not None and liquid_height_cm is not None:
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
                        "Liquidlevel is not detected.",
                        (30, 40),
                        cv2.FONT_ITALIC,
                        1,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "Tank is not detected.",
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

                # ESC closes the window and stops the run.
                if key & 0xFF == 27:
                    print("\nShutting down")
                    stop_event.set()
                    break

                # Manually closing the window also stops the run.
                visible = cv2.getWindowProperty(
                    window_name,
                    cv2.WND_PROP_VISIBLE,
                )
                if visible < 1:
                    print("\nShutting down")
                    stop_event.set()
                    break

            frame_display_done_time = time.perf_counter()
            capture_to_display_ms = (
                frame_display_done_time - loop_start_time
            ) * 1000
            
            if liquid_height_cm is not None:
                print(
                    f"level:{liquid_height_cm:.2f}cm, "
                    f"inference:{inference_time_ms:.1f}ms, "
                    f"frame:{capture_to_display_ms:.1f}ms",
                    end="\r",
                )
            else:
                print(
                    "level:None, "
                    f"inference:{inference_time_ms:.1f}ms, "
                    f"frame:{capture_to_display_ms:.1f}ms",
                    end="\r",
                )

    finally:
        camera.release()
        if show_display:
            cv2.destroyAllWindows()


def control_thread_fn(
    shared: SharedLevel,
    pump: PumpController,
    stop_event: threading.Event,
    log: SharedLog,
):
    """Tick the PI controller and drive the pump until stopped.

    Args:
        shared: Source of the latest liquid-height reading.
        pump: Pump and valve driver.
        stop_event: Event used to cooperatively stop the thread.
        log: Destination for per-tick samples used in the final plot.
    """
    pi_controller = PIController(
        kp,
        ki,
        state_steady_speed,
        max_pump_speed,
        control_period_s,
    )

    # Stop the pump if the last reading is older than this.
    stale_sec = 3.0

    next_tick = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            tick_start = time.time()

            liquid_height_cm, ts, valid = shared.get()
            age = time.time() - ts

            if (not valid) or (liquid_height_cm is None) or (age > stale_sec):
                pump.set_pump_speed(0)
                pi_controller.reset()
                log.add(
                    time.time(),
                    liquid_height_cm,
                    0,
                    setpoint_cm,
                    0.0,
                )
                next_tick = tick_start + control_period_s
                continue

            speed, integral_error = pi_controller.update(
                setpoint_cm,
                liquid_height_cm,
            )

            pump.set_pump_speed(speed)

            log.add(
                time.time(),
                liquid_height_cm,
                speed,
                setpoint_cm,
                integral_error,
            )

            next_tick = tick_start + control_period_s

    finally:
        pump.shutdown()


def main():
    """Start the sensing and control threads and plot the final run."""
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
        args=(shared, pump, stop_event, log),
        daemon=True,
    )

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupt received")
        stop_event.set()

    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)

    print("Shutdown complete")

    plot_results(log)


if __name__ == "__main__":
    main()
