import time
import threading
import cv2
from ultralytics import YOLO
from PyArduino import PyArduino

camera_index = 0

model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1
tank_height_cm = 8.0

setpoint_cm = 4.00

show_display = True

# PI gain
Kp = 6.06
Ki = 0.0505

level_tolerance_cm = 0.05

# 제어 주기 고정
controltime_step_s = 1.0

def clamp(value, lower, upper):
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value

class SharedLevel:
    def __init__(self):
        self.lock = threading.Lock()
        self.liquid_height_cm = None
        self.timestamp = 0.0
        self.valid = False

    def update(self, liquid_height_cm):
        with self.lock:
            self.liquid_height_cm = liquid_height_cm
            self.timestamp = time.time()
            self.valid = liquid_height_cm is not None

    def get(self):
        with self.lock:
            return self.liquid_height_cm, self.timestamp, self.valid


class PI_control:
    """
    출력 범위: [-1.0, 1.0]
      +1.0: inlet 밸브 최대
      -1.0: outlet 밸브 최대
       0.0: 모든 밸브 off
    """
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.integral_max = 10.0
    
    def reset(self):
        self.integral = 0.0
    
    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement
        
        if abs(error) < level_tolerance_cm:
            error = 0.0
        
        p_term = self.kp * error
        
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral
        
        result = p_term + i_term
        return clamp(result, -1.0, 1.0)

class ValveController:
    """
    control_input in [-1, 1]
      control_input > 0 : inlet valve PWM (pin 7)
      control_input < 0 : outlet valve PWM (pin 5)
      control_input = 0 : all off
    """
    def __init__(self, board_type="minima", inlet_pin=7, outlet_pin=5):
        self.pa = PyArduino(board_type)
        self.inlet_pin = inlet_pin
        self.outlet_pin = outlet_pin
        self.all_off()

    def _write(self, inlet_on: bool, outlet_on: bool):
        # 동시 ON 방지
        if inlet_on and outlet_on:
            inlet_on = False
            outlet_on = False
        self.pa.run_digital_write(self.inlet_pin, inlet_on)
        self.pa.run_digital_write(self.outlet_pin, outlet_on)

    def all_off(self):
        self._write(False, False)

    def drive_for_period(self, control_input: float, period_s: float, stop_event: threading.Event):
        if stop_event.is_set():
            self.all_off()
            return

        control_input = clamp(control_input, -1.0, 1.0)
        duty = abs(control_input)

        if duty < 0.1:
            self.all_off()
            self._sleep_interruptible(period_s, stop_event)
            return

        on_time = duty * period_s
        off_time = period_s - on_time

        if control_input > 0:
            # inlet
            self._write(True, False)
            self._sleep_interruptible(on_time, stop_event)
            self._write(False, False)
            self._sleep_interruptible(off_time, stop_event)
        elif control_input < 0:
            # outlet
            self._write(False, True)
            self._sleep_interruptible(on_time, stop_event)
            self._write(False, False)
            self._sleep_interruptible(off_time, stop_event)
        else:
            self.all_off()
            self._sleep_interruptible(period_s, stop_event)

    @staticmethod
    def _sleep_interruptible(duration_s: float, stop_event: threading.Event):
        end = time.time() + max(0.0, duration_s)
        while time.time() < end:
            if stop_event.is_set():
                break
            time.sleep(0.02)


def detect_tank_and_liquid(frame, model):
    inference_start_time = time.perf_counter()
    results = model(frame, conf=0.9, verbose=False)[0]
    inference_time_ms = (time.perf_counter() - inference_start_time) * 1000.0

    tank_box = None
    best_tank_confidence = -1.0

    liquid_line = None
    best_liquid_confidence = -1.0

    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        if class_id == tank_class_id and conf > best_tank_confidence:
            best_tank_confidence = conf
            tank_box = (x1, y1, x2, y2)

        elif class_id == liquid_class_id and conf > best_liquid_confidence:
            best_liquid_confidence = conf
            liquid_line = y1

    return tank_box, liquid_line, inference_time_ms


def calculate_liquidlevel_cm(liquid_line, tank_box):
    _, tank_top_y, _, tank_bottom_y = tank_box
    
    if tank_bottom_y <= tank_top_y:
        return None
    
    liquid_height_cm = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * tank_height_cm
    liquid_height_cm = max(0.0, min(tank_height_cm, liquid_height_cm))
    return liquid_height_cm


def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    model = YOLO(model_path)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "Liquidleveling"
    try:
        while not stop_event.is_set():
            ret, frame = camera.read()
            inference_start_time = time.perf_counter()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                stop_event.set()
                break

            tank_box, liquid_line, inference_time_ms = detect_tank_and_liquid(frame, model)

            liquid_height_cm = None
            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if liquid_line is not None:
                    liquid_height_cm = calculate_liquidlevel_cm(liquid_line, tank_box)
                    cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)
                    cv2.putText(frame, f"Height: {liquid_height_cm:.2f}cm", (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Liquidlevel is not detected.", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tank is not detected.", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            shared.update(liquid_height_cm)

            if show_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10)
                if key & 0xFF == 27:
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            frame_display_done_time = time.perf_counter()
            capture_to_display_ms = (frame_display_done_time - inference_start_time) * 1000
            if liquid_height_cm is not None:
                print(f"level:{liquid_height_cm:.2f}cm, inference:{inference_time_ms:.1f}ms, frame:{capture_to_display_ms:.1f}ms", end="\r")
            else:
                print(f"level:None, inference:{inference_time_ms:.1f}ms, frame:{capture_to_display_ms:.1f}ms", end="\r")

    finally:
        camera.release()
        if show_display:
            cv2.destroyAllWindows()

def control_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    pi = PI_control(Kp, Ki)
    valve = ValveController(board_type="minima", inlet_pin=7, outlet_pin=5)

    # 센싱 데이터가 오래되면 안전 OFF
    STALE_SEC = 3.0

    next_tick = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            # 이번 tick 시작
            tick_start = time.time()

            liquid_height_cm, ts, valid = shared.get()
            age = time.time() - ts

            if (not valid) or (liquid_height_cm is None) or (age > STALE_SEC):
                valve.all_off()
                pi.reset()
                # 다음 tick 예약
                next_tick = tick_start + controltime_step_s
                continue

            # dt는 고정(=controltime_step_s)로 사용
            control_input = pi.update(setpoint_cm, liquid_height_cm, dt=controltime_step_s)

            # 이번 2초 동안 듀티로 밸브 구동
            valve.drive_for_period(control_input, controltime_step_s, stop_event)

            # 다음 tick 예약 (드리프트 최소화)
            next_tick = tick_start + controltime_step_s

    finally:
        valve.all_off()


def main():
    shared = SharedLevel()
    stop_event = threading.Event()

    t_sense = threading.Thread(target=sensing_thread_fn, args=(shared, stop_event), daemon=True)
    t_ctrl = threading.Thread(target=control_thread_fn, args=(shared, stop_event), daemon=True)

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    t_sense.join(timeout=2.0)
    t_ctrl.join(timeout=2.0)
    print("\n종료 완료.")


if __name__ == "__main__":
    main()