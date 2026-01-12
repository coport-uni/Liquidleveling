import time
import threading
import cv2
from ultralytics import YOLO
from PyArduino import PyArduino

# ===================== 사용자 설정 =====================
camera_index = 1
model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1
tank_height_cm = 8.0

SETPOINT_CM = 4.00          # 목표 수위(cm)
CONF_TH = 0.90
SHOW_WINDOW = True

# PI gains (K=0.33, tau=120s -> IMC PI with lambda=60s)
Kp = 6.06
Ki = 0.0505                 # 1/s

DEADBAND_CM = 0.05
I_MIN, I_MAX = -1.0, 1.0

# 제어 주기(샘플링 타임) 고정
CONTROL_PERIOD = 1.0        # seconds

# 듀티 컷오프: 너무 작으면 OFF로 처리
DUTY_CUTOFF = 0.1
# =======================================================


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class SharedLevel:
    def __init__(self):
        self.lock = threading.Lock()
        self.level_cm = None
        self.timestamp = 0.0
        self.valid = False

    def update(self, level_cm):
        with self.lock:
            self.level_cm = level_cm
            self.timestamp = time.time()
            self.valid = level_cm is not None

    def get(self):
        with self.lock:
            return self.level_cm, self.timestamp, self.valid


class PIController:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.i = 0.0

    def reset(self):
        self.i = 0.0

    def update(self, setpoint, measurement, dt):
        e = setpoint - measurement

        if abs(e) < DEADBAND_CM:
            e = 0.0

        # integrate (anti-windup)
        self.i += e * dt
        self.i = clamp(self.i, I_MIN, I_MAX)

        u = self.kp * e + self.ki * self.i
        return clamp(u, -1.0, 1.0)


class ValveController:
    """
    u in [-1, 1]
      u > 0 : inlet valve PWM (pin 7)
      u < 0 : outlet valve PWM (pin 5)
      u = 0 : all off
    - 여기서는 제어 주기=2초 안에서 듀티 구동만 수행 (별도 스위칭 제한 없음)
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

    def drive_for_period(self, u: float, period_s: float, stop_event: threading.Event):
        """
        period_s 동안 u에 해당하는 듀티로 밸브를 구동하고 리턴.
        """
        if stop_event.is_set():
            self.all_off()
            return

        u = clamp(u, -1.0, 1.0)
        duty = abs(u)

        if duty < DUTY_CUTOFF:
            self.all_off()
            self._sleep_interruptible(period_s, stop_event)
            return

        on_time = duty * period_s
        off_time = period_s - on_time

        if u > 0:
            # inlet
            self._write(True, False)
            self._sleep_interruptible(on_time, stop_event)
            self._write(False, False)
            self._sleep_interruptible(off_time, stop_event)
        elif u < 0:
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
    t0 = time.perf_counter()
    results = model(frame, conf=CONF_TH, verbose=False)[0]
    infer_ms = (time.perf_counter() - t0) * 1000.0

    tank_box = None
    best_tank_conf = -1.0
    liquid_y = None
    best_liquid_conf = -1.0

    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        if class_id == tank_class_id and conf > best_tank_conf:
            best_tank_conf = conf
            tank_box = (x1, y1, x2, y2)
        elif class_id == liquid_class_id and conf > best_liquid_conf:
            best_liquid_conf = conf
            liquid_y = y1

    return tank_box, liquid_y, infer_ms


def calculate_liquidlevel_cm(liquid_y, tank_box):
    x1, tank_top_y, x2, tank_bottom_y = tank_box
    if tank_bottom_y <= tank_top_y:
        return None
    level_cm = float(tank_bottom_y - liquid_y) / float(tank_bottom_y - tank_top_y) * tank_height_cm
    return max(0.0, min(tank_height_cm, level_cm))


def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    model = YOLO(model_path)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "Liquidleveling"
    try:
        while not stop_event.is_set():
            ret, frame = camera.read()
            t0 = time.perf_counter()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                stop_event.set()
                break

            tank_box, liquid_y, infer_ms = detect_tank_and_liquid(frame, model)

            level_cm = None
            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if liquid_y is not None:
                    level_cm = calculate_liquidlevel_cm(liquid_y, tank_box)
                    cv2.line(frame, (x1, liquid_y), (x2, liquid_y), (0, 255, 255), 2)
                    if level_cm is not None:
                        cv2.putText(frame, f"Height: {level_cm:.2f}cm", (30, 40),
                                    cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Liquidlevel is not detected.", (30, 40),
                                cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tank is not detected.", (30, 40),
                            cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            shared.update(level_cm)

            if SHOW_WINDOW:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1)
                if key & 0xFF == 27:
                    stop_event.set()
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break

            t1 = time.perf_counter()
            delay_ms = (t1 - t0) * 1000.0
            if level_cm is not None:
                print(f"level={level_cm:.2f}cm | infer={infer_ms:.1f}ms | frame={delay_ms:.1f}ms", end="\r")
            else:
                print(f"level=None | infer={infer_ms:.1f}ms | frame={delay_ms:.1f}ms", end="\r")

    finally:
        camera.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


def control_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    pi = PIController(Kp, Ki)
    valve = ValveController(board_type="minima", inlet_pin=7, outlet_pin=5)

    # 센싱 데이터가 오래되면 안전 OFF
    STALE_SEC = 3.0

    next_tick = time.time()

    try:
        while not stop_event.is_set():
            # 2초 주기 고정
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            # 이번 tick 시작
            tick_start = time.time()

            level_cm, ts, valid = shared.get()
            age = time.time() - ts

            if (not valid) or (level_cm is None) or (age > STALE_SEC):
                valve.all_off()
                pi.reset()
                # 다음 tick 예약
                next_tick = tick_start + CONTROL_PERIOD
                continue

            # dt는 고정(=CONTROL_PERIOD)로 사용
            u = pi.update(SETPOINT_CM, level_cm, dt=CONTROL_PERIOD)

            # 이번 2초 동안 듀티로 밸브 구동
            valve.drive_for_period(u, CONTROL_PERIOD, stop_event)

            # 다음 tick 예약 (드리프트 최소화)
            next_tick = tick_start + CONTROL_PERIOD

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
