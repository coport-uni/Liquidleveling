import time
import threading
import cv2
from ultralytics import YOLO
from PyArduino import PyArduino
import matplotlib.pyplot as plt

# 주석, class 정리 필요 (변수명도)
camera_index = 1

model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1
tank_height_cm = 8.0

setpoint_cm = 4.00

show_display = True

# PI gains
Kp = 4.0
Ki = 0.005  # 적분 이득 (조정 필요)

# 정상상태 펌프 속도 (수위 유지에 필요한 기본 펌프 속도)
state_steady_speed = 10

level_tolerance_cm = 0.05

# 제어 주기 고정
control_period_s = 3.0

# 펌프 속도 범위
max_pump_speed = 20
min_pump_speed = 0

def clamp(value, lower, upper):
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value

class sharedlevel:
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


# log 저장용
class sharedlog:
    def __init__(self):
        self.lock = threading.Lock()
        self.t = []
        self.level = []
        self.speed = []
        self.setpoint = []
        self.error_i = []  # 적분 항 기록용

    def add(self, t, level, speed, setpoint, error_i=0.0):
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)
            self.error_i.append(error_i)

    def snapshot(self):
        with self.lock:
            return (self.t[:], self.level[:], self.speed[:], self.setpoint[:], self.error_i[:])

# plot
def plot_results(log: sharedlog):
    t, level, speed, sp, error_i = log.snapshot()
    if len(t) < 2:
        print("plot할 데이터가 충분하지 않습니다.")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # None 값 제거 (수위 그래프용)
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    # (1) state 그래프: 수위 + setpoint
    plt.figure()
    plt.plot(t_level, level_valid, label="Liquid level (cm)")
    plt.axhline(setpoint_cm, linestyle="--", label="Setpoint (cm)")
    plt.xlabel("Time (s)")
    plt.ylabel("Level (cm)")
    plt.title("Liquid level")
    plt.grid(True)
    plt.legend()

    # (2) Input 그래프: Pump speed
    plt.figure()
    plt.plot(t_rel, speed, label="Pump speed (%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pump speed (%)")
    plt.title("Pump speed")
    plt.grid(True)
    plt.legend()

    # (3) 적분 항 그래프
    plt.figure()
    plt.plot(t_rel, error_i, label="Integral error")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral error")
    plt.title("Integral error")
    plt.grid(True)
    plt.legend()

    plt.show()


class PI_controller:
    """
    PI 제어 + 정상상태
    출력: state_steady_speed + Kp * error + Ki * integral_error
    """
    def __init__(self, kp, ki, steady_state_speed, max_speed, dt):
        self.kp = kp
        self.ki = ki
        self.steady_state_speed = steady_state_speed
        self.max_speed = max_speed
        self.dt = dt  # 제어 주기
        
        # 적분 항 초기화
        self.integral_error = 0.0
    
    def update(self, setpoint, measurement):
        error = setpoint - measurement
        
        # Dead-band 적용
        if abs(error) < level_tolerance_cm:
            error = 0.0
        
        # 적분 항 업데이트
        self.integral_error += error * self.dt
        
        # PI 제어 + 정상상태 유량
        speed_raw = self.steady_state_speed + self.kp * error + self.ki * self.integral_error
        
        # Anti-windup: Clamping 방식
        speed = clamp(speed_raw, min_pump_speed, self.max_speed)
        
        # 출력이 포화되면 적분 항 조정 (back-calculation)
        if speed_raw != speed:
            # 포화 발생 시 적분 항을 역계산하여 조정
            self.integral_error = (speed - self.steady_state_speed - self.kp * error) / self.ki if self.ki != 0 else 0.0
        
        speed = int(speed)
        
        return speed, self.integral_error
    
    def reset(self):
        """적분 항 리셋 (필요시 사용)"""
        self.integral_error = 0.0


class pump_controller:
    """
    Inlet 펌프 제어 + 밸브 제어
    """
    def __init__(self, board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5):
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin
        
        # 시작 시 모든 밸브 열기
        self.open_all_valves()
        
        # 펌프는 정지 상태로 시작
        self.set_pump_speed(0)

    def open_all_valves(self):
        # 시작 시 모든 valve 열기
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print("모든 밸브 열림 (Inlet: Pin 7, Outlet: Pin 5)")

    def close_all_valves(self):
        # 종료 시 모든 valve 닫기
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\n모든 밸브 닫힘")

    def set_pump_speed(self, speed: int):
        # 펌프 속도 설정 (0~20)
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        # 종료: 펌프 정지 & 밸브 닫기
        self.set_pump_speed(0)
        self.close_all_valves()


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


def sensing_thread_fn(shared: sharedlevel, stop_event: threading.Event):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                
                # ESC 키를 누르면 종료
                if key & 0xFF == 27:
                    print("\n프로그램 종료 중")
                    stop_event.set()
                    break
                
                # 창을 닫으면 종료
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n프로그램 종료 중")
                    stop_event.set()
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


# log 인자 추가
def control_thread_fn(shared: sharedlevel, pump: pump_controller, stop_event: threading.Event, log: sharedlog):
    pi_controller = PI_controller(Kp, Ki, state_steady_speed, max_pump_speed, control_period_s)

    # 센싱 데이터가 오래되면 안전 정지
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
                pump.set_pump_speed(0)
                pi_controller.reset()  # 적분 항 리셋
                log.add(time.time(), liquid_height_cm, 0, setpoint_cm, 0.0)
                next_tick = tick_start + control_period_s
                continue

            # PI 제어 계산
            speed, integral_error = pi_controller.update(setpoint_cm, liquid_height_cm)

            # 펌프 제어
            pump.set_pump_speed(speed)

            # 기록
            log.add(time.time(), liquid_height_cm, speed, setpoint_cm, integral_error)

            # 다음 tick 예약
            next_tick = tick_start + control_period_s

    finally:
        pump.shutdown()  # 종료 시 펌프 정지 & 밸브 닫기


def main():
    shared = sharedlevel()
    log = sharedlog()
    stop_event = threading.Event()
    pump = pump_controller(board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5)

    t_sense = threading.Thread(target=sensing_thread_fn, args=(shared, stop_event), daemon=True)
    t_ctrl = threading.Thread(target=control_thread_fn, args=(shared, pump, stop_event, log), daemon=True)

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 신호 감지")
        stop_event.set()

    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)
    
    print("종료 완료")

    plot_results(log)

if __name__ == "__main__":
    main()