import time
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from py_arduino import PyArduino
import matplotlib.pyplot as plt

# 학습 때 사용한 DQN_training과 동일해야 함
from DQN_training import (
    DQNAgent, tank_height_cm, setpoint_cm, control_period_s,
    min_pump_speed, max_pump_speed, state_dims, action_dims
)

epsilon = 0.05

model_path_dqn = "dqn_liquid_level_model.pth"

camera_index = 1
model_path = "20251223nano.pt"
liquid_class_id = 0
tank_class_id = 1
show_display = True


# 데이터 공유 class

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


class SharedLog:
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
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)
            self.reward.append(reward)
            self.action.append(action)
            self.q_value.append(q_value)

    def snapshot(self):
        with self.lock:
            return (self.t[:], self.level[:], self.speed[:],
                    self.setpoint[:], self.reward[:], self.action[:],
                    self.q_value[:])


# 펌프 제어

class PumpController:
    def __init__(self, board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5):
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin

        self.open_all_valves()
        self.set_pump_speed(0)

    def open_all_valves(self):
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print("모든 밸브 열림")

    def close_all_valves(self):
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\n모든 밸브 닫힘")

    def set_pump_speed(self, speed: int):
        speed = int(np.clip(speed, min_pump_speed, max_pump_speed))
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        self.set_pump_speed(0)
        self.close_all_valves()


def detect_tank_and_liquid(frame, model):
    results = model(frame, conf=0.9, verbose=False)[0]

    tank_box = None
    best_tank_conf = -1.0
    liquid_line = None
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
            liquid_line = y1

    return tank_box, liquid_line


def calculate_liquid_level_cm(liquid_line, tank_box):
    _, tank_top_y, _, tank_bottom_y = tank_box
    if tank_bottom_y <= tank_top_y:
        return None

    liquid_height_cm = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * tank_height_cm
    liquid_height_cm = max(0.0, min(tank_height_cm, liquid_height_cm))
    return liquid_height_cm


def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    model = YOLO(model_path)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "DQN liquid level control"

    try:
        while not stop_event.is_set():
            ret, frame = camera.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                stop_event.set()
                break

            tank_box, liquid_line = detect_tank_and_liquid(frame, model)

            liquid_height_cm = None
            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if liquid_line is not None:
                    liquid_height_cm = calculate_liquid_level_cm(liquid_line, tank_box)
                    cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)
                    cv2.putText(frame, f"Height: {liquid_height_cm:.2f}cm", (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Liquid not detected", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tank not detected", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            shared.update(liquid_height_cm)

            if show_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10)
                if key & 0xFF == 27:
                    stop_event.set()
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break

    finally:
        camera.release()
        if show_display:
            cv2.destroyAllWindows()


def control_thread_fn(shared:SharedLevel, pump:PumpController, stop_event:threading.Event, log:SharedLog, agent:DQNAgent):

    print(f"\n[Validation mode]")
    print(f"action: [{min_pump_speed} - {max_pump_speed}] (정수)")
    print(f"state dim: {state_dims}\n")

    agent.epsilon = float(epsilon)

    # 내부 상태 초기화
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

            if (not valid) or (h is None) or (age > stale_sec):
                pump.set_pump_speed(0)
                agent.u_prev = 0.0
                log.add(time.time(), h, 0, setpoint_cm, 0.0, -1, 0.0)
                
                next_tick = tick_start + control_period_s
                continue

            # 적분 업데이트
            agent.error_int = float(np.clip(agent.error_int + (setpoint_cm - h) * control_period_s, -10.0, 10.0))
            
            # 상태 구성
            state = agent.build_state(h)

            # 액션 선택
            action = agent.select_action(state, training=False)

            # action -> 절대 u
            u_cmd = int(agent.action_to_u(action))

            # q 모니터링
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = agent.qNet(st)[0]
                q_selected = float(q_values[action].item())

            # 펌프 제어
            pump.set_pump_speed(u_cmd)

            # 보상 계산 (학습과 동일한 reward 정의가 중요)
            reward = agent.compute_reward(h, agent.u_prev)
            agent.u_prev = float(u_cmd)

            log.add(time.time(), h, u_cmd, setpoint_cm, reward, action, q_selected)

            print(f"\n[h={h:.2f}cm / action={action} -> u={u_cmd}] / "
                  f"[q={q_selected:.2f} / reward={reward:.2f}]")

            next_tick = tick_start + control_period_s

    finally:
        print("제어 종료")
        pump.shutdown()


# 결과 plot

def plot_results(log:SharedLog):
    t, level, speed, sp, reward, action, q_value = log.snapshot()
    if len(t) < 2:
        print("그래프 데이터 부족")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    ax1.plot(t_level, level_valid, linewidth=2, label="level")
    ax1.axhline(setpoint_cm, linestyle="--", linewidth=2, label="setpoint")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("level (cm)")
    ax1.set_title("liquid level control (DQN)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_rel, speed, linewidth=2, label="pump speed")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed")
    ax2.set_title("control input")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t_rel, reward, linewidth=2, label="reward")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("reward")
    ax3.set_title("reward")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(t_rel, q_value, linewidth=2, label="q-value")
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("q-value")
    ax4.set_title("selected action q-value")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def main():
    print(f"DQN - Validation mode")

    agent = DQNAgent(state_dims, action_dims)

    # 모델 로드
    try:
        agent.load(model_path_dqn)
        print(f"Model loaded: {model_path_dqn}")
    except Exception as e:
        print(f"Model load failed: {model_path_dqn}")
        print("error:", e)
        return

    shared = SharedLevel()
    log = SharedLog()
    stop_event = threading.Event()
    pump = PumpController(board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5)

    t_sense = threading.Thread(target=sensing_thread_fn, args=(shared, stop_event), daemon=True)
    t_ctrl = threading.Thread(target=control_thread_fn, args=(shared, pump, stop_event, log, agent), daemon=True)

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)

    print("\n프로그램 종료")
    plot_results(log)


if __name__ == "__main__":
    main()
