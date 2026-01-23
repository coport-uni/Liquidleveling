"""
=============================================================================
DQN 모델 배포 - 실제 수위 제어 시스템 (u=5..15, 11 actions, no time)
=============================================================================
- 학습 코드와 호환:
  * STATE_DIM = 5  -> build_state(h) 사용
  * ACTION_DIM = 11 -> action(0..10) -> u = ACT_U_MIN + action
  * time state 없음
  * Δu 기반 아님 (action_to_delta_u 사용 안 함)
=============================================================================
"""

import time
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PyArduino import PyArduino
import matplotlib.pyplot as plt
from collections import deque

# ✅ 반드시 "학습 때 사용한 코드"와 동일한 DQN_control을 import 해야 함
from DQN_control import (
    DQNAgent,
    # 학습코드와 동일 상수들
    TANK_HEIGHT_CM, SETPOINT_CM, CONTROL_PERIOD_S,
    PHYS_MIN_PUMP, PHYS_MAX_PUMP,
    ACT_U_MIN, ACT_U_MAX, ACTION_DIM, STATE_DIM,
    SAFETY_H_MAX, SAFETY_H_MIN,
    CAMERA_INDEX, MODEL_PATH,
    LIQUID_CLASS_ID, TANK_CLASS_ID,
    SHOW_DISPLAY
)

# =============================================================================
# 배포 설정
# =============================================================================
DEPLOYMENT_PHASE = 1  # 1: 파인튜닝, 2: 검증, 3: 최종 배포

EPSILON_BY_PHASE = {
    1: 0.2,
    2: 0.05,
    3: 0.0
}

ENABLE_LEARNING = {
    1: True,
    2: False,
    3: False
}

# ✅ 네 학습 코드 저장 파일명과 일치
MODEL_PATH_DQN = "dqn_water_level_model_u11_notime.pth"


# =============================================================================
# 데이터 공유 클래스
# =============================================================================

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


# =============================================================================
# 펌프 제어
# =============================================================================

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
        speed = int(np.clip(speed, PHYS_MIN_PUMP, PHYS_MAX_PUMP))
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        self.set_pump_speed(0)
        self.close_all_valves()


# =============================================================================
# 영상 처리
# =============================================================================

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

        if class_id == TANK_CLASS_ID and conf > best_tank_conf:
            best_tank_conf = conf
            tank_box = (x1, y1, x2, y2)
        elif class_id == LIQUID_CLASS_ID and conf > best_liquid_conf:
            best_liquid_conf = conf
            liquid_line = y1

    return tank_box, liquid_line


def calculate_liquid_level_cm(liquid_line, tank_box):
    _, tank_top_y, _, tank_bottom_y = tank_box
    if tank_bottom_y <= tank_top_y:
        return None

    liquid_height_cm = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * TANK_HEIGHT_CM
    liquid_height_cm = max(0.0, min(TANK_HEIGHT_CM, liquid_height_cm))
    return liquid_height_cm


# =============================================================================
# 스레드 함수
# =============================================================================

def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    model = YOLO(MODEL_PATH)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "DQN Water Level Control (u=5..15)"

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
                    cv2.putText(frame, f"Height: {liquid_height_cm:.2f}cm",
                                (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Liquid not detected",
                                (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tank not detected",
                            (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            shared.update(liquid_height_cm)

            if SHOW_DISPLAY:
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
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()


def control_thread_fn(shared: SharedLevel, pump: PumpController,
                     stop_event: threading.Event, log: SharedLog,
                     agent: DQNAgent):
    """
    ✅ 네 학습 코드와 동일한 방식으로 제어:
    - state = agent.build_state(h)
    - action = agent.select_action(state, training=enable_learning)
    - u_cmd = agent.action_to_u(action)  (u=5..15)
    - reward/온라인학습도 agent.buffer/train_step/update_target 사용
    """
    epsilon = EPSILON_BY_PHASE[DEPLOYMENT_PHASE]
    enable_learning = ENABLE_LEARNING[DEPLOYMENT_PHASE]

    print(f"\n[Phase {DEPLOYMENT_PHASE}] epsilon={epsilon}, learning={enable_learning}")
    print(f"Action: {ACTION_DIM}개, u ∈ [{ACT_U_MIN}..{ACT_U_MAX}] (정수)")
    print(f"State dim: {STATE_DIM} (no time)\n")

    agent.epsilon = epsilon

    # 내부 상태 초기화: 첫 유효 측정이 들어오기 전까지 u=0 유지
    agent.h_prev = SETPOINT_CM
    agent.u_prev = float((ACT_U_MIN + ACT_U_MAX) // 2)  # 10
    agent.error_int = 0.0

    prev_state = None
    prev_action = None
    prev_reward = None

    next_tick = time.time()
    step_count = 0
    STALE_SEC = 3.0

    recent_q_values = deque(maxlen=20)

    try:
        while not stop_event.is_set():
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            tick_start = time.time()

            # 최신 수위 읽기
            h, ts, valid = shared.get()
            age = time.time() - ts

            # 데이터 유효성 검사
            if (not valid) or (h is None) or (age > STALE_SEC):
                pump.set_pump_speed(0)
                log.add(time.time(), h, 0, SETPOINT_CM, 0.0, -1, 0.0)

                # 온라인 학습 시에도 끊긴 구간은 학습 데이터로 넣지 않는게 안전
                prev_state, prev_action, prev_reward = None, None, None

                next_tick = tick_start + CONTROL_PERIOD_S
                continue

            # ===== 1) 상태 구성 (학습과 동일) =====
            state = agent.build_state(h)

            # ===== 2) 액션 선택 =====
            action = agent.select_action(state, training=enable_learning)

            # ===== 3) action -> 절대 u (5..15) =====
            u_cmd = int(agent.action_to_u(action))

            # ===== Q값 모니터링 =====
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(st)[0]
                q_selected = float(q_values[action].item())
                recent_q_values.append(q_selected)

            # ===== 4) 안전 체크(추가 보호) =====
            if h > SAFETY_H_MAX - 0.5:
                print(f"\n⚠️ 수위 상한 근접: {h:.2f}cm -> 펌프 정지")
                u_cmd = 0
            elif h < SAFETY_H_MIN + 0.5:
                print(f"\n⚠️ 수위 하한 근접: {h:.2f}cm")

            # ===== 5) 펌프 제어 =====
            pump.set_pump_speed(u_cmd)

            # ===== 6) 보상 계산 (학습과 동일) =====
            reward = agent.compute_reward(h, u_cmd)

            # ===== 7) 적분 업데이트(스텝당 1회, 학습과 동일) =====
            agent.error_int = float(np.clip(
                agent.error_int + (SETPOINT_CM - h) * CONTROL_PERIOD_S,
                -10.0, 10.0
            ))

            # ===== 8) 온라인 학습 (Phase 1만) =====
            if enable_learning and prev_state is not None:
                # done은 실제 시스템에선 명확히 정의하기 어려워서 일단 False(계속학습)로 둠
                agent.buffer.push(prev_state, prev_action, prev_reward, state, 0.0)

                if len(agent.buffer) >= 1000:
                    loss = agent.train_step()
                    if (step_count % 10) == 0:
                        agent.update_target()

            prev_state, prev_action, prev_reward = state, action, reward

            # ===== 9) 로깅 =====
            avg_q = float(np.mean(recent_q_values)) if recent_q_values else 0.0
            log.add(time.time(), h, u_cmd, SETPOINT_CM, reward, action, q_selected)

            print(f"\n[DQN] h={h:.2f}cm | action={action} -> u={u_cmd} | "
                  f"Q={q_selected:.2f} (avg {avg_q:.2f}) | reward={reward:.2f}")

            # ===== 10) 다음 주기 준비 =====
            agent.h_prev = h
            agent.u_prev = float(u_cmd)
            step_count += 1
            next_tick = tick_start + CONTROL_PERIOD_S

    finally:
        print("\n" + "=" * 70)
        print("제어 종료")
        print("=" * 70)

        # Phase 1: 파인튜닝한 모델 저장
        if enable_learning:
            save_path = f"dqn_finetuned_u11_phase{DEPLOYMENT_PHASE}.pth"
            agent.save(save_path)

        pump.shutdown()


# =============================================================================
# 결과 시각화
# =============================================================================

def plot_results(log: SharedLog):
    t, level, speed, sp, reward, action, q_value = log.snapshot()
    if len(t) < 2:
        print("그래프 데이터 부족")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # None 제거
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    ax1.plot(t_level, level_valid, linewidth=2, label="Level")
    ax1.axhline(SETPOINT_CM, linestyle='--', linewidth=2, label="Setpoint")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Level (cm)")
    ax1.set_title("Water Level Control (DQN)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(t_rel, speed, linewidth=2, label="Pump Speed")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed")
    ax2.set_title("Control Input")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(t_rel, reward, linewidth=2, label="Reward")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Instantaneous Reward")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(t_rel, q_value, linewidth=2, label="Q-value")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Q-value")
    ax4.set_title("Selected Action Q-value")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 메인
# =============================================================================

def main():
    print("=" * 70)
    print(f"DQN 배포 - Phase {DEPLOYMENT_PHASE}")
    print("=" * 70)
    print(f"Epsilon: {EPSILON_BY_PHASE[DEPLOYMENT_PHASE]}")
    print(f"Learning: {ENABLE_LEARNING[DEPLOYMENT_PHASE]}")
    print("=" * 70)

    agent = DQNAgent()

    # ✅ 모델 로드
    try:
        agent.load(MODEL_PATH_DQN)
        print(f"✅ 모델 로드 성공: {MODEL_PATH_DQN}")
    except Exception as e:
        print(f"⚠️ 모델 로드 실패: {MODEL_PATH_DQN}")
        print("에러:", e)
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
        print("\n종료 신호 (Ctrl+C)")
        stop_event.set()

    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)

    print("\n프로그램 종료")
    plot_results(log)


if __name__ == "__main__":
    main()
