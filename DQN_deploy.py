"""
=============================================================================
DQN 모델 배포 - 실제 수위 제어 시스템
=============================================================================
시뮬레이터에서 학습한 DQN 모델을 실제 시스템에 배포

⚠️ 배포 전 체크리스트:
1. ✅ 시뮬레이터에서 충분히 학습됨 (1000+ 에피소드)
2. ✅ 학습 곡선이 수렴함
3. ✅ 시뮬레이터 테스트 성공
4. ✅ 안전 제약 확인
5. ✅ 비상 정지 버튼 준비

배포 단계:
- Phase 1: epsilon=0.2로 파인튜닝 (안전한 탐험)
- Phase 2: epsilon=0.05로 성능 검증
- Phase 3: epsilon=0.0으로 배포

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

# DQN 관련 임포트 (위에서 정의한 클래스들)
from DQN_control import (
    DQNAgent, DQN, 
    STATE_DIM, ACTION_DIM,
    CAMERA_INDEX, MODEL_PATH,
    LIQUID_CLASS_ID, TANK_CLASS_ID, TANK_HEIGHT_CM,
    SETPOINT_CM, CONTROL_PERIOD_S,
    MAX_PUMP_SPEED, MIN_PUMP_SPEED,
    SAFETY_H_MAX, SAFETY_H_MIN,
    SHOW_DISPLAY
)


# =============================================================================
# 배포 설정
# =============================================================================

# 배포 모드 (Phase 1, 2, 3)
DEPLOYMENT_PHASE = 1  # 1: 파인튜닝, 2: 검증, 3: 최종 배포

# Phase별 epsilon
EPSILON_BY_PHASE = {
    1: 0.2,   # Phase 1: 파인튜닝 (적당한 탐험)
    2: 0.05,  # Phase 2: 검증 (최소 탐험)
    3: 0.0    # Phase 3: 배포 (탐험 없음)
}

# 학습 여부
ENABLE_LEARNING = {
    1: True,   # Phase 1: 온라인 학습
    2: False,  # Phase 2: 학습 중단
    3: False   # Phase 3: 학습 중단
}

# 모델 경로
MODEL_PATH_DQN = "dqn_water_level_model.pth"


# =============================================================================
# 데이터 공유 클래스
# =============================================================================

class SharedLevel:
    """스레드 간 수위 데이터 공유"""
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
    """실험 데이터 기록"""
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
    """Arduino 펌프 제어"""
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
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        self.set_pump_speed(0)
        self.close_all_valves()


# =============================================================================
# 영상 처리
# =============================================================================

def detect_tank_and_liquid(frame, model):
    """YOLO로 탱크와 수위선 탐지"""
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

        if class_id == TANK_CLASS_ID and conf > best_tank_confidence:
            best_tank_confidence = conf
            tank_box = (x1, y1, x2, y2)
        elif class_id == LIQUID_CLASS_ID and conf > best_liquid_confidence:
            best_liquid_confidence = conf
            liquid_line = y1

    return tank_box, liquid_line, inference_time_ms


def calculate_liquid_level_cm(liquid_line, tank_box):
    """픽셀 → cm 변환"""
    _, tank_top_y, _, tank_bottom_y = tank_box
    
    if tank_bottom_y <= tank_top_y:
        return None
    
    liquid_height_cm = float(tank_bottom_y - liquid_line) / \
                      float(tank_bottom_y - tank_top_y) * TANK_HEIGHT_CM
    liquid_height_cm = max(0.0, min(TANK_HEIGHT_CM, liquid_height_cm))
    
    return liquid_height_cm


# =============================================================================
# 스레드 함수
# =============================================================================

def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    """센싱 스레드"""
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    model = YOLO(MODEL_PATH)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "DQN Water Level Control"
    
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
                    liquid_height_cm = calculate_liquid_level_cm(liquid_line, tank_box)
                    cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)
                    cv2.putText(frame, f"Height: {liquid_height_cm:.2f}cm", 
                               (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Phase {DEPLOYMENT_PHASE}", 
                               (30, 80), cv2.FONT_ITALIC, 1, (255, 255, 0), 2)
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
                
                if key & 0xFF == 27:  # ESC
                    print("\n프로그램 종료 중")
                    stop_event.set()
                    break
                
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n프로그램 종료 중")
                    stop_event.set()
                    break

            frame_time = (time.perf_counter() - inference_start_time) * 1000
            if liquid_height_cm is not None:
                print(f"Level: {liquid_height_cm:.2f}cm, "
                      f"Inference: {inference_time_ms:.1f}ms", end="\r")

    finally:
        camera.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()


def control_thread_fn(shared: SharedLevel, pump: PumpController,
                     stop_event: threading.Event, log: SharedLog, 
                     agent: DQNAgent):
    """
    DQN 제어 스레드
    
    작동:
    1. 상태 읽기
    2. DQN으로 액션 선택
    3. 펌프 제어
    4. (선택) 온라인 학습
    5. 로깅
    """
    epsilon = EPSILON_BY_PHASE[DEPLOYMENT_PHASE]
    enable_learning = ENABLE_LEARNING[DEPLOYMENT_PHASE]
    
    print(f"\n[Phase {DEPLOYMENT_PHASE}] epsilon={epsilon}, learning={enable_learning}\n")
    
    # 에피소드 초기화
    agent.reset_episode()
    agent.epsilon = epsilon  # 배포용 epsilon 설정
    
    next_tick = time.time()
    step_count = 0
    STALE_SEC = 3.0
    
    # 최근 Q값 추적 (디버깅용)
    recent_q_values = deque(maxlen=20)

    try:
        while not stop_event.is_set():
            now = time.time()
            
            # 제어 주기 대기
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            tick_start = time.time()

            # 최신 수위 읽기
            liquid_height_cm, ts, valid = shared.get()
            age = time.time() - ts

            # 데이터 유효성 검사
            if (not valid) or (liquid_height_cm is None) or (age > STALE_SEC):
                pump.set_pump_speed(0)
                log.add(time.time(), liquid_height_cm, 0, SETPOINT_CM, 
                       0.0, 2, 0.0)  # action=2 (Δu=0)
                agent.reset_episode()
                next_tick = tick_start + CONTROL_PERIOD_S
                continue

            # ==== 1) 상태 구성 ====
            state = agent.get_state(liquid_height_cm)
            
            # ==== 2) 액션 선택 ====
            action = agent.select_action(state, training=enable_learning)
            delta_u = agent.action_to_delta_u(action)
            u_cmd = int(np.clip(agent.u_prev + delta_u, MIN_PUMP_SPEED, MAX_PUMP_SPEED))
            
            # Q값 확인 (모니터링용)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_value_selected = q_values[0][action].item()
                recent_q_values.append(q_value_selected)
            
            # ==== 3) 안전 체크 ====
            if liquid_height_cm > SAFETY_H_MAX - 0.5:
                print(f"\n⚠️ 수위 경고: {liquid_height_cm:.2f}cm (상한 근접)")
                u_cmd = 0  # 안전을 위해 펌프 정지
            elif liquid_height_cm < SAFETY_H_MIN + 0.5:
                print(f"\n⚠️ 수위 경고: {liquid_height_cm:.2f}cm (하한 근접)")
            
            # ==== 4) 펌프 제어 ====
            pump.set_pump_speed(u_cmd)
            
            # ==== 5) 보상 계산 ====
            reward = agent.compute_reward(liquid_height_cm, u_cmd)
            
            # ==== 6) 온라인 학습 (Phase 1만) ====
            if enable_learning and step_count > 0:
                # 이전 transition 저장
                next_state = state
                agent.replay_buffer.push(
                    agent.prev_state, agent.prev_action, 
                    agent.prev_reward, next_state, False
                )
                
                # 학습
                if len(agent.replay_buffer) >= 1000:
                    loss = agent.train_step()
                    
                    # 주기적으로 타겟 네트워크 업데이트
                    if step_count % 10 == 0:
                        agent.update_target_network()
            
            # 다음 스텝 준비
            agent.prev_state = state
            agent.prev_action = action
            agent.prev_reward = reward
            
            # ==== 7) 로깅 ====
            log.add(time.time(), liquid_height_cm, u_cmd, SETPOINT_CM,
                   reward, action, q_value_selected)
            
            # ==== 8) 제어 정보 출력 ====
            avg_q = np.mean(recent_q_values) if recent_q_values else 0.0
            print(f"\n[DQN] h={liquid_height_cm:.2f}cm, "
                  f"action={action} (Δu={delta_u:+d}), "
                  f"u={u_cmd}, "
                  f"Q={q_value_selected:.2f}, "
                  f"Q_avg={avg_q:.2f}, "
                  f"reward={reward:.2f}")
            
            # ==== 9) 다음 제어 주기 ====
            agent.h_prev = liquid_height_cm
            agent.u_prev = float(u_cmd)
            step_count += 1
            next_tick = tick_start + CONTROL_PERIOD_S

    finally:
        print("\n" + "="*70)
        print("제어 종료")
        print("="*70)
        
        # Phase 1에서는 학습된 모델 저장
        if enable_learning:
            save_path = f"dqn_finetuned_phase{DEPLOYMENT_PHASE}.pth"
            agent.save(save_path)
        
        pump.shutdown()


# =============================================================================
# 결과 시각화
# =============================================================================

def plot_results(log: SharedLog):
    """실험 결과 그래프"""
    t, level, speed, sp, reward, action, q_value = log.snapshot()
    
    if len(t) < 2:
        print("그래프 데이터 부족")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # None 제거
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    # 4개 서브플롯
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1) 수위
    ax1.plot(t_level, level_valid, 'b-', linewidth=2, label="Level")
    ax1.axhline(SETPOINT_CM, color='r', linestyle='--', linewidth=2, label="Setpoint")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Level (cm)")
    ax1.set_title("Water Level Control (DQN)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2) 펌프 속도
    ax2.plot(t_rel, speed, 'g-', linewidth=2, label="Pump Speed")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (%)")
    ax2.set_title("Control Input")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3) 보상
    ax3.plot(t_rel, reward, 'm-', linewidth=2, label="Reward")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Instantaneous Reward")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4) Q값
    ax4.plot(t_rel, q_value, 'c-', linewidth=2, label="Q-value")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Q-value")
    ax4.set_title("Selected Action Q-value")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    """메인 배포 함수"""
    print("=" * 70)
    print(f"DQN 배포 - Phase {DEPLOYMENT_PHASE}")
    print("=" * 70)
    print(f"Epsilon: {EPSILON_BY_PHASE[DEPLOYMENT_PHASE]}")
    print(f"Learning: {ENABLE_LEARNING[DEPLOYMENT_PHASE]}")
    print("=" * 70)
    
    # DQN 에이전트 초기화
    agent = DQNAgent()
    
    # 학습된 모델 로드
    try:
        agent.load(MODEL_PATH_DQN)
        print(f"✅ 모델 로드 성공: {MODEL_PATH_DQN}")
    except:
        print(f"⚠️ 모델 로드 실패: {MODEL_PATH_DQN}")
        print("시뮬레이터에서 먼저 학습하세요!")
        return
    
    # 초기화
    shared = SharedLevel()
    log = SharedLog()
    stop_event = threading.Event()
    pump = PumpController(board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5)

    # 스레드 생성
    t_sense = threading.Thread(
        target=sensing_thread_fn,
        args=(shared, stop_event),
        daemon=True
    )
    t_ctrl = threading.Thread(
        target=control_thread_fn,
        args=(shared, pump, stop_event, log, agent),
        daemon=True
    )

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
    
    # 결과 시각화
    plot_results(log)


if __name__ == "__main__":
    main()
