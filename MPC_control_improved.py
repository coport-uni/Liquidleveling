"""
=============================================================================
수위 제어 시스템 - 적응형 MPC (Model Predictive Control)
=============================================================================
온라인 모델 식별(RLS)과 MPC를 결합한 적응형 수위 제어 시스템

주요 기능:
- YOLO 객체 탐지를 통한 실시간 수위 측정
- RLS(Recursive Least Squares)를 통한 온라인 시스템 식별
- MPC 기반 최적 제어 (추적 + 평활 + Bias 유지)
- 멀티스레딩 기반 센싱/제어 분리
- 실험 결과 시각화 (수위, 펌프 속도, 목적함수)

MPC vs P 제어 비교:
- 장점: 미래 예측 기반 최적 제어, 제약 조건 체계적 처리, 모델 적응
- 단점: 계산 부담 증가, 튜닝 파라미터 많음, 구현 복잡도 높음
=============================================================================
"""

import time
import threading
import cv2
import numpy as np
from scipy.optimize import minimize
from ultralytics import YOLO
from py_arduino import PyArduino
import matplotlib.pyplot as plt


# =============================================================================
# 1. 시스템 설정 및 상수 정의
# =============================================================================

# --- 카메라 및 YOLO 설정 ---
CAMERA_INDEX = 1
MODEL_PATH = "20251223nano.pt"
LIQUID_CLASS_ID = 0
TANK_CLASS_ID = 1

# --- 물리적 탱크 사양 ---
TANK_HEIGHT_CM = 8.0

# --- 제어 목표 ---
SETPOINT_CM = 3.85

# --- 디스플레이 설정 ---
SHOW_DISPLAY = True

# --- 제어 주기 ---
CONTROL_PERIOD_S = 3.0  # 3초마다 MPC 최적화 실행

# --- 펌프 속도 제한 ---
MAX_PUMP_SPEED = 20
MIN_PUMP_SPEED = 0

# --- MPC 파라미터 ---
PREDICTION_HORIZON = 5  # 예측 구간: 5 스텝 (15초)
CONTROL_HORIZON = 3     # 제어 구간: 3 스텝 (9초) - 현재 미사용, 향후 개선 가능

# MPC 목적함수 가중치
Q_WEIGHT = 10.0   # 추적 성능 가중치 (높을수록 목표값 추종 강화)
R_WEIGHT = 0.2    # 입력 평활화 가중치 (높을수록 급격한 입력 변화 억제)
S_WEIGHT = 0.05   # Bias 유지 가중치 (높을수록 정상상태 입력 근처 유지)

# --- 정상상태 설정 ---
U_SS = 10.0  # 정상상태 펌프 속도 (수위 유지에 필요한 기본 유량)
LEVEL_TOLERANCE_CM = 0.05  # Dead-zone: 이 범위 내 오차는 0으로 처리

# --- RLS 파라미터 ---
RLS_INITIAL_A = 1.0      # 초기 모델: a (1차 시스템 가정)
RLS_INITIAL_B = 0.01     # 초기 모델: b (게인)
RLS_INITIAL_C = -0.1     # 초기 모델: c (바이어스, 유출 효과)
RLS_INITIAL_P = 2000.0   # 초기 공분산 (큰 값 = 빠른 초기 학습)
RLS_FORGETTING_FACTOR = 0.985  # 망각 인자 (약 67샘플 기억, 1/(1-λ))

# --- 센싱 타임아웃 ---
STALE_DATA_TIMEOUT_S = 3.0


# =============================================================================
# 2. 유틸리티 함수
# =============================================================================

def clamp(value, lower, upper):
    """값을 지정된 범위로 제한"""
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


# =============================================================================
# 3. 데이터 공유 클래스 (Thread-safe)
# =============================================================================

class SharedLevel:
    """
    스레드 간 수위 데이터 공유
    센싱 스레드 → 제어 스레드로 측정값 전달
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.liquid_height_cm = None
        self.timestamp = 0.0
        self.valid = False

    def update(self, liquid_height_cm):
        """센싱 스레드에서 호출: 새 수위 업데이트"""
        with self.lock:
            self.liquid_height_cm = liquid_height_cm
            self.timestamp = time.time()
            self.valid = liquid_height_cm is not None

    def get(self):
        """제어 스레드에서 호출: 현재 수위 조회"""
        with self.lock:
            return self.liquid_height_cm, self.timestamp, self.valid


class SharedLog:
    """
    실험 데이터 기록
    MPC 추가 정보: 목적함수 값(J*), 최적화 성공 여부
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.t = []           # 시간
        self.level = []       # 수위
        self.speed = []       # 펌프 속도
        self.setpoint = []    # 목표값
        self.cost = []        # MPC 목적함수 J*
        self.cost_ok = []     # 최적화 성공 여부
        self.opt_time = []    # 최적화 계산 시간 (ms)

    def add(self, t, level, speed, setpoint, cost, ok, opt_time_ms=0.0):
        """한 시점의 데이터 기록"""
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)
            self.cost.append(cost)
            self.cost_ok.append(ok)
            self.opt_time.append(opt_time_ms)

    def snapshot(self):
        """현재까지 기록된 모든 데이터 반환"""
        with self.lock:
            return (self.t[:], self.level[:], self.speed[:], 
                   self.setpoint[:], self.cost[:], self.cost_ok[:], 
                   self.opt_time[:])


# =============================================================================
# 4. 온라인 시스템 식별 (RLS)
# =============================================================================

class OnlineRLS:
    """
    재귀 최소자승법 (Recursive Least Squares)
    
    온라인으로 시스템 모델을 식별합니다:
        h(k+1) = a * h(k) + b * u(k) + c
    
    여기서:
        h(k): k 시점의 수위
        u(k): k 시점의 펌프 속도
        a: 시스템 극점 (0.7~1.0, 안정성)
        b: 입력 게인 (펌프 효과)
        c: 바이어스 (유출 효과 등)
    
    Parameters:
        lam: Forgetting factor (0 < λ ≤ 1)
             - λ=1: 모든 과거 데이터 동일 가중치
             - λ<1: 최근 데이터에 더 큰 가중치 (시변 시스템에 유리)
             - 메모리 길이 ≈ 1/(1-λ) 샘플
        P0: 초기 공분산 행렬 크기
            - 큰 값: 빠른 초기 학습, 초기 불확실성 높음
            - 작은 값: 느린 초기 학습, 초기 모델에 대한 신뢰도 높음
    """
    
    def __init__(self, a0=1.0, b0=0.01, c0=-0.1, P0=2000.0, lam=0.985):
        """
        Args:
            a0, b0, c0: 초기 모델 파라미터
            P0: 초기 공분산 크기
            lam: 망각 인자
        """
        self.theta = np.array([a0, b0, c0], dtype=float)  # [a, b, c]
        self.P = np.eye(3, dtype=float) * float(P0)       # 공분산 행렬
        self.lam = float(lam)
        self.n_updates = 0  # 업데이트 횟수

    def reset(self, a0=1.0, b0=0.01, c0=-0.1, P0=2000.0):
        """RLS 상태 초기화 (제어 재시작 시 사용)"""
        self.theta = np.array([a0, b0, c0], dtype=float)
        self.P = np.eye(3, dtype=float) * float(P0)
        self.n_updates = 0

    def update(self, h_k, u_k, h_k1):
        """
        RLS 업데이트: 새 측정값으로 모델 파라미터 갱신
        
        Args:
            h_k: k 시점 수위
            u_k: k 시점 펌프 속도
            h_k1: k+1 시점 수위 (실제 측정값)
        
        Returns:
            (a, b, c): 업데이트된 모델 파라미터
        
        RLS 알고리즘:
            1. 예측: y_hat = φᵀθ
            2. 오차: e = y - y_hat
            3. 게인: K = Pφ / (λ + φᵀPφ)
            4. 업데이트: θ ← θ + Ke
            5. 공분산: P ← (P - KφᵀP) / λ
        """
        # Regressor 벡터: φ = [h(k), u(k), 1]ᵀ
        phi = np.array([h_k, u_k, 1.0], dtype=float).reshape(3, 1)
        
        # RLS 게인 계산
        denom = self.lam + (phi.T @ self.P @ phi).item()
        K = (self.P @ phi) / denom
        
        # 예측 및 오차
        y_hat = (phi.T @ self.theta.reshape(3, 1)).item()
        err = float(h_k1 - y_hat)
        
        # 파라미터 업데이트
        self.theta = self.theta + (K.flatten() * err)
        
        # 공분산 업데이트
        self.P = (self.P - K @ (phi.T @ self.P)) / self.lam
        
        self.n_updates += 1
        
        return float(self.theta[0]), float(self.theta[1]), float(self.theta[2])

    def get_params(self):
        """현재 식별된 모델 파라미터 반환"""
        return float(self.theta[0]), float(self.theta[1]), float(self.theta[2])
    
    def get_confidence(self):
        """
        모델 신뢰도 지표 반환 (공분산 trace)
        작을수록 파라미터 추정이 신뢰성 있음
        """
        return float(np.trace(self.P))


# =============================================================================
# 5. MPC 제어기
# =============================================================================

class MPCController:
    """
    모델 예측 제어기 (Model Predictive Control)
    
    작동 원리:
    1. 현재 상태에서 N 스텝 미래까지 예측
    2. 다음 N개의 제어 입력을 최적화하여 목적함수 최소화
    3. 첫 번째 제어 입력만 적용
    4. 다음 샘플링 시점에 위 과정 반복 (Receding Horizon)
    
    모델:
        h(k+1) = a * h(k) + b * u(k) + c
    
    목적함수:
        J = Q·Σ(h - sp)² + R·Σ(Δu)² + S·Σ(u - u_ss)²
        
        항목별 의미:
        - Q·Σ(h - sp)²: 추적 오차 (목표값 추종)
        - R·Σ(Δu)²: 입력 평활화 (급격한 변화 억제)
        - S·Σ(u - u_ss)²: Bias 유지 (정상상태 입력 근처 유지)
    """
    
    def __init__(self, dt, max_speed, min_speed, prediction_horizon,
                 control_horizon, Q, R, u_init, deadzone, u_ss=10.0, S=0.05):
        """
        Args:
            dt: 샘플링 주기 (초)
            max_speed, min_speed: 펌프 속도 제약
            prediction_horizon (N): 예측 구간 스텝 수
            control_horizon (M): 제어 구간 스텝 수 (현재 미사용)
            Q: 추적 성능 가중치
            R: 입력 평활화 가중치
            S: Bias 유지 가중치
            u_init: 초기 펌프 속도
            deadzone: Dead-zone 폭 (cm)
            u_ss: 정상상태 펌프 속도
        """
        self.dt = dt
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.N = int(prediction_horizon)
        self.M = int(control_horizon)
        self.Q = float(Q)
        self.R = float(R)
        self.S = float(S)
        self.u_ss = float(u_ss)
        self.deadzone = float(deadzone)
        
        # 시스템 모델 파라미터 (RLS로 온라인 업데이트됨)
        self.a = 1.0
        self.b = 0.01
        self.c = -0.1
        
        # 이전 제어 입력 (Δu 계산 및 warm-start용)
        self.u_prev = float(u_init)
        
        # 최적화 실패 카운터
        self.consecutive_failures = 0

    def reset(self):
        """제어기 상태 초기화"""
        self.u_prev = float(self.u_ss)
        self.consecutive_failures = 0

    def set_model(self, a, b, c):
        """RLS로 식별된 모델 파라미터 설정"""
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def predict(self, x0, u_seq):
        """
        시스템 모델로 미래 상태 예측
        
        Args:
            x0: 초기 상태 (현재 수위)
            u_seq: 제어 입력 시퀀스 [u(0), u(1), ..., u(N-1)]
        
        Returns:
            예측 상태 [h(1), h(2), ..., h(N)]
        """
        x = float(x0)
        xs = []
        for u in u_seq:
            x = self.a * x + self.b * float(u) + self.c
            xs.append(x)
        return np.array(xs, dtype=float)

    def cost(self, u_seq, x0, sp):
        """
        MPC 목적함수 계산
        
        Args:
            u_seq: 제어 입력 후보 시퀀스
            x0: 현재 상태
            sp: 목표값 (setpoint)
        
        Returns:
            목적함수 값 (작을수록 좋음)
        """
        u_seq = np.array(u_seq, dtype=float)
        
        # 1. 미래 상태 예측
        xs = self.predict(x0, u_seq)
        
        # 2. 추적 오차 (Dead-zone 적용)
        err = xs - sp
        err[np.abs(err) < self.deadzone] = 0.0
        tracking_cost = self.Q * np.sum(err ** 2)
        
        # 3. 입력 평활화 (Δu 페널티)
        u_full = np.concatenate([[self.u_prev], u_seq])
        du = np.diff(u_full)
        smoothness_cost = self.R * np.sum(du ** 2)
        
        # 4. Bias 유지 (정상상태 입력 근처 유지)
        bias_cost = self.S * np.sum((u_seq - self.u_ss) ** 2)
        
        return float(tracking_cost + smoothness_cost + bias_cost)

    def update(self, setpoint, measurement):
        """
        MPC 최적화 실행
        
        Args:
            setpoint: 목표 수위
            measurement: 현재 측정 수위
        
        Returns:
            (제어 입력, 목적함수 값, 최적화 성공 여부)
        """
        x0 = float(measurement)
        sp = float(setpoint)
        
        # 초기 추측: 이전 입력으로 warm-start
        u0 = np.ones(self.N, dtype=float) * self.u_prev
        
        # 제약 조건: 펌프 속도 범위
        bounds = [(self.min_speed, self.max_speed)] * self.N
        
        # SLSQP 최적화 실행
        res = minimize(
            self.cost,
            u0,
            args=(x0, sp),
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 80,      # 최대 반복 횟수
                "ftol": 1e-3,       # 목적함수 수렴 허용오차
                "disp": False       # 최적화 진행 메시지 숨김
            }
        )
        
        # 최적화 성공
        if res.success and res.x is not None and len(res.x) > 0:
            u = float(res.x[0])  # Receding horizon: 첫 번째 입력만 사용
            self.u_prev = u
            J_star = float(res.fun)
            self.consecutive_failures = 0
            return int(clamp(u, self.min_speed, self.max_speed)), J_star, True
        
        # 최적화 실패: 이전 입력 유지
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 5:
                print(f"\n⚠️ MPC 최적화 {self.consecutive_failures}회 연속 실패!")
            
            return int(clamp(self.u_prev, self.min_speed, self.max_speed)), float("nan"), False


# =============================================================================
# 6. 펌프 제어 클래스
# =============================================================================

class PumpController:
    """Arduino를 통한 펌프 및 밸브 제어"""
    
    def __init__(self, board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5):
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin
        
        self.open_all_valves()
        self.set_pump_speed(0)

    def open_all_valves(self):
        """모든 밸브 열기"""
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print("모든 밸브 열림 (Inlet: Pin 7, Outlet: Pin 5)")

    def close_all_valves(self):
        """모든 밸브 닫기"""
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\n모든 밸브 닫힘")

    def set_pump_speed(self, speed: int):
        """펌프 속도 설정"""
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        """안전한 종료"""
        self.set_pump_speed(0)
        self.close_all_valves()


# =============================================================================
# 7. 영상 처리 함수
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
    """픽셀 좌표를 실제 수위(cm)로 변환"""
    _, tank_top_y, _, tank_bottom_y = tank_box
    
    if tank_bottom_y <= tank_top_y:
        return None
    
    liquid_height_cm = float(tank_bottom_y - liquid_line) / \
                      float(tank_bottom_y - tank_top_y) * TANK_HEIGHT_CM
    liquid_height_cm = max(0.0, min(TANK_HEIGHT_CM, liquid_height_cm))
    
    return liquid_height_cm


# =============================================================================
# 8. 스레드 함수
# =============================================================================

def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    """센싱 스레드: 카메라로 수위 측정"""
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    model = YOLO(MODEL_PATH)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "Liquid Level Monitoring (MPC)"
    
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
                else:
                    cv2.putText(frame, "Liquid level not detected", 
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

            frame_display_done_time = time.perf_counter()
            capture_to_display_ms = (frame_display_done_time - inference_start_time) * 1000
            
            if liquid_height_cm is not None:
                print(f"Level: {liquid_height_cm:.2f}cm, "
                      f"Inference: {inference_time_ms:.1f}ms, "
                      f"Frame: {capture_to_display_ms:.1f}ms", end="\r")
            else:
                print(f"Level: None, "
                      f"Inference: {inference_time_ms:.1f}ms, "
                      f"Frame: {capture_to_display_ms:.1f}ms", end="\r")

    finally:
        camera.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()


def control_thread_fn(shared: SharedLevel, pump: PumpController,
                     stop_event: threading.Event, log: SharedLog):
    """
    제어 스레드: RLS 모델 식별 + MPC 최적 제어
    
    작동 과정:
    1. 현재 수위 읽기
    2. RLS로 시스템 모델 업데이트 (h_k, u_k, h_k+1)
    3. 모델 파라미터 안전 범위 클램핑
    4. MPC 최적화 실행
    5. 제어 입력 적용
    6. 결과 로깅
    """
    # RLS 초기화
    rls = OnlineRLS(
        a0=RLS_INITIAL_A,
        b0=RLS_INITIAL_B,
        c0=RLS_INITIAL_C,
        P0=RLS_INITIAL_P,
        lam=RLS_FORGETTING_FACTOR
    )
    
    # MPC 초기화
    mpc = MPCController(
        dt=CONTROL_PERIOD_S,
        max_speed=MAX_PUMP_SPEED,
        min_speed=MIN_PUMP_SPEED,
        prediction_horizon=PREDICTION_HORIZON,
        control_horizon=CONTROL_HORIZON,
        Q=Q_WEIGHT,
        R=R_WEIGHT,
        u_init=U_SS,
        deadzone=LEVEL_TOLERANCE_CM,
        u_ss=U_SS,
        S=S_WEIGHT
    )

    next_tick = time.time()
    h_prev = None  # 이전 수위 (RLS 업데이트용)
    u_prev_applied = float(U_SS)  # 이전 제어 입력

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
            if (not valid) or (liquid_height_cm is None) or (age > STALE_DATA_TIMEOUT_S):
                pump.set_pump_speed(0)
                log.add(time.time(), liquid_height_cm, 0, SETPOINT_CM, 
                       float("nan"), False, 0.0)
                mpc.reset()
                h_prev = None
                u_prev_applied = float(U_SS)
                next_tick = tick_start + CONTROL_PERIOD_S
                continue

            h_now = float(liquid_height_cm)

            # ==== 1) RLS 온라인 모델 업데이트 ====
            if h_prev is not None:
                a_hat, b_hat, c_hat = rls.update(h_prev, u_prev_applied, h_now)
                
                # 모델 파라미터 안전 범위 클램핑
                # (불안정한 모델이나 비물리적 값 방지)
                a_hat = float(clamp(a_hat, 0.7, 1.02))  # a>1이면 불안정
                b_hat = float(clamp(b_hat, 0.0, 0.2))   # 음수 게인 방지
                c_hat = float(clamp(c_hat, -2.0, 2.0))  # 과도한 바이어스 방지
                
                mpc.set_model(a_hat, b_hat, c_hat)

            # ==== 2) MPC 최적화 ====
            opt_start = time.perf_counter()
            u_cmd, J_star, ok = mpc.update(SETPOINT_CM, h_now)
            opt_time_ms = (time.perf_counter() - opt_start) * 1000

            # ==== 3) 펌프 제어 적용 ====
            pump.set_pump_speed(u_cmd)

            # ==== 4) 제어 정보 출력 ====
            print(f"\n[MPC] h={h_now:.2f}cm, u={u_cmd:2d}, J*={J_star:.4f}, "
                  f"opt_time={opt_time_ms:.1f}ms, ok={ok}")
            
            # 계산 시간 경고
            if opt_time_ms > CONTROL_PERIOD_S * 1000:
                print(f"⚠️ WARNING: 최적화 시간이 제어 주기를 초과!")

            # ==== 5) 로그 기록 ====
            log.add(time.time(), h_now, u_cmd, SETPOINT_CM, 
                   J_star, ok, opt_time_ms)

            # ==== 6) 다음 제어 주기 준비 ====
            h_prev = h_now
            u_prev_applied = float(u_cmd)
            next_tick = tick_start + CONTROL_PERIOD_S

    finally:
        print("\n" + "="*70)
        
        # 최종 식별된 모델 출력
        a_hat, b_hat, c_hat = rls.get_params()
        confidence = rls.get_confidence()
        print(f"[Final Identified Model]")
        print(f"  a = {a_hat:.4f}, b = {b_hat:.4f}, c = {c_hat:.4f}")
        print(f"  Updates: {rls.n_updates}, Confidence: {confidence:.2f}")
        print("="*70)
        
        pump.shutdown()


# =============================================================================
# 9. 결과 시각화
# =============================================================================

def plot_results(log: SharedLog):
    """실험 결과 그래프 생성"""
    t, level, speed, sp, cost, ok, opt_time = log.snapshot()
    
    if len(t) < 2:
        print("그래프를 그릴 데이터가 충분하지 않습니다.")
        return

    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # None 값 제거
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    # 그래프 1: 수위 변화
    plt.figure(figsize=(12, 4))
    plt.plot(t_level, level_valid, 'b-', linewidth=2, label="Measured Level")
    plt.axhline(SETPOINT_CM, color='r', linestyle='--', linewidth=2, label="Setpoint")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Water Level (cm)", fontsize=12)
    plt.title("MPC Water Level Control - System Response", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 그래프 2: 펌프 속도
    plt.figure(figsize=(12, 4))
    plt.plot(t_rel, speed, 'g-', linewidth=2, label="Pump Speed")
    plt.axhline(U_SS, color='orange', linestyle='--', linewidth=1, 
                label=f"Steady State ({U_SS}%)")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Pump Speed (%)", fontsize=12)
    plt.title("MPC Water Level Control - Control Input", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 그래프 3: MPC 목적함수
    plt.figure(figsize=(12, 4))
    plt.plot(t_rel, cost, 'm-', linewidth=2, label="MPC Cost J*")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.title("MPC Optimization - Objective Function", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 그래프 4: 최적화 계산 시간
    plt.figure(figsize=(12, 4))
    plt.plot(t_rel, opt_time, 'c-', linewidth=2, label="Optimization Time")
    plt.axhline(CONTROL_PERIOD_S * 1000, color='r', linestyle='--', 
                linewidth=1, label=f"Control Period ({CONTROL_PERIOD_S*1000:.0f}ms)")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    plt.title("MPC Computational Performance", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.show()


# =============================================================================
# 10. 메인 함수
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("적응형 MPC 수위 제어 시스템 시작")
    print("=" * 70)
    print(f"목표 수위: {SETPOINT_CM} cm")
    print(f"제어 주기: {CONTROL_PERIOD_S} 초")
    print(f"예측 구간: {PREDICTION_HORIZON} 스텝 ({PREDICTION_HORIZON * CONTROL_PERIOD_S:.0f}초)")
    print("\n[MPC 파라미터]")
    print(f"  Q (추적): {Q_WEIGHT}")
    print(f"  R (평활): {R_WEIGHT}")
    print(f"  S (Bias): {S_WEIGHT}")
    print(f"  U_SS: {U_SS}%")
    print("\n[RLS 파라미터]")
    print(f"  Forgetting factor: {RLS_FORGETTING_FACTOR}")
    print(f"  메모리 길이: ~{1/(1-RLS_FORGETTING_FACTOR):.0f} 샘플")
    print("=" * 70)
    print("\n종료: ESC 키 또는 Ctrl+C\n")
    
    # 초기화
    shared = SharedLevel()
    log = SharedLog()
    stop_event = threading.Event()
    pump = PumpController(board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5)

    # 스레드 생성 및 시작
    t_sense = threading.Thread(
        target=sensing_thread_fn,
        args=(shared, stop_event),
        daemon=True,
        name="SensingThread"
    )
    t_ctrl = threading.Thread(
        target=control_thread_fn,
        args=(shared, pump, stop_event, log),
        daemon=True,
        name="ControlThread"
    )

    t_sense.start()
    t_ctrl.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n종료 신호 감지 (Ctrl+C)")
        stop_event.set()

    # 스레드 종료 대기
    print("스레드 종료 대기 중...")
    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)
    
    print("\n" + "=" * 70)
    print("프로그램 종료 완료")
    print("=" * 70)

    # 결과 시각화
    print("\n실험 결과 그래프를 생성합니다...")
    plot_results(log)


# =============================================================================
# 프로그램 시작점
# =============================================================================

if __name__ == "__main__":
    main()
