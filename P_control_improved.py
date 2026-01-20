"""
=============================================================================
수위 제어 시스템 - P 제어기
=============================================================================
YOLO 객체 탐지를 통한 실시간 수위 측정 및 P 제어기 기반 펌프 제어

주요 기능:
- 카메라를 통한 실시간 수위 감지 (YOLO 모델 사용)
- P 제어 알고리즘으로 목표 수위 유지
- Arduino를 통한 펌프 및 밸브 제어
- 멀티스레딩 기반 센싱/제어 분리
- 실험 결과 시각화 (수위, 펌프 속도 그래프)
=============================================================================
"""

import time
import threading
import cv2
from ultralytics import YOLO
from PyArduino import PyArduino
import matplotlib.pyplot as plt


# =============================================================================
# 1. 시스템 설정 및 상수 정의
# =============================================================================

# --- 카메라 설정 ---
camera_index = 1  # 사용할 카메라 번호 (0: 기본 카메라, 1: 외부 카메라)

# --- YOLO 모델 설정 ---
model_path = "20251223nano.pt"  # 학습된 YOLO 모델 파일 경로
liquid_class_id = 0  # YOLO 모델에서 액체(수위선)의 클래스 ID
tank_class_id = 1    # YOLO 모델에서 탱크의 클래스 ID

# --- 물리적 탱크 사양 ---
tank_height_cm = 8.0  # 실제 탱크의 높이 (cm)

# --- 제어 목표 ---
setpoint_cm = 3.85  # 목표 수위 (cm)

# --- P 제어기 파라미터 ---
Kp = 6.0  # P 게인 (비례 제어 게인)
steady_state_speed = 10  # 정상상태 펌프 속도 (수위 유지에 필요한 기본 유량)
level_tolerance_cm = 0.05  # 수위 오차 허용 범위 (이 이내면 오차를 0으로 간주)

# --- 제어 주기 ---
control_period_s = 3.0  # 제어 주기 (초) - 3초마다 제어 실행

# --- 펌프 속도 제한 ---
max_pump_speed = 20  # 최대 펌프 속도 (%)
min_pump_speed = 0   # 최소 펌프 속도 (%)

# --- 디스플레이 설정 ---
show_display = True  # 실시간 카메라 영상 표시 여부

# --- 센싱 타임아웃 ---
STALE_DATA_TIMEOUT_S = 3.0  # 센싱 데이터가 이 시간(초) 이상 업데이트 안 되면 안전 정지


# =============================================================================
# 2. 유틸리티 함수
# =============================================================================

def clamp(value, lower, upper):
    """
    값을 지정된 범위로 제한하는 함수
    
    Args:
        value: 제한할 값
        lower: 최솟값
        upper: 최댓값
    
    Returns:
        lower와 upper 사이로 제한된 값
    """
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
    스레드 간 수위 데이터 공유를 위한 클래스
    
    센싱 스레드에서 측정한 수위를 제어 스레드로 안전하게 전달합니다.
    Lock을 사용하여 thread-safe를 보장합니다.
    
    Attributes:
        liquid_height_cm: 현재 측정된 수위 (cm)
        timestamp: 마지막 업데이트 시각
        valid: 데이터 유효성 여부
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.liquid_height_cm = None
        self.timestamp = 0.0
        self.valid = False

    def update(self, liquid_height_cm):
        """
        수위 데이터 업데이트 (센싱 스레드에서 호출)
        
        Args:
            liquid_height_cm: 새로 측정된 수위 (cm), None이면 측정 실패
        """
        with self.lock:
            self.liquid_height_cm = liquid_height_cm
            self.timestamp = time.time()
            self.valid = liquid_height_cm is not None

    def get(self):
        """
        현재 수위 데이터 조회 (제어 스레드에서 호출)
        
        Returns:
            tuple: (수위(cm), 타임스탬프, 유효성)
        """
        with self.lock:
            return self.liquid_height_cm, self.timestamp, self.valid


class SharedLog:
    """
    실험 데이터 기록을 위한 클래스
    
    제어 과정에서 발생하는 시간, 수위, 펌프 속도, 목표값 등을 기록하여
    나중에 그래프로 시각화합니다.
    
    Attributes:
        t: 시간 데이터 리스트
        level: 수위 데이터 리스트
        speed: 펌프 속도 데이터 리스트
        setpoint: 목표 수위 데이터 리스트
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.t = []
        self.level = []
        self.speed = []
        self.setpoint = []

    def add(self, t, level, speed, setpoint):
        """
        한 시점의 데이터를 기록
        
        Args:
            t: 현재 시각 (timestamp)
            level: 현재 수위 (cm)
            speed: 현재 펌프 속도 (%)
            setpoint: 목표 수위 (cm)
        """
        with self.lock:
            self.t.append(t)
            self.level.append(level)
            self.speed.append(speed)
            self.setpoint.append(setpoint)

    def snapshot(self):
        """
        현재까지 기록된 모든 데이터의 복사본 반환
        
        Returns:
            tuple: (시간 리스트, 수위 리스트, 펌프속도 리스트, 목표값 리스트)
        """
        with self.lock:
            return (self.t[:], self.level[:], self.speed[:], self.setpoint[:])


# =============================================================================
# 4. 제어기 클래스
# =============================================================================

class PController:
    """
    비례(P) 제어기 클래스
    
    P 제어는 목표값과 현재값의 오차에 비례하여 제어 출력을 생성합니다.
    이 구현에서는 정상상태 유량(feedforward)을 추가하여 성능을 개선했습니다.
    
    제어 출력 = 정상상태_속도 + Kp × 오차
    
    Attributes:
        kp: 비례 게인
        steady_state_speed: 정상상태 펌프 속도
        max_speed: 최대 펌프 속도
    """
    
    def __init__(self, kp, steady_state_speed, max_speed):
        """
        Args:
            kp: 비례 게인
            steady_state_speed: 정상상태 유지를 위한 기본 펌프 속도
            max_speed: 펌프 속도 상한값
        """
        self.kp = kp
        self.steady_state_speed = steady_state_speed
        self.max_speed = max_speed
    
    def update(self, setpoint, measurement):
        """
        제어 출력 계산
        
        Args:
            setpoint: 목표 수위 (cm)
            measurement: 현재 측정된 수위 (cm)
        
        Returns:
            int: 계산된 펌프 속도 (0~max_speed 범위로 제한됨)
        """
        # 오차 계산 (목표값 - 측정값)
        error = setpoint - measurement
        
        # 오차가 허용 범위 이내면 0으로 처리 (Dead-band)
        if abs(error) < level_tolerance_cm:
            error = 0.0
        
        # P 제어 출력 = 정상상태 속도 + 비례 제어
        speed = self.steady_state_speed + self.kp * error
        
        # 펌프 속도를 허용 범위로 제한
        speed = int(clamp(speed, min_pump_speed, self.max_speed))
        
        return speed


class PumpController:
    """
    펌프 및 밸브 제어 클래스
    
    Arduino를 통해 펌프 속도와 밸브 개폐를 제어합니다.
    - Inlet valve: 유입 밸브 (Pin 7)
    - Outlet valve: 유출 밸브 (Pin 5)
    
    Attributes:
        pa: PyArduino 인스턴스
        inlet_valve_pin: 유입 밸브 핀 번호
        outlet_valve_pin: 유출 밸브 핀 번호
    """
    
    def __init__(self, board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5):
        """
        Args:
            board_type: Arduino 보드 타입
            inlet_valve_pin: 유입 밸브 제어 핀
            outlet_valve_pin: 유출 밸브 제어 핀
        """
        self.pa = PyArduino(board_type)
        self.inlet_valve_pin = inlet_valve_pin
        self.outlet_valve_pin = outlet_valve_pin
        
        # 시작 시 모든 밸브 열기
        self.open_all_valves()
        
        # 펌프는 정지 상태로 시작
        self.set_pump_speed(0)

    def open_all_valves(self):
        """모든 밸브를 엽니다 (실험 시작 시 호출)"""
        self.pa.run_digital_write(self.inlet_valve_pin, True)
        self.pa.run_digital_write(self.outlet_valve_pin, True)
        print("모든 밸브 열림 (Inlet: Pin 7, Outlet: Pin 5)")

    def close_all_valves(self):
        """모든 밸브를 닫습니다 (실험 종료 시 호출)"""
        self.pa.run_digital_write(self.inlet_valve_pin, False)
        self.pa.run_digital_write(self.outlet_valve_pin, False)
        print("\n모든 밸브 닫힘")

    def set_pump_speed(self, speed: int):
        """
        펌프 속도 설정
        
        Args:
            speed: 펌프 속도 (0~20 범위)
        """
        self.pa.run_pump_speed(speed)

    def shutdown(self):
        """
        안전한 종료 처리
        - 펌프 정지
        - 모든 밸브 닫기
        """
        self.set_pump_speed(0)
        self.close_all_valves()


# =============================================================================
# 5. 영상 처리 함수
# =============================================================================

def detect_tank_and_liquid(frame, model):
    """
    YOLO 모델을 사용하여 프레임에서 탱크와 수위선 탐지
    
    Args:
        frame: OpenCV 이미지 프레임
        model: YOLO 모델 객체
    
    Returns:
        tuple: (탱크 박스, 수위선 y좌표, 추론 시간(ms))
            - 탱크 박스: (x1, y1, x2, y2) 또는 None
            - 수위선: y좌표 또는 None
            - 추론 시간: 밀리초 단위
    """
    # YOLO 추론 시작
    inference_start_time = time.perf_counter()
    results = model(frame, conf=0.9, verbose=False)[0]
    inference_time_ms = (time.perf_counter() - inference_start_time) * 1000.0

    # 탱크와 수위선 정보 초기화
    tank_box = None
    best_tank_confidence = -1.0

    liquid_line = None
    best_liquid_confidence = -1.0

    # 탐지된 모든 객체에 대해 반복
    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 탱크 탐지 (가장 높은 신뢰도의 탱크만 사용)
        if class_id == tank_class_id and conf > best_tank_confidence:
            best_tank_confidence = conf
            tank_box = (x1, y1, x2, y2)

        # 수위선 탐지 (가장 높은 신뢰도의 수위선만 사용)
        elif class_id == liquid_class_id and conf > best_liquid_confidence:
            best_liquid_confidence = conf
            liquid_line = y1  # 수위선의 상단 y좌표

    return tank_box, liquid_line, inference_time_ms


def calculate_liquid_level_cm(liquid_line, tank_box):
    """
    픽셀 좌표를 실제 수위(cm)로 변환
    
    탱크의 상단과 하단 픽셀 좌표를 이용하여
    수위선의 픽셀 위치를 실제 높이(cm)로 변환합니다.
    
    Args:
        liquid_line: 수위선의 y좌표 (픽셀)
        tank_box: 탱크 박스 (x1, y1, x2, y2)
    
    Returns:
        float: 수위 높이 (cm), 계산 실패 시 None
    """
    _, tank_top_y, _, tank_bottom_y = tank_box
    
    # 탱크 박스가 유효하지 않으면 None 반환
    if tank_bottom_y <= tank_top_y:
        return None
    
    # 픽셀 좌표를 cm로 변환
    # (탱크 바닥 - 수위선) / (탱크 전체 높이) × 실제 탱크 높이
    liquid_height_cm = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * tank_height_cm
    
    # 물리적으로 가능한 범위로 제한 (0 ~ 탱크 높이)
    liquid_height_cm = max(0.0, min(tank_height_cm, liquid_height_cm))
    
    return liquid_height_cm


# =============================================================================
# 6. 스레드 함수
# =============================================================================

def sensing_thread_fn(shared: SharedLevel, stop_event: threading.Event):
    """
    센싱 스레드 함수
    
    카메라로부터 영상을 읽어 YOLO로 수위를 측정하고,
    측정 결과를 SharedLevel 객체에 업데이트합니다.
    
    작동 과정:
    1. 카메라에서 프레임 읽기
    2. YOLO로 탱크와 수위선 탐지
    3. 픽셀 좌표를 실제 수위(cm)로 변환
    4. SharedLevel에 결과 업데이트
    5. (옵션) 화면에 결과 표시
    
    Args:
        shared: 수위 데이터 공유 객체
        stop_event: 스레드 종료 신호
    """
    # 카메라 초기화
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    model = YOLO(model_path)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        stop_event.set()
        return

    window_name = "Liquid Level Monitoring"
    
    try:
        while not stop_event.is_set():
            # 프레임 읽기
            ret, frame = camera.read()
            inference_start_time = time.perf_counter()

            if not ret:
                print("프레임을 읽을 수 없습니다.")
                stop_event.set()
                break

            # YOLO로 탱크와 수위선 탐지
            tank_box, liquid_line, inference_time_ms = detect_tank_and_liquid(frame, model)

            liquid_height_cm = None
            
            # 탱크가 탐지된 경우
            if tank_box is not None:
                x1, y1, x2, y2 = tank_box
                # 탱크 박스 그리기 (녹색)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 수위선이 탐지된 경우
                if liquid_line is not None:
                    # 수위 계산
                    liquid_height_cm = calculate_liquid_level_cm(liquid_line, tank_box)
                    # 수위선 그리기 (노란색)
                    cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)
                    # 수위 텍스트 표시
                    cv2.putText(frame, f"Height: {liquid_height_cm:.2f}cm", 
                               (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Liquid level not detected", 
                               (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Tank not detected", 
                           (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

            # 측정 결과를 공유 객체에 업데이트
            shared.update(liquid_height_cm)

            # 화면 표시
            if show_display:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(10)
                
                # ESC 키로 종료
                if key & 0xFF == 27:
                    print("\n프로그램 종료 중")
                    stop_event.set()
                    break
                
                # 창 닫기로 종료
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n프로그램 종료 중")
                    stop_event.set()
                    break

            # 성능 정보 출력
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
        if show_display:
            cv2.destroyAllWindows()


def control_thread_fn(shared: SharedLevel, pump: PumpController, 
                     stop_event: threading.Event, log: SharedLog):
    """
    제어 스레드 함수
    
    SharedLevel에서 수위 데이터를 읽어 P 제어기로 펌프 속도를 계산하고,
    Arduino를 통해 펌프를 제어합니다.
    
    작동 과정:
    1. control_period_s 마다 제어 실행
    2. SharedLevel에서 최신 수위 읽기
    3. 데이터가 오래되었거나 유효하지 않으면 안전 정지
    4. P 제어기로 펌프 속도 계산
    5. 펌프 제어 명령 전송
    6. 결과를 로그에 기록
    
    Args:
        shared: 수위 데이터 공유 객체
        pump: 펌프 제어 객체
        stop_event: 스레드 종료 신호
        log: 데이터 로깅 객체
    """
    # P 제어기 초기화
    p_controller = PController(Kp, steady_state_speed, max_pump_speed)

    # 다음 제어 시점 계산
    next_tick = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()
            
            # 다음 제어 시점까지 대기
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            # 이번 제어 주기 시작
            tick_start = time.time()

            # 최신 수위 데이터 읽기
            liquid_height_cm, ts, valid = shared.get()
            age = time.time() - ts

            # 데이터 유효성 검사
            # - 데이터가 유효하지 않거나
            # - 수위가 None이거나
            # - 데이터가 너무 오래된 경우 → 안전을 위해 펌프 정지
            if (not valid) or (liquid_height_cm is None) or (age > STALE_DATA_TIMEOUT_S):
                pump.set_pump_speed(0)
                log.add(time.time(), liquid_height_cm, 0, setpoint_cm)
                next_tick = tick_start + control_period_s
                continue

            # P 제어기로 펌프 속도 계산
            speed = p_controller.update(setpoint_cm, liquid_height_cm)

            # 펌프 제어 실행
            pump.set_pump_speed(speed)

            # 제어 결과 기록 (시간, 수위, 펌프속도, 목표값)
            log.add(time.time(), liquid_height_cm, speed, setpoint_cm)

            # 다음 제어 시점 예약
            next_tick = tick_start + control_period_s

    finally:
        # 스레드 종료 시 안전하게 펌프 정지 및 밸브 닫기
        pump.shutdown()


# =============================================================================
# 7. 결과 시각화
# =============================================================================

def plot_results(log: SharedLog):
    """
    실험 결과를 그래프로 시각화
    
    두 개의 그래프를 생성합니다:
    1. 상태 그래프: 시간에 따른 수위 변화 및 목표값
    2. 입력 그래프: 시간에 따른 펌프 속도 변화
    
    Args:
        log: 실험 데이터가 기록된 로그 객체
    """
    # 로그 데이터 가져오기
    t, level, speed, sp = log.snapshot()
    
    if len(t) < 2:
        print("그래프를 그릴 데이터가 충분하지 않습니다.")
        return

    # 상대 시간으로 변환 (실험 시작 시점을 0으로)
    t0 = t[0]
    t_rel = [x - t0 for x in t]

    # None 값 제거 (수위 그래프용)
    # 센서가 탱크를 감지하지 못한 구간의 데이터 제거
    t_level = [tt for tt, lv in zip(t_rel, level) if lv is not None]
    level_valid = [lv for lv in level if lv is not None]

    # 그래프 1: 수위 변화
    plt.figure(figsize=(10, 5))
    plt.plot(t_level, level_valid, 'b-', linewidth=2, label="Measured Level")
    plt.axhline(setpoint_cm, color='r', linestyle='--', linewidth=2, label="Setpoint")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Water Level (cm)", fontsize=12)
    plt.title("Water Level Control - System Response", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # 그래프 2: 펌프 속도 변화
    plt.figure(figsize=(10, 5))
    plt.plot(t_rel, speed, 'g-', linewidth=2, label="Pump Speed")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Pump Speed (%)", fontsize=12)
    plt.title("Water Level Control - Control Input", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.show()


# =============================================================================
# 8. 메인 함수
# =============================================================================

def main():
    """
    메인 함수
    
    프로그램의 전체 실행 흐름을 관리합니다:
    1. 공유 객체 및 제어기 초기화
    2. 센싱 스레드와 제어 스레드 시작
    3. 사용자 종료 신호 대기
    4. 안전한 종료 처리
    5. 실험 결과 시각화
    """
    print("=" * 70)
    print("수위 제어 시스템 시작")
    print("=" * 70)
    print(f"목표 수위: {setpoint_cm} cm")
    print(f"P 게인: {Kp}")
    print(f"제어 주기: {control_period_s} 초")
    print(f"펌프 속도 범위: {min_pump_speed}~{max_pump_speed}%")
    print("=" * 70)
    print("\n종료하려면 ESC 키를 누르거나 Ctrl+C를 입력하세요.\n")
    
    # 공유 객체 초기화
    shared = SharedLevel()
    log = SharedLog()
    stop_event = threading.Event()
    
    # 펌프 제어기 초기화
    pump = PumpController(board_type="minima", inlet_valve_pin=7, outlet_valve_pin=5)

    # 스레드 생성
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

    # 스레드 시작
    t_sense.start()
    t_ctrl.start()

    try:
        # 메인 스레드는 종료 신호 대기
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Ctrl+C로 종료
        print("\n종료 신호 감지 (Ctrl+C)")
        stop_event.set()

    # 스레드 종료 대기
    print("스레드 종료 대기 중...")
    t_sense.join(timeout=1.0)
    t_ctrl.join(timeout=1.0)
    
    print("\n" + "=" * 70)
    print("프로그램 종료 완료")
    print("=" * 70)

    # 실험 결과 시각화
    print("\n실험 결과 그래프를 생성합니다...")
    plot_results(log)


# =============================================================================
# 프로그램 시작점
# =============================================================================

if __name__ == "__main__":
    main()
