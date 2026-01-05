import time
import cv2
from ultralytics import YOLO
from PyArduino import PyArduino
import numpy as np
import pandas as pd

# 설정
camera_index = 0
model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1
tank_height = 8  # cm

# 핀 설정
SUPPLY_VALVE_PIN = 5  # 급수 밸브
DRAIN_VALVE_PIN = 6   # 배수 밸브

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        """PID 제어기"""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None
        
    def compute(self, current_value):
        """PID 계산 - 출력: -100 ~ +100"""
        current_time = time.perf_counter()
        
        if self.previous_time is None:
            self.previous_time = current_time
            return 0.0
        
        dt = current_time - self.previous_time
        if dt <= 0.0:
            return 0.0
        
        # 오차 계산
        error = self.setpoint - current_value
        
        # 비례항 (P)
        P = self.Kp * error
        
        # 적분항 (I) - Anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, 50), -50)
        I = self.Ki * self.integral
        
        # 미분항 (D)
        derivative = (error - self.previous_error) / dt
        D = self.Kd * derivative
        
        # PID 출력
        output = P + I + D
        output = max(min(output, 100), -100)
        
        # 다음 계산을 위해 저장
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def set_setpoint(self, new_setpoint):
        """목표 수위 변경"""
        self.setpoint = new_setpoint
        
    def reset(self):
        """PID 초기화"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None


class ValveController:
    """ON/OFF 밸브 시간 비율 제어"""
    def __init__(self, pa, supply_pin, drain_pin, cycle_time=10.0):
        """
        pa: PyArduino 인스턴스
        supply_pin: 급수 밸브 핀
        drain_pin: 배수 밸브 핀
        cycle_time: 제어 주기 (초) - 이 시간마다 ON/OFF 결정
        """
        self.pa = pa
        self.supply_pin = supply_pin
        self.drain_pin = drain_pin
        self.cycle_time = cycle_time
        
        self.cycle_start_time = time.time()
        self.current_supply_state = False
        self.current_drain_state = False
        
        # 초기 상태: 모든 밸브 OFF
        self.set_valves(False, False)
    
    def set_valves(self, supply_state, drain_state):
        """밸브 상태 설정"""
        if supply_state != self.current_supply_state:
            self.pa.run_digital_write(self.supply_pin, supply_state)
            self.current_supply_state = supply_state
            
        if drain_state != self.current_drain_state:
            self.pa.run_digital_write(self.drain_pin, drain_state)
            self.current_drain_state = drain_state
    
    def update(self, pid_output):
        """
        PID 출력에 따라 밸브 제어
        pid_output: -100(배수) ~ 0(정지) ~ +100(급수)
        """
        current_time = time.time()
        elapsed = current_time - self.cycle_start_time
        
        # 새 주기 시작
        if elapsed >= self.cycle_time:
            self.cycle_start_time = current_time
            elapsed = 0
        
        # 시간 비율 계산
        duty_ratio = elapsed / self.cycle_time
        
        # PID 출력의 절대값을 듀티비로 사용
        output_magnitude = abs(pid_output)
        
        if pid_output > 5:  # 급수 필요 (양수)
            on_ratio = output_magnitude / 100.0
            supply_on = duty_ratio < on_ratio
            drain_on = False
            
        elif pid_output < -5:  # 배수 필요 (음수)
            on_ratio = output_magnitude / 100.0
            supply_on = False
            drain_on = duty_ratio < on_ratio
            
        else:  # 데드존 (-5 ~ +5): 모든 밸브 OFF
            supply_on = False
            drain_on = False
        
        self.set_valves(supply_on, drain_on)
        
        return supply_on, drain_on
    
    def emergency_stop(self):
        """비상 정지: 모든 밸브 닫기"""
        self.set_valves(False, False)


class LiquidLevelPIDSystem:
    def __init__(self, model_path, camera_index, board_type, target_level=4.0):
        """
        수위 PID 제어 시스템
        """
        # 아두이노 초기화
        print("아두이노 초기화 중...")
        self.pa = PyArduino(board_type)
        time.sleep(1)
        
        # YOLO 모델 및 카메라 초기화
        print("카메라 및 YOLO 모델 로딩 중...")
        self.model = YOLO(model_path)
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.camera.isOpened():
            raise Exception("카메라를 열 수 없습니다.")
        
        # 밸브 컨트롤러 초기화
        self.valve_controller = ValveController(
            self.pa, 
            SUPPLY_VALVE_PIN, 
            DRAIN_VALVE_PIN,
            cycle_time=10.0  # 10초 주기
        )
        
        # PID 제어기 초기화
        self.pid = PIDController(
            Kp=20.0,   # 비례 게인
            Ki=2.0,    # 적분 게인
            Kd=8.0,    # 미분 게인
            setpoint=target_level
        )
        
        # 데이터 로깅
        self.data_log = {
            'time': [],
            'current_level': [],
            'target_level': [],
            'error': [],
            'pid_output': [],
            'supply_state': [],
            'drain_state': []
        }
        
        self.start_time = time.time()
        self.control_mode = 'AUTO'  # 'AUTO', 'MANUAL', 'STOP'
        self.manual_mode = 'OFF'  # 'SUPPLY', 'DRAIN', 'OFF'
        
        # 안전 설정
        self.max_level = 7.5  # 최대 수위 (cm)
        self.min_level = 0.5  # 최소 수위 (cm)
        
    def detect_tank_and_liquid(self, frame):
        """수위 감지"""
        results = self.model(frame, conf=0.9, verbose=False)[0]
        
        tank_line = None
        highest_tank_confidence = -1.0
        liquid_line = None
        highest_liquid_confidence = -1.0
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            if class_id == tank_class_id and conf > highest_tank_confidence:
                highest_tank_confidence = conf
                tank_line = (x1, y1, x2, y2)
            elif class_id == liquid_class_id and conf > highest_liquid_confidence:
                highest_liquid_confidence = conf
                liquid_line = (y1 + y2) // 2
        
        return tank_line, liquid_line
    
    def calculate_liquid_level(self, liquid_line, tank_line):
        """수위 계산"""
        if tank_line is None or liquid_line is None:
            return None
        
        _, tank_top_y, _, tank_bottom_y = tank_line
        
        if tank_bottom_y <= tank_top_y:
            return None
        
        liquid_height = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * tank_height
        liquid_height = max(0.0, min(tank_height, liquid_height))
        
        return liquid_height
    
    def safety_check(self, current_level):
        """안전 체크"""
        if current_level is None:
            return True
        
        # 최대 수위 초과
        if current_level >= self.max_level:
            print(f"\n⚠️ 경고: 최대 수위 초과! ({current_level:.2f}cm)")
            self.valve_controller.emergency_stop()
            return False
        
        # 최소 수위 미달
        if current_level <= self.min_level:
            print(f"\n⚠️ 경고: 최소 수위 미달! ({current_level:.2f}cm)")
            return True
        
        return True
    
    def draw_info(self, frame, current_level, pid_output, supply_on, drain_on, tank_line, liquid_line):
        """화면에 정보 표시"""
        # 탱크 및 액체 라인 그리기
        if tank_line is not None:
            x1, y1, x2, y2 = tank_line
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 목표 수위 라인
            if current_level is not None:
                target_y = int(y2 - (self.pid.setpoint / tank_height) * (y2 - y1))
                cv2.line(frame, (x1, target_y), (x2, target_y), (255, 0, 0), 2)
                cv2.putText(frame, "Target", (x2 + 5, target_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 안전 범위 표시
            max_y = int(y2 - (self.max_level / tank_height) * (y2 - y1))
            min_y = int(y2 - (self.min_level / tank_height) * (y2 - y1))
            cv2.line(frame, (x1, max_y), (x2, max_y), (0, 0, 255), 1)
            cv2.line(frame, (x1, min_y), (x2, min_y), (0, 0, 255), 1)
            
            if liquid_line is not None:
                cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)
        
        # 정보 패널
        info_y = 30
        line_height = 30
        
        # 현재 수위
        if current_level is not None:
            level_color = (0, 255, 255)
            if current_level >= self.max_level or current_level <= self.min_level:
                level_color = (0, 0, 255)
            cv2.putText(frame, f"Current: {current_level:.2f}cm", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
        else:
            cv2.putText(frame, "Level: N/A", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        info_y += line_height
        
        # 목표 수위
        cv2.putText(frame, f"Target: {self.pid.setpoint:.2f}cm", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        info_y += line_height
        
        # 오차
        if current_level is not None:
            error = self.pid.setpoint - current_level
            cv2.putText(frame, f"Error: {error:+.2f}cm", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            info_y += line_height
        
        # PID 출력
        output_color = (0, 255, 0) if pid_output > 0 else (255, 0, 255) if pid_output < 0 else (128, 128, 128)
        cv2.putText(frame, f"PID: {pid_output:+.1f}%", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, output_color, 2)
        info_y += line_height
        
        # 밸브 상태
        supply_text = "SUPPLY: ON" if supply_on else "SUPPLY: OFF"
        supply_color = (0, 255, 0) if supply_on else (128, 128, 128)
        cv2.putText(frame, supply_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, supply_color, 2)
        info_y += line_height
        
        drain_text = "DRAIN: ON" if drain_on else "DRAIN: OFF"
        drain_color = (0, 255, 255) if drain_on else (128, 128, 128)
        cv2.putText(frame, drain_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, drain_color, 2)
        info_y += line_height
        
        # 제어 모드
        mode_color = (0, 255, 0) if self.control_mode == 'AUTO' else (255, 165, 0) if self.control_mode == 'MANUAL' else (0, 0, 255)
        mode_text = f"Mode: {self.control_mode}"
        if self.control_mode == 'MANUAL':
            mode_text += f" ({self.manual_mode})"
        cv2.putText(frame, mode_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # 조작 안내
        help_y = frame.shape[0] - 120
        cv2.putText(frame, "=== Commands ===", (10, help_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        help_y += 25
        cv2.putText(frame, "Q/A: Target +/-0.5cm  |  W/S: PID Kp +/-1.0", 
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        help_y += 20
        cv2.putText(frame, "M: Manual  |  P: Auto  |  SPACE: Stop  |  ESC: Exit", 
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        help_y += 20
        cv2.putText(frame, "[Manual] 1:Supply  2:Drain  0:Off", 
                   (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def log_data(self, current_level, pid_output, supply_on, drain_on):
        """데이터 기록"""
        if current_level is not None:
            elapsed_time = time.time() - self.start_time
            error = self.pid.setpoint - current_level
            
            self.data_log['time'].append(elapsed_time)
            self.data_log['current_level'].append(current_level)
            self.data_log['target_level'].append(self.pid.setpoint)
            self.data_log['error'].append(error)
            self.data_log['pid_output'].append(pid_output)
            self.data_log['supply_state'].append(1 if supply_on else 0)
            self.data_log['drain_state'].append(1 if drain_on else 0)
    
    def save_log(self, filename='pid_log.csv'):
        """로그 CSV 저장"""
        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        print(f"로그 저장 완료: {filename}")
    
    def run(self):
        """메인 제어 루프"""
        print("\n" + "="*50)
        print("PID 수위 제어 시스템 시작")
        print("="*50)
        print(f"목표 수위: {self.pid.setpoint}cm")
        print(f"안전 범위: {self.min_level}cm ~ {self.max_level}cm")
        print(f"급수 밸브: 핀 {SUPPLY_VALVE_PIN}")
        print(f"배수 밸브: 핀 {DRAIN_VALVE_PIN}")
        print("="*50 + "\n")
        
        window_name = "PID Liquid Level Control"
        cv2.namedWindow(window_name)
        
        pid_output = 0.0
        supply_on = False
        drain_on = False
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # 수위 감지
                tank_line, liquid_line = self.detect_tank_and_liquid(frame)
                current_level = self.calculate_liquid_level(liquid_line, tank_line)
                
                # 안전 체크
                if not self.safety_check(current_level):
                    self.control_mode = 'STOP'
                
                # 제어 로직
                if self.control_mode == 'AUTO' and current_level is not None:
                    # PID 계산
                    pid_output = self.pid.compute(current_level)
                    # 밸브 제어
                    supply_on, drain_on = self.valve_controller.update(pid_output)
                    
                elif self.control_mode == 'MANUAL':
                    if self.manual_mode == 'SUPPLY':
                        self.valve_controller.set_valves(True, False)
                        supply_on, drain_on = True, False
                        pid_output = 100
                    elif self.manual_mode == 'DRAIN':
                        self.valve_controller.set_valves(False, True)
                        supply_on, drain_on = False, True
                        pid_output = -100
                    else:  # OFF
                        self.valve_controller.set_valves(False, False)
                        supply_on, drain_on = False, False
                        pid_output = 0
                        
                else:  # STOP 또는 수위 감지 실패
                    self.valve_controller.emergency_stop()
                    supply_on, drain_on = False, False
                    pid_output = 0
                
                # 데이터 로깅
                self.log_data(current_level, pid_output, supply_on, drain_on)
                
                # 화면 표시
                frame = self.draw_info(frame, current_level, pid_output, supply_on, drain_on, tank_line, liquid_line)
                cv2.imshow(window_name, frame)
                
                # 키 입력 처리
                key = cv2.waitKey(10) & 0xFF
                
                if key == 27:  # ESC - 종료
                    break
                    
                elif key == ord('q') or key == ord('Q'):  # 목표 수위 증가
                    new_target = min(self.max_level - 0.5, self.pid.setpoint + 0.5)
                    self.pid.set_setpoint(new_target)
                    print(f"목표 수위: {self.pid.setpoint:.1f}cm")
                    
                elif key == ord('a') or key == ord('A'):  # 목표 수위 감소
                    new_target = max(self.min_level + 0.5, self.pid.setpoint - 0.5)
                    self.pid.set_setpoint(new_target)
                    print(f"목표 수위: {self.pid.setpoint:.1f}cm")
                    
                elif key == ord('w') or key == ord('W'):  # Kp 증가
                    self.pid.Kp += 1.0
                    print(f"Kp: {self.pid.Kp:.1f}")
                    
                elif key == ord('s') or key == ord('S'):  # Kp 감소
                    self.pid.Kp = max(0, self.pid.Kp - 1.0)
                    print(f"Kp: {self.pid.Kp:.1f}")
                    
                elif key == ord('m') or key == ord('M'):  # 수동 모드
                    self.control_mode = 'MANUAL'
                    self.manual_mode = 'OFF'
                    print("수동 모드 (1:급수, 2:배수, 0:정지)")
                    
                elif key == ord('p') or key == ord('P'):  # 자동 모드
                    self.control_mode = 'AUTO'
                    self.pid.reset()
                    print("자동 모드")
                    
                elif key == ord(' '):  # 스페이스 - 정지
                    self.control_mode = 'STOP'
                    self.valve_controller.emergency_stop()
                    print("정지")
                    
                # 수동 모드 밸브 제어
                elif key == ord('1') and self.control_mode == 'MANUAL':
                    self.manual_mode = 'SUPPLY'
                    print("급수 밸브 ON")
                    
                elif key == ord('2') and self.control_mode == 'MANUAL':
                    self.manual_mode = 'DRAIN'
                    print("배수 밸브 ON")
                    
                elif key == ord('0') and self.control_mode == 'MANUAL':
                    self.manual_mode = 'OFF'
                    print("모든 밸브 OFF")
                
                # 윈도우 닫힘 감지
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            print("\n프로그램 중단 (Ctrl+C)")
        except Exception as e:
            print(f"\n오류 발생: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("\n시스템 종료 중...")
        self.valve_controller.emergency_stop()
        time.sleep(0.5)
        
        self.camera.release()
        cv2.destroyAllWindows()
        self.save_log()
        print("종료 완료")


# ========== 실행 ==========
if __name__ == "__main__":
    system = LiquidLevelPIDSystem(
        model_path="20251223nano.pt",
        camera_index=0,
        board_type="minima",  # 또는 "wifi"
        target_level=4.0  # 초기 목표 수위 4cm
    )
    system.run()