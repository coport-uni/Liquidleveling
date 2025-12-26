import time
import cv2
from ultralytics import YOLO

camera_index = 0
model_path = "20251223nano.pt"

liquid_class_id = 0
tank_class_id = 1

tank_height = 8

def detect_tank_and_liquidlevel(frame, model):
    inference_start_time = time.perf_counter()
    results = model(frame, conf=0.9, verbose=False)[0]
    inference_time_ms = (time.perf_counter() - inference_start_time) * 1000

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
            liquid_line = y1

    return tank_line, liquid_line, inference_time_ms

def calculate_liquidlevel(liquid_line, tank_line):
    _, tank_top_y, _, tank_bottom_y = tank_line

    if tank_bottom_y <= tank_top_y:
        return 0.0, tank_top_y, tank_bottom_y
    
    liquid_height = float(tank_bottom_y - liquid_line) / float(tank_bottom_y - tank_top_y) * tank_height
    liquid_height = max(0.0, min(tank_height, liquid_height))

    return liquid_height, tank_top_y, tank_bottom_y

def main():
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    model = YOLO(model_path)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    while True:
        ret, frame = camera.read()
        frame_received_time = time.perf_counter()

        if not ret:
            break

        tank_line, liquid_line, inference_time_ms = detect_tank_and_liquidlevel(frame, model)

        if tank_line is not None:
            x1, y1, x2, y2 = tank_line
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if liquid_line is not None:
                liquid_height, _, _ = calculate_liquidlevel(liquid_line, tank_line)

                cv2.line(frame, (x1, liquid_line), (x2, liquid_line), (0, 255, 255), 2)

                cv2.putText(frame, f"Height: {liquid_height:.1f}cm", (30, 40), cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Liquidlevel is not detected.", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Tank is not detected.", (30, 40), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

        window_name = "Liquidleveling"
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(10)

        if key & 0xFF == 27:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_display_done_time = time.perf_counter()
        capture_to_display_ms = (frame_display_done_time - frame_received_time) * 1000
        print(f"time delay: {capture_to_display_ms:.1f} ms", f"inference: {inference_time_ms:.1f} ms", end="\r")

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
