import cv2
import time
from ultralytics import YOLO

CONF_THRESHOLD = 0.5    # 신뢰도 임계값
TARGET_CLASSES = None   # None이면 전체 탐지, 예: [0, 2] → 사람·자동차만
MODEL_PATH = 'yolov8n.pt'
IMGSZ = 640             # 입력 해상도 (320/416/640). 낮출수록 빠르지만 정확도 하락


def compute_fps(prev_time: float) -> tuple[float, float]:
    now = time.time()
    fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
    return fps, now


def draw_overlay(frame, fps: float, num_objects: int) -> None:
    cv2.rectangle(frame, (0, 0), (220, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS : {fps:5.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, f"OBJ : {num_objects}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)


def run_detection():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # stream=True → 제너레이터 반환으로 메모리 효율 향상
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            classes=TARGET_CLASSES,
            imgsz=IMGSZ,
            stream=True,
            verbose=False,
        )

        annotated_frame = frame
        num_objects = 0

        for r in results:
            annotated_frame = r.plot()  # 바운딩 박스·라벨 렌더링
            num_objects = len(r.boxes)

            # 탐지된 각 객체의 정보 출력 (터미널)
            for box in r.boxes:
                cx, cy, bw, bh = box.xywh[0]
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"  {label:<12} conf={conf:.2f}  center=({cx:.0f},{cy:.0f})  size={bw:.0f}x{bh:.0f}")

        fps, prev_time = compute_fps(prev_time)
        draw_overlay(annotated_frame, fps, num_objects)

        cv2.imshow("YOLOv8 Real-time Detection  [q to quit]", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
