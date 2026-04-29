import cv2
import numpy as np
from ultralytics import YOLO

# 클래스별 고정 색상 (COCO 80클래스 대응)
COLORS = np.random.default_rng(42).uniform(0, 255, size=(80, 3)).astype(np.uint8)

def draw_segmentation(frame, r):
    output = frame.copy()

    # 마스크 오버레이
    if r.masks is not None:
        for i, mask in enumerate(r.masks.data):
            cls_id = int(r.boxes.cls[i])
            color = COLORS[cls_id % len(COLORS)].tolist()

            overlay = output.copy()
            overlay[cv2.resize(mask.cpu().numpy(), frame.shape[1::-1]) > 0.5] = color
            output = cv2.addWeighted(output, 0.6, overlay, 0.4, 0)  # 투명도 40%

    # 바운딩 박스 + 라벨
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = COLORS[cls_id % len(COLORS)].tolist()
            label = f"{r.names[cls_id]} {conf:.2f}"

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n-seg.pt')

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=True)

        for r in results:
            output = draw_segmentation(frame, r)
            cv2.imshow('YOLO Segmentation', output)

        # 'q' 키를 누르면 종료 (25ms 대기)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
edge_detection_pipeline()
