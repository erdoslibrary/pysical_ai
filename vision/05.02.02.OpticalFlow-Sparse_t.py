import cv2
import numpy as np

# 특징점 검출 파라미터 (Shi-Tomasi)
FEATURE_PARAMS = dict(
    maxCorners=150,    # 최대 추적 점 수
    qualityLevel=0.3,  # 품질 임계값 (낮을수록 점이 많아짐)
    minDistance=7,     # 점 간 최소 거리
    blockSize=7
)

# Lucas-Kanade Optical Flow 파라미터
LK_PARAMS = dict(
    winSize=(15, 15),   # 탐색 윈도우 크기
    maxLevel=2,         # 피라미드 레벨
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 추적 점마다 고유 색상
COLORS = np.random.randint(0, 255, size=(500, 3), dtype=np.uint8)

def sparse_optical_flow():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    # 궤적을 그릴 빈 캔버스
    trail_mask = np.zeros_like(prev_frame)
    # 각 점에 인덱스 색상 할당
    pt_colors = [COLORS[i % len(COLORS)].tolist() for i in range(len(pts))] if pts is not None else []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if pts is not None and len(pts) > 0:
            # 이전 프레임에서 찾은 점들을 현재 프레임에서 추적
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pts, None, **LK_PARAMS
            )

            # 추적 성공한 점만 필터링
            good_prev = pts[status == 1]
            good_next = next_pts[status == 1]
            good_colors = [pt_colors[i] for i, s in enumerate(status.flatten()) if s == 1]

            for i, (prev_pt, next_pt) in enumerate(zip(good_prev, good_next)):
                px, py = map(int, prev_pt.ravel())
                nx, ny = map(int, next_pt.ravel())
                color = good_colors[i]

                # 궤적 선
                cv2.line(trail_mask, (px, py), (nx, ny), color, 2)
                # 현재 위치 점
                cv2.circle(frame, (nx, ny), 4, color, -1)

            output = cv2.add(frame, trail_mask)
            pts = good_next.reshape(-1, 1, 2)
            pt_colors = good_colors
        else:
            # 추적할 점이 없으면 재검출
            output = frame
            trail_mask = np.zeros_like(frame)

        # 점이 너무 적어지면 재검출
        if pts is None or len(pts) < 10:
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(frame)
            pt_colors = [COLORS[i % len(COLORS)].tolist() for i in range(len(pts))] if pts is not None else []

        cv2.imshow('Sparse Optical Flow', output)
        prev_gray = gray.copy()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

sparse_optical_flow()
