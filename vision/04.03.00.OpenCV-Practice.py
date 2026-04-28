import cv2
import numpy as np

def edge_detection_pipeline():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 창 생성 및 위치 지정 (3×2 격자)
    #   [ Original  | Grayscale | Red Mask ]
    #   [   Edge    | Red Edge  | Controls ]
    GAP = 20
    W, H = 640, 480

    for name in ('Original', 'Grayscale', 'Red Mask', 'Edge', 'Red Edge', 'Controls'):
        cv2.namedWindow(name)

    cv2.moveWindow('Original',   0,           0)
    cv2.moveWindow('Grayscale',  W + GAP,     0)
    cv2.moveWindow('Red Mask',   2*(W + GAP), 0)
    cv2.moveWindow('Edge',       0,           H + GAP)
    cv2.moveWindow('Red Edge',   W + GAP,     H + GAP)
    cv2.moveWindow('Controls',   2*(W + GAP), H + GAP)

    # Canny 임계값
    cv2.createTrackbar('Low Threshold',  'Controls', 100, 255, lambda _: None)
    cv2.createTrackbar('High Threshold', 'Controls', 200, 255, lambda _: None)
    # ③ 빨간색 HSV 채도·명도 하한 (실시간 조명 적응)
    cv2.createTrackbar('S Min', 'Controls',  80, 255, lambda _: None)
    cv2.createTrackbar('V Min', 'Controls',  50, 255, lambda _: None)

    # ② 모폴로지 커널 (루프 밖에서 한 번만 생성)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar 값 읽기
        low   = cv2.getTrackbarPos('Low Threshold',  'Controls')
        high  = cv2.getTrackbarPos('High Threshold', 'Controls')
        s_min = cv2.getTrackbarPos('S Min', 'Controls')
        v_min = cv2.getTrackbarPos('V Min', 'Controls')

        # 일반 엣지 검출
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blurred, low, high)

        # ① 이중 마스크: 빨간색은 Hue 0~10 + 170~180 두 구간
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([  0, s_min, v_min]), np.array([ 10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        # ② 모폴로지 정제: 열기(노이즈 제거) → 닫기(구멍 메우기)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 마스킹된 영역에만 Canny 적용
        masked   = cv2.bitwise_and(gray, gray, mask=red_mask)
        red_edge = cv2.Canny(masked, low, high)

        cv2.imshow('Original',  frame)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Red Mask',  red_mask)
        cv2.imshow('Edge',      edge)
        cv2.imshow('Red Edge',  red_edge)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

edge_detection_pipeline()
