import cv2       # OpenCV 라이브러리 임포트
import numpy as np  # 행렬 연산을 위한 NumPy

def edge_detection_pipeline():
    # 1. 웹캠 연결 (0번은 내장 웹캠)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 창 생성 및 위치 지정 (루프 전에 한 번만)
    #   [ Original  | Grayscale ]
    #   [   Edge    | Controls  ]
    GAP = 20
    W, H = 640, 480

    cv2.namedWindow('Original')
    cv2.namedWindow('Grayscale')
    cv2.namedWindow('Edge')
    cv2.namedWindow('Controls')

    cv2.moveWindow('Original',  0,         0)
    cv2.moveWindow('Grayscale', W + GAP,   0)
    cv2.moveWindow('Edge',      0,         H + GAP)
    cv2.moveWindow('Controls',  W + GAP,   H + GAP)

    cv2.createTrackbar('Low Threshold',  'Controls', 100, 255, lambda _: None)
    cv2.createTrackbar('High Threshold', 'Controls', 200, 255, lambda _: None)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전 보정 (1: 수평 뒤집기)
        frame = cv2.flip(frame, 1)

        # Grayscale 변환 (04.01.OpenCV-Basic 참고)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Trackbar에서 임계값 읽기
        low  = cv2.getTrackbarPos('Low Threshold',  'Controls')
        high = cv2.getTrackbarPos('High Threshold', 'Controls')

        # Edge 검출: 노이즈 제거 후 Canny 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blurred, low, high)

        cv2.imshow('Original',  frame)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Edge',      edge)

        # 'q' 키를 누르면 종료 (25ms 대기)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 실행
edge_detection_pipeline()