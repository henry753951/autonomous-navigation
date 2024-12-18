import time

import cv2


def measure_fps(video_path: str) -> None:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame_count += 1

        # 如果需要顯示畫面，取消註解
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    print(
        f"Processed {frame_count} frames in {elapsed_time:.2f} seconds. FPS: {fps:.2f}",
    )

    capture.release()
    cv2.destroyAllWindows()


# 測試影片路徑
measure_fps("data/test_videos/1.mp4")
