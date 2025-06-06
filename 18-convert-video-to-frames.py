import cv2
import os

video_path = './data/videos/style-transfer/input512.mp4'
output_dir = './data/videos/style-transfer/frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_cnt = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break;
    cv2.imwrite(f'{output_dir}/frame_{frame_cnt:05d}.jpg', frame)
    frame_cnt += 1

cap.release()