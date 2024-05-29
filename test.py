# import libraries
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import torch

# Load the YOLOv10s model
model = YOLO("yolov8s.pt")

def process_video(video_path, top_left, bottom_right, skip_frames=2, min_width=50, min_height=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    gallery_folder_path = os.path.join('static/uploads/gallery', video_name)
    os.makedirs(gallery_folder_path, exist_ok=True)
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
         
        # Skip frames
        if i % skip_frames != 0:
            continue

        x1_roi, y1_roi = top_left
        x2_roi, y2_roi = bottom_right
        mask = np.zeros_like(frame)
        mask[y1_roi:y2_roi, x1_roi:x2_roi] = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        
        detections = model(mask)
        for detection in detections[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = detection[:6]
            if int(cls) == 2:
                width = x2 - x1
                height = y2 - y1
                # Check if the car is fully within the ROI
                if width >= min_width and height >= min_height:
                    car = frame[int(y1):int(y2), int(x1):int(x2)]
                    timestamp = i / fps  # Calculate the timestamp
                    car_filename = os.path.join(gallery_folder_path, f'{timestamp:.2f}.jpg')
                    cv2.imwrite(car_filename, car)
                    results.append({"frame": i, "path": car_filename, "timestamp": timestamp})
        
    cap.release()
    return results


# top_left = (75, 200)  # Replace with your top-left coordinates
# bottom_right = (1205, 600)  # Replace with your bottom-right coordinates

# # Example usage
# start_time = time.time()
# process_video('sample_video.mp4', top_left, bottom_right, skip_frames=2, min_width=60, min_height=90)
# end_time = time.time()

# detection_time = end_time - start_time
# print(f"Detection process time is: {detection_time} s")
