# import libraries
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

def process_video(video_path, top_left, bottom_right, skip_frames=2):

    results = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    gallery_folder_path = os.path.join('static/uploads/gallery', video_name)
    os.makedirs(gallery_folder_path, exist_ok=True)
    
    car_counter = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
         
        # Skip frames
        if i % skip_frames != 0:
            continue

        x1, y1 = top_left
        x2, y2 = bottom_right
        mask = np.zeros_like(frame)
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        detections = model(mask)
        for detection in detections[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = detection[:6]
            if int(cls) == 2:
                car_counter += 1
                car = frame[int(y1):int(y2), int(x1):int(x2)]
                car_filename = os.path.join(gallery_folder_path, f'car_{car_counter:04d}.jpg')
                cv2.imwrite(car_filename, car)
                results.append({"frame": i, "car_number": car_counter, "path": car_filename})
    
    cap.release()
    return results


top_left = (75, 200)  # Replace with your top-left coordinates
bottom_right = (1205, 600)  # Replace with your bottom-right coordinates

# Example usage
start_time = time.time()
process_video('127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4', top_left, bottom_right, 2)
end_time = time.time()

detection_time = end_time - start_time
print(f"Detection process time is: {detection_time} s")
