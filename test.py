import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

def process_video(video_path):
    results = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 60  # you can set this to the actual number of frames in the video if needed
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    car_folder_path = os.path.join('cars', video_name)
    os.makedirs(car_folder_path, exist_ok=True)
    
    car_counter = 0
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect cars in the frame
        detections = model(frame)
        for detection in detections[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = detection[:6]
            if int(cls) == 2:  # Class 2 is 'car' in COCO dataset
                car_counter += 1
                car = frame[int(y1):int(y2), int(x1):int(x2)]
                car_resized=cv2.resize(car,(256,256))
                car_filename = os.path.join(car_folder_path, f'car_{car_counter:04d}.jpg')
                cv2.imwrite(car_filename, car_resized)
                results.append({"frame": i, "car_number": car_counter, "path": car_filename})
    
    cap.release()
    return results

# Example usage
process_video('127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4')
