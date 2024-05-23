# import libraries
import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

def process_video(video_path, top_left, bottom_right):
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

        # Draw the rectangle on the frame
        x1, y1 = top_left
        x2, y2 = bottom_right
        # cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Create a mask and apply it to the frame
        mask = np.zeros_like(frame)
        mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        # Detect cars in the frame
        detections = model(mask)
        for detection in detections[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = detection[:6]
            if int(cls) == 2:  # Class 2 is 'car' in COCO dataset
                car_counter += 1
                car = frame[int(y1):int(y2), int(x1):int(x2)]
                car_resized=cv2.resize(car,(256,256))
                car_filename = os.path.join(car_folder_path, f'car_{car_counter:04d}.jpg')
                cv2.imwrite(car_filename, car_resized)
                results.append({"frame": i, "car_number": car_counter, "path": car_filename})
    #     # Display the frame with the mask rectangle
    #     cv2.imshow('Video with Mask', frame)

    #     # Break the loop if 'q' is pressed
    #     if cv2.waitKey(30) & 0xFF == ord('q'):
    #         break
    
    # cap.release()
    # cv2.destroyAllWindows()
    # return results
    cap.release()
    return results


top_left = (75, 200)  # Replace with your top-left coordinates
bottom_right = (1205, 600)  # Replace with your bottom-right coordinates
# Example usage
process_video('127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4', top_left, bottom_right)
