import cv2
from ultralytics import YOLO
import numpy as np
import os
import time

def vehicles_detection(
    model,
    video_path,
    save_path,
    top_left,
    bottom_right,
    skip_frames=3,
    min_width=50,
    min_height=80,
):
    # define region of interest
    x1_roi, y1_roi = top_left
    x2_roi, y2_roi = bottom_right

    car_counter = 1
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        if frame_count % skip_frames == 0:
            time_stamp = int(frame_count / fps)

            # Define the mask
            mask = np.zeros_like(frame)
            mask[y1_roi:y2_roi, x1_roi:x2_roi] = frame[y1_roi:y2_roi, x1_roi:x2_roi]

            # Run YOLOv8 inference on the frame
            results = model(mask, device="cuda", classes=[2])
            for detections in results:
                # Handle each car detection in the frame
                for detection in detections.boxes.data:
                    x1, y1, x2, y2, confidence, cls = detection[:6]
                    width = x2 - x1
                    height = y2 - y1

                    # Check if detected car is full car not parts of the cars
                    if width >= min_width and height >= min_height:
                        car_counter += 1
                        car = frame[int(y1) : int(y2), int(x1) : int(x2)]
                        car_path = os.path.join(
                            save_path, f"car_{car_counter}_{time_stamp}.jpg"
                        )
                        cv2.imwrite(car_path, car)

    # Release the video capture object and close the display window
    cap.release()


# import time

# save_path = "static/uploads"     # change this
# video_path = 'Untitled_Project_V1.mp4'    # change this

# top_left = (75, 200)  # Replace with your top-left coordinates
# bottom_right = (1205, 600)  # Replace with your bottom-right coordinates

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")

# start = time.time()
# vehicles_detection(model, video_path, save_path, top_left, bottom_right)
# end = time.time()
# print(f"Detection Time is: {end - start}")