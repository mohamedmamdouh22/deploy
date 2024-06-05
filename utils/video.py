import cv2
import numpy as np
import os

def vehicles_detection(model, video_path, save_path, top_left, bottom_right, skip_frames=5):
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
            results = model(mask, device="cuda", classes=[2], conf=0.6, iou=0.2)
            for detections in results:
                # Handle each car detection in the frame
                for detection in detections.boxes.data:
                    x1, y1, x2, y2, confidence, cls = detection[:6]
                    car_counter += 1
                    car = frame[int(y1) : int(y2), int(x1) : int(x2)]
                    car_path = os.path.join(save_path, f"car_{car_counter}_{time_stamp}.jpg")
                    cv2.imwrite(car_path, car)

    # Release the video capture object and close the display window
    cap.release()
