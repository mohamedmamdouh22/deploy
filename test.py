import time
from utils import *

# example
top_left = (75, 200)  # Replace with your top-left coordinates
bottom_right = (1205, 600)  # Replace with your bottom-right coordinates
skip_frames, min_width, min_height, batch_size = 3 , 50 , 80 , 32
video_path = 'sample_video.mp4'

model, yolo = load_models()

frames = extract_frames(video_path, 3)
print("len frames: ", len(frames))

start_time = time.time()
emb = detect_objects(yolo, frames, top_left, bottom_right, min_width=50, min_height=80, batch_size=16)
end_time = time.time()

print("detection time is: ",end_time - start_time)


