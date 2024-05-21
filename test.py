# from models.models import MBR_model
# from PIL import Image
# from torchvision import transforms
# import numpy as np
# import torch
# import pandas as pd
# model_path = 'models/best_mAP.pt'
# y_length= 256
# x_length= 256
# n_mean= [0.5, 0.5, 0.5]
# n_std= [0.5, 0.5, 0.5]
# model= MBR_model(13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses ="LBS", LAI=False)
# model_path = 'models/best_mAP.pt'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_state_dict = torch.load(model_path, map_location=device)
# data={}
# current_model=[]
# print("Current model keys:")
# for k in model.state_dict().keys():
#     current_model.append(k)
# data['current mode']=current_model

# loaded_model=[]
# # Print loaded state dict keys
# print("\nLoaded state dict keys:")
# for k in model_state_dict.keys():
#     loaded_model.append(k)
# data['loaded model']=loaded_model
# # df=pd.DataFrame(data)
# # print(len(current_model),len(loaded_model))
# # df.to_csv('test.csv')
# model.load_state_dict(model_state_dict)  # Load it properly into the model instance
# print(model)
# model.eval()  # Set the model to evaluation mode
import cv2
import torch
import os
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
def process_video(video_path):
    results = []
    cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count=300
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
        for detection in detections.xyxy[0]:
            x1, y1, x2, y2, confidence, cls = detection
            if int(cls) == 2:  # Class 2 is 'car' in COCO dataset
                car_counter += 1
                car = frame[int(y1):int(y2), int(x1):int(x2)]
                car_filename = os.path.join(car_folder_path, f'car_{car_counter:04d}.jpg')
                cv2.imwrite(car_filename, car)
                results.append({"frame": i, "car_number": car_counter, "path": car_filename})
    
    cap.release()
    return results
process_video('127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4')