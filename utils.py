# import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from models.models import MBR_model
import numpy as np
from PIL import Image
import os
from globals import processing_status, data_transform

def load_models():
    # set the device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the Reid model
    model_path = "models/best_mAP.pt"
    model = MBR_model(13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses="LBS", LAI=False)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)

    # Load the YOLOv8 model
    yolo = YOLO("yolov8n.pt")

    # Move the models to the device
    model.to(device)
    yolo.to(device)
    print(f"Models are loaded in device: {device}")

    model.eval()
    return model, yolo

# def extract_frames(video_path, skip_frames=2):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
#     frames = []
#     timestamps = []
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % skip_frames == 0:
#             frames.append(frame)
#             timestamp = int(frame_count / fps)  # Calculate the timestamp
#             timestamps.append(timestamp)
#         frame_count += 1
#     cap.release()
#     return frames, timestamps


# def detect_objects(model, frames, timestamps, top_left, bottom_right, min_width=50, min_height=80,save_path='static/uploads/gallery', batch_size=16):

#     objects = []
#     car_paths = []

#     # Check the device of the model
#     # device = next(model.parameters()).device

#     # define region of interest
#     x1_roi, y1_roi = top_left
#     x2_roi, y2_roi = bottom_right
#     os.makedirs(save_path, exist_ok=True)
#     car_counter = 0

#     for i in range(0, len(frames), batch_size):
#         batch_frames = frames[i:i + batch_size]
#         batch_masks = []
#         for frame in batch_frames:
#             # Define the mask
#             mask = np.zeros_like(frame)
#             mask[y1_roi:y2_roi, x1_roi:x2_roi] = frame[y1_roi:y2_roi, x1_roi:x2_roi]
#             batch_masks.append(mask)

#         # Convert batch masks to numpy arrays
#         # batch_masks_np = np.array(batch_masks)

#         # Perform YOLO detection on the batch
#         detections = model(batch_masks)
            
#         for batch_idx, detection_set in enumerate(detections):
#             frame_idx = i + batch_idx
#             frame = frames[frame_idx]
#             time_stamp = timestamps[frame_idx]
            
#             # Handle each car detection in the frame
#             for detection in detection_set.boxes.data:
#                 x1, y1, x2, y2, confidence, cls = detection[:6]
#                 # check if the detected object is car
#                 if int(cls) == 2:
#                     width = x2 - x1
#                     height = y2 - y1

#                     # Check if detected car is full car not parts of the cars
#                     if width >= min_width and height >= min_height:
#                         car = frame[int(y1):int(y2), int(x1):int(x2)]                    
#                         car_path = os.path.join(save_path, f'car_{car_counter}_{time_stamp}.jpg')
#                         cv2.imwrite(car_path, car)
#                         car_counter += 1
#                         car_paths.append(car_path)
#                         objects.append(car)
#     return objects, car_paths

# def preprocess_images(batch, transform=data_transform):
#     img_tensors = []
#     for idx,img in enumerate(batch):
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format
#         img_pil = Image.fromarray(img)  # Convert NumPy array to PIL Image
#         img_tensor = transform(img_pil)
#         img_tensors.append(img_tensor)
#     tensors_batch = torch.stack(img_tensors)  # Create a batch
#     return tensors_batch

# def cars_embeddings(model, images, images_paths, batch_size=32):
#     feature_vector_imgs = []
#     db = {}
#     num_images = len(images)
    
#     # for i in range(num_images):
#     #     car_pil = Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#     #     car_path = os.path.join('static/uploads/gallery', f'car_{i}.jpg')
#     #     car_pil.save(car_path)

#     # Check the device of the model
#     device = next(model.parameters()).device

#     # Process images in batches
#     for i in range(0, num_images, batch_size):
#         batch = images[i:i+batch_size]
        
#         # preprocess the batch 
#         tensors_batch = preprocess_images(batch).to(device)

#         # Perform batched inference
#         model.eval()
#         with torch.no_grad():
#             predictions = model(tensors_batch)
        
#         # Process the predictions
#         ffs_batch = predictions[2]      

#         for iter in range(len(batch)):
#             end_vec = [F.normalize(item[iter], dim=0) for item in ffs_batch]
#             concatenated_vec = torch.cat(end_vec, 0)
#             feature_vector_imgs.append(concatenated_vec)
#             db.update({images_paths[i+iter]: concatenated_vec})
    
#     return feature_vector_imgs, db
            
# def video_embeddings(video_path, model, yolo, top_left, bottom_right, skip_frames=3, min_width=50, min_height=80, yolo_batch_size=32, model_batch_size=32):

#     # extract frames
#     frames, timestamps = extract_frames(video_path, skip_frames)

#     # extract detections
#     cars, cars_paths = detect_objects(yolo, frames, timestamps, top_left, bottom_right, min_width, min_height, batch_size=yolo_batch_size)


#     # extract cars embeddings
#     embeddings = cars_embeddings(model, cars, cars_paths, batch_size=model_batch_size)
#     processing_status['status'] = 'done'

#     return embeddings

def find_most_similar(query, gallery, top_k=5):
    """
    Find the most similar tensors in the gallery to the query tensor using cosine similarity.

    Parameters:
    - query (torch.Tensor): A 1xN tensor representing the query image features.
    - gallery (list of torch.Tensor): A list of 1xN tensors representing the gallery image features.
    - top_k (int): The number of top similar items to return.

    Returns:
    - list of int: Indices of the top_k most similar tensors in the gallery.
    """
    # Convert the list of tensors to a single tensor
    gallery = [item.unsqueeze(0) if item.dim() == 1 else item for item in gallery]
    gallery_tensor = torch.stack(gallery)
    # Normalize the query and gallery tensors to unit form
    query_normalized = F.normalize(query, p=2, dim=1)
    gallery_normalized = F.normalize(gallery_tensor, p=2, dim=1).squeeze(1)
    # Compute cosine similarity
    similarities = torch.mm(query_normalized, gallery_normalized.transpose(0, 1))
    # Get the top_k similar indices
    top_scores, top_indices = torch.topk(similarities, top_k, largest=True, sorted=True)
    # Ensure scores are between 0 and 1
    top_scores = [(score.item() + 1) / 2 for score in top_scores]
    print(top_scores[0])
    if top_scores[0] < 5:
        return None,None
    return top_indices, top_scores