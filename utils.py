import cv2
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
    yolo = YOLO("yolov8s.pt")

    # Move the models to the device
    model.to(device)
    yolo.to(device)
    print(f"Models are loaded in device: {device}")

    model.eval()
    return model, yolo

def time_stamp():
    # fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    # video_name = os.path.splitext(os.path.basename(video_path))[0]
    # gallery_folder_path = os.path.join('static/uploads/gallery', video_name)
    # os.makedirs(gallery_folder_path, exist_ok=True)
    pass

def extract_frames(video_path, skip_frames=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames


def detect_objects(model, frames, top_left, bottom_right, min_width=50, min_height=80,save_path='static/uploads/gallery'):
    results = []
    # define region of interest
    x1_roi, y1_roi = top_left
    x2_roi, y2_roi = bottom_right
    os.makedirs(save_path, exist_ok=True)
    # car_counter = 0

    for idx,frame in enumerate(frames):
        # define the mask
        mask = np.zeros_like(frame)
        mask[y1_roi:y2_roi, x1_roi:x2_roi] = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        detections = model(mask)

        # handle each car detection in the frame
        for detection in detections[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = detection[:6]

            # check if the detected object is car
            if int(cls) == 2:
                width = x2 - x1
                height = y2 - y1

                # Check if detected car is full car not parts of the cars
                if width >= min_width and height >= min_height:
                    car = frame[int(y1):int(y2), int(x1):int(x2)]
                    # timestamp = i / fps  # Calculate the timestamp
                    # car_filename = os.path.join('test_gallery', f'{car_counter}.jpg')
                    # car_counter +=1
                    # cv2.imwrite(car_filename, car)
                    # results.append({"frame": i, "path": car_filename, "timestamp": timestamp})
                    car_pil = Image.fromarray(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))
                    car_path = os.path.join(save_path, f'car_{idx}.jpg')
                    car_pil.save(car_path)
                    # add the car frame
                    results.append(car)
    return results

def preprocess_images(batch, transform=data_transform):
    img_tensors = []
    for idx,img in enumerate(batch):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure the image is in RGB format
        img_pil = Image.fromarray(img)  # Convert NumPy array to PIL Image
        img_tensor = data_transform(img_pil)
        img_tensors.append(img_tensor)
    tensors_batch = torch.stack(img_tensors)  # Create a batch
    return tensors_batch

def cars_embeddings(model, images, batch_size=32):
    feature_vector_imgs = []
    db = {}
    num_images = len(images)
    for i in range(num_images):
        car_pil = Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        car_path = os.path.join('static/uploads/gallery', f'car_{i}.jpg')
        car_pil.save(car_path)

    # Check the device of the model
    device = next(model.parameters()).device

    # Process images in batches
    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size]
        
        # preprocess the batch 
        tensors_batch = preprocess_images(batch).to(device)

        # Perform batched inference
        model.eval()
        with torch.no_grad():
            predictions = model(tensors_batch)
        
        # Process the predictions
        ffs_batch = predictions[2]      

        for iter in range(len(batch)):
            end_vec = [F.normalize(item[iter], dim=0) for item in ffs_batch]
            concatenated_vec = torch.cat(end_vec, 0)
            feature_vector_imgs.append(concatenated_vec)
            db.update({f"static/uploads/gallery/car_{i+iter}.jpg": concatenated_vec})
    
    return feature_vector_imgs, db
            
def video_embeddings(video_path, model, yolo, top_left, bottom_right, skip_frames=2, min_width=50, min_height=80, batch_size=32):
    # extract frames
    frames = extract_frames(video_path, skip_frames)

    # extract detections
    cars = detect_objects(yolo, frames, top_left, bottom_right, min_width, min_height)

    # extract cars embeddings
    embeddings = cars_embeddings(model, cars, batch_size)
    processing_status['status'] = 'done'

    return embeddings

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
    return top_indices, top_scores


