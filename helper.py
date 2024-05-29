import os
import shutil
import torch
from models.models import MBR_model
from torchvision import transforms
from PIL import Image
from test import process_video
import torch.nn.functional as F
import time
from concurrent.futures import ThreadPoolExecutor
from globals import *
from globals import x_length,y_length,processing_status,n_mean,n_std

# this function checks the extension of the file passed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "mp4",
        "avi",
    }

def clear_query_directory(parent_dir):
    # query_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'query')
    query_folder = os.path.join(parent_dir, 'query')
    for filename in os.listdir(query_folder):
        file_path = os.path.join(query_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def load_model():
    model_path = "models/best_mAP.pt"
    model = MBR_model(
    13164, ["R50", "R50", "BoT", "BoT"], n_groups=0, losses="LBS", LAI=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)

    # Move the model to the device
    model.to(device)

    model.eval()
    return model
# ===================================================================================
# ===================================================================================
def process_image(file_path, model, g_images: list, gallery: dict):
    print(file_path)
    device = next(model.parameters()).device
    if allowed_file(file_path):
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            test_transform = transforms.Compose(
                [
                    transforms.Resize((y_length, x_length), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(n_mean, n_std),
                ]
            )
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                prediction = model(img_tensor)

            ffs = prediction[2]
            end_vec = [F.normalize(item) for item in ffs]

            g_images.append(torch.cat(end_vec, 1))
            gallery.update({f"{file_path}": torch.cat(end_vec, 1)})

def handle_uploaded_car_images(image_paths, g_images, gallery):
    model = load_model()
    model.eval()  
    
    # Use ThreadPoolExecutor to handle threading
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_image, file_path, model, g_images, gallery) for file_path in image_paths]
        for future in futures:
            future.result()

def process_video_and_handle_images(video_path, top_left, bottom_right):

    results = process_video(video_path, top_left, bottom_right)
    car_image_paths = [result['path'] for result in results]
    start_time = time.time()
    handle_uploaded_car_images(car_image_paths)
    end_time = time.time()
    model_time=end_time-start_time
    print(f"model time is: {model_time} s")
    # Set processing status to 'done'
    processing_status['status'] = 'done'


