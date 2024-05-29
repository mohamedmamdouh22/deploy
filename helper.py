import os
import shutil
import torch
from models.models import MBR_model
from torchvision import transforms
from PIL import Image
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

def handle_uploaded_car_images(model, image_paths, g_images, gallery):
    model.eval() 
    
    # Use ThreadPoolExecutor to handle threading
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_image, file_path, model, g_images, gallery) for file_path in image_paths]
        for future in futures:
            future.result()


