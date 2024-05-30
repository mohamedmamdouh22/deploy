import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from globals import data_transform
import torch.nn.functional as F
import torch

# Create a custom dataset class
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return both image and its path
    

def gallery_embeddings(gallery_path, model, g_images: list, gallery: dict, batch_size=32):

    # define the Dataloader
    dataset = ImageFolderDataset(gallery_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  
    
    # Check the device of the model
    device = next(model.parameters()).device

    for batch in dataloader:

        images, paths = batch
        images = images.to(device)

        # model inference
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        
        # Process the predictions
        ffs_batch = predictions[2]
        for iter in range(len(images)):
                end_vec = [F.normalize(item[iter], dim=0) for item in ffs_batch]
                concatenated_vec = torch.cat(end_vec, 0)
                g_images.append(concatenated_vec)
                gallery.update({f"{paths[iter]}": concatenated_vec})  


