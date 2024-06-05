import torch
import torch.nn.functional as F
from ultralytics import YOLO
from models.models import MBR_model

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