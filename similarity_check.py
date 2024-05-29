import torch.nn.functional as F
import torch
def find_most_similar(query, gallery, top_k=5):
    gallery = [item.unsqueeze(0) if item.dim() == 1 else item for item in gallery]
    gallery_tensor = torch.stack(gallery)
    query_normalized = F.normalize(query, p=2, dim=1)
    gallery_normalized = F.normalize(gallery_tensor, p=2, dim=1).squeeze(1)
    similarities = torch.mm(query_normalized, gallery_normalized.transpose(0, 1))
    top_scores, top_indices = torch.topk(similarities, top_k, largest=True, sorted=True)
    
    # Ensure scores are between 0 and 1
    top_scores = [(score.item() + 1) / 2 for score in top_scores]
    
    return top_indices, top_scores


