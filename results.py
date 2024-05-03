import torch
from torch.nn import functional as F

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
    gallery_tensor = torch.stack(gallery)

    # Normalize the query and gallery tensors to unit form
    query_normalized = F.normalize(query, p=2, dim=1)
    gallery_normalized = F.normalize(gallery_tensor, p=2, dim=1)
    # print("query_normalized shape:", query_normalized.shape)
    # print("gallery_normalized shape:", gallery_normalized.shape)
    # print("gallery_normalized transposed shape:", gallery_normalized.transpose(0, 1).shape)

    # Compute cosine similarity
    similarities = torch.mm(query_normalized, gallery_normalized.transpose(0, 1))

    # Get the top_k similar indices
    _, top_indices = torch.topk(similarities, top_k, largest=True, sorted=True)

    return top_indices[0].tolist()

# Example Usage:
query_tensor = torch.tensor([[2.5213e-03, -2.2885e-02, 2.3471e-02, 3.3205e-40, 1.0001e-01, 1.4585e-40]])
gallery_tensors = [
    torch.tensor([[4.3813e-03, -2.0963e-02, 2.2577e-02, 3.2988e-40, 1.2362e-01, 1.4490e-40]]),
    # Add other tensors here
]

# Find indices of the top 5 most similar tensors
top_5_indices = find_most_similar(query_tensor, gallery_tensors)
print("Indices of the most similar tensors:", top_5_indices)
