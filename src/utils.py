import torch

def cosine_similarity(a,b):
    a = a / a.norm(dim=1, keepdim=True)
    b = b / b.norm(dim=1, keepdim=True)
    return a @ b.T