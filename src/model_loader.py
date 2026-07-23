import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_PATH = "model"

def load_model():
    model = CLIPModel.from_pretrained(MODEL_PATH)
    processor = CLIPProcessor.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=True)
    model.eval()
    return model, processor