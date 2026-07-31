import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_PATH = "model"

_model = None
_processor = None

def load_model():
    global _model, _processor
    if _model is None:
      _model = CLIPModel.from_pretrained(MODEL_PATH)
      _processor = CLIPProcessor.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=True)
      _model.eval()
    return _model, _processor