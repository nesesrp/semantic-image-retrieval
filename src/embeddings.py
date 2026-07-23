import csv
from pathlib import Path
import torch
from PIL import Image
from model_loader import load_model

DATASET_DIR = Path.home() / "Desktop" / "archive"
IMAGES_DIR = DATASET_DIR / "Images"
CAPTIONS_FILE = DATASET_DIR / "captions.txt"


## Empty list to store the names of images to be processed.
filenames = []
seen =set()
with open(CAPTIONS_FILE, newline="" , encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["image"] not in seen:
            seen.add(row["image"])
            filenames.append(row["image"])


#upload the model
model, processor = load_model()

#empty dict
embeddings = {} 
with torch.no_grad():
    for i, name in enumerate(filenames):
        img = Image.open(IMAGES_DIR / name).convert("RGB")
        inputs = processor(images=img, return_tensors="pt") ## Preprocess the image for the CLIP model.
        features = model.get_image_features(**inputs)

        # Remove the batch dimension: [1, 512] -> [512]
        embeddings[name] = features.squeeze(0)

        if(i+1)% 100 == 0:
            print(f"{i+1}/{len(filenames)} Processed ")

example_shape = next(iter(embeddings.values())).shape
print("Embeddings Size: ", example_shape)
print("Total images processed: ", len(embeddings))

torch.save(embeddings, "outputs/embeddings.pt")
print("Saved: outputs/embeddings.pt")
