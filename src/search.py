import os
import time
import torch
import faiss
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from model_loader import load_model
from utils import cosine_similarity

IMAGES_DIR = Path.home() / "Desktop" / "archive" / "Images"
INDEX_PATH = "outputs/faiss.index"
embeddings = torch.load("outputs/embeddings.pt")

filenames = list(embeddings.keys())
image_matrix = torch.stack(list(embeddings.values()))
image_matrix = image_matrix / image_matrix.norm(dim=1, keepdim=True)
image_matrix = image_matrix.numpy().astype("float32")
model, processor = load_model()

index_start = time.perf_counter()
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatIP(image_matrix.shape[1]) #embedding size
    index.add(image_matrix)
    faiss.write_index(index, INDEX_PATH)
index_time = time.perf_counter() - index_start

query = input("Search: ")
text_inputs = processor(text=[query], return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features /text_features.norm(dim=1, keepdim=True)
    text_features = text_features.numpy().astype("float32")

k = 5
search_start = time.perf_counter()
scores, indices = index.search(text_features, k)
search_time = time.perf_counter() - search_start

for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    print(f"{rank}. {filenames[idx]} (score: {score: .4f})")

print(f"Indexing time: {index_time*1000:.2f} ms")
print(f"Search time: {search_time*1000:.2f} ms")

best_idx = indices[0][0]
best_filename = filenames[best_idx]
best_score = scores[0][0]


img = Image.open(IMAGES_DIR / best_filename)
plt.imshow(img)
plt.title(f"Best match for '{query}'")
plt.axis("off")
plt.savefig("outputs/search_result.png")

print("Saved: outputs/search_result.png")

