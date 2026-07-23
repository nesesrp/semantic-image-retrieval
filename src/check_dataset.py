import csv 
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATASET_DIR = Path.home() / "Desktop" / "archive"
IMAGES_DIR = DATASET_DIR / "Images"
CAPTIONS_FILE = DATASET_DIR / "captions.txt"

#read caption.txt
caption_map = {}
with open(CAPTIONS_FILE, newline="" , encoding="utf8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption_map.setdefault(row["image"], []).append(row["caption"])

# select n random images
sample_names = random.sample(list(caption_map.keys()), 10)

#create a grid
fig, axes = plt.subplots(2,5, figsize=(20,8))

for ax, name in zip(axes.flatten(), sample_names):
    img = Image.open(IMAGES_DIR / name)
    ax.imshow(img)
    ax.set_title(caption_map[name][0], fontsize=8, wrap=True)
    ax.axis("off")


plt.tight_layout()
plt.savefig("outputs/dataset_check.png", dpi=150)
print("Kaydedildi: outputs/dataset_check.png")

