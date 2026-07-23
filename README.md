# flickr8k-clip

A small text-to-image search project built on the Flickr8k dataset using CLIP embeddings. Images are encoded into vectors with CLIP, indexed with FAISS, and the closest images to a text query are retrieved.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`model/` is a symlink pointing to a local CLIP model (`clip-vit-base-patch32`). Update this link to your own model path, or download the model from Hugging Face and place it at the same location.

## Dataset

Place the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset's `Images/` folder and `captions.txt` file under `~/Desktop/archive/` (the code uses this path by default).

## Usage

```bash
# Visualize random samples from the dataset with their captions
python3 src/check_dataset.py

# Compute and save CLIP embeddings for all images
python3 src/embeddings.py

# Search for the most similar images to a text query (FAISS)
python3 src/search.py
```

## Outputs

- `outputs/embeddings.pt` — image filename → CLIP embedding vector
- `outputs/faiss.index` — FAISS index used for search
- `outputs/dataset_check.png` — dataset preview grid
- `outputs/search_result.png` — best match for the last search query
