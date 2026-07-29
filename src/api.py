from pathlib import Path
import os
import time
from contextlib import asynccontextmanager

import torch
import faiss
from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model

INDEX_PATH = "outputs/faiss.index"
EMBEDDINGS_PATH = "outputs/embeddings.pt"
IMAGES_DIR = Path.home() / "Desktop" / "archive" / "Images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Model and index loading")
    start = time.perf_counter()

    model, processor = load_model()

    embeddings = torch.load(EMBEDDINGS_PATH)
    filenames = list(embeddings.keys())
    image_matrix = torch.stack(list(embeddings.values()))
    image_matrix = image_matrix / image_matrix.norm(dim=1, keepdim=True)
    image_matrix = image_matrix.numpy().astype("float32")

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(image_matrix.shape[1])
        index.add(image_matrix)
        faiss.write_index(index, INDEX_PATH)

    app.state.model = model
    app.state.processor = processor
    app.state.index = index
    app.state.filenames = filenames

    print(f"Ready ({time.perf_counter() - start:.2f}s)")

    yield

    app.state.model = None
    app.state.processor = None
    app.state.index = None
    app.state.filenames = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    filename: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request):
    model = request.app.state.model
    processor = request.app.state.processor
    index = request.app.state.index
    filenames = request.app.state.filenames

    text_inputs = processor(text=[payload.query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.numpy().astype("float32")

    scores, indices = index.search(text_features, payload.k)

    results = [
        SearchResult(
            filename=filenames[idx],
            score=float(score),
            image_url=f"/images/{filenames[idx]}",
        )
        for idx, score in zip(indices[0], scores[0])
    ]

    return SearchResponse(results=results)


@app.get("/search-image")
def search_image(request: Request, q: str = Query(..., min_length=1)):
    model = request.app.state.model
    processor = request.app.state.processor
    index = request.app.state.index
    filenames = request.app.state.filenames

    text_inputs = processor(text=[q], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.numpy().astype("float32")

    scores, indices = index.search(text_features, 1)

    best_filename = filenames[indices[0][0]]
    return RedirectResponse(url=f"/images/{best_filename}")