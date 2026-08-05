from pathlib import Path
import os
import time
import io
from contextlib import asynccontextmanager

import uuid
import torch
import faiss
from PIL import Image
from fastapi import FastAPI, Request, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model
from .moondream_loader import load_moondream

INDEX_PATH = "outputs/faiss.index"
EMBEDDINGS_PATH = "outputs/embeddings.pt"
IMAGES_DIR = Path.home() / "Desktop" / "archive" / "Images"
BASE_URL = "http://127.0.0.1:8000"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Model and index loading")
    start = time.perf_counter()

    model, processor = load_model()
    moondream = load_moondream()

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
    app.state.moondream = moondream
    app.state.embeddings = embeddings

    print(f"Ready ({time.perf_counter() - start:.2f}s)")

    yield

    app.state.model = None
    app.state.processor = None
    app.state.index = None
    app.state.filenames = None
    app.state.moondream = None
    app.state.embeddings = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    filename: str
    score: float
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


def get_state(request: Request):
    state = request.app.state
    return state.model, state.processor, state.index, state.filenames


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/download/{filename}")
def download_image(filename: str):
    return FileResponse(IMAGES_DIR / filename, filename=filename)


@app.get("/describe/{filename}")
def describe_image(filename: str, request: Request):
    moondream = request.app.state.moondream

    image = Image.open(IMAGES_DIR / filename).convert("RGB")
    image.thumbnail((768, 768))

    caption = moondream.caption(image, length="short")["caption"]
    return {"caption": caption}


@app.get("/ask/{filename}")
def ask_image(filename: str, request: Request, q: str = Query(..., min_length=1)):
    moondream = request.app.state.moondream

    image = Image.open(IMAGES_DIR / filename).convert("RGB")
    image.thumbnail((768, 768))

    answer = moondream.query(image, q)["answer"]
    return {"answer": answer}


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request):
    model, processor, index, filenames = get_state(request)

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
            image_url=f"{BASE_URL}/images/{filenames[idx]}",
        )
        for idx, score in zip(indices[0], scores[0])
    ]

    return SearchResponse(results=results)


@app.get("/search-image")
def search_image(request: Request, q: str = Query(..., min_length=1)):
    model, processor, index, filenames = get_state(request)

    text_inputs = processor(text=[q], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.numpy().astype("float32")

    scores, indices = index.search(text_features, 6)

    results = [
        SearchResult(
            filename=filenames[idx],
            score=float(score),
            image_url=f"{BASE_URL}/images/{filenames[idx]}",
        )
        for idx, score in zip(indices[0], scores[0])
    ]

    return SearchResponse(results=results)


@app.post("/search-by-image", response_model=SearchResponse)
async def search_by_image(request: Request, file: UploadFile = File(...)):
    model, processor, index, filenames = get_state(request)
    state = request.app.state

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        raw_features = model.get_image_features(**image_inputs)  # (1, dim), unnormalized

    normalized = raw_features / raw_features.norm(dim=1, keepdim=True)
    query_vec = normalized.numpy().astype("float32")

    scores, indices = index.search(query_vec, 6)

    results = [
        SearchResult(
            filename=filenames[idx],
            score=float(score),
            image_url=f"{BASE_URL}/images/{filenames[idx]}",
        )
        for idx, score in zip(indices[0], scores[0])
    ]

    # add to uploaded image to the dataset
    new_filename = f"upload_{uuid.uuid4().hex}_{file.filename}"
    image.save(IMAGES_DIR / new_filename)

    state.embeddings[new_filename] = raw_features[0].detach().clone()
    state.filenames.append(new_filename)
    index.add(query_vec)

    torch.save(state.embeddings, EMBEDDINGS_PATH)
    faiss.write_index(index, INDEX_PATH)

    return SearchResponse(results=results)

 