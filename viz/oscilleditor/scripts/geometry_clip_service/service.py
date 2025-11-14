"""
FastAPI service exposing geometry-aware CLIP embeddings.

Run with:
    uvicorn service:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import base64
import os
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .encoder import EncoderConfig, GeometryAwareCLIPEncoder


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Geometry-Aware CLIP Service", version="0.1.0")

_encoder: Optional[GeometryAwareCLIPEncoder] = None


def get_encoder() -> GeometryAwareCLIPEncoder:
    global _encoder
    if _encoder is None:
        config = EncoderConfig(
            latent_dim=int(os.getenv("GEOMETRY_CLIP_LATENT_DIM", 256)),
            model_name=os.getenv("GEOMETRY_CLIP_MODEL", "openai/clip-vit-base-patch32"),
            fine_tune_path=os.getenv("GEOMETRY_CLIP_PROJECTION_PATH"),
            device=os.getenv("GEOMETRY_CLIP_DEVICE"),
        )
        _encoder = GeometryAwareCLIPEncoder(config=config)
    return _encoder


# -----------------------------------------------------------------------------
# Request/response models
# -----------------------------------------------------------------------------
class EmbedTextRequest(BaseModel):
    text: str = Field(..., description="Text prompt to embed.")


class EmbedImageRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded RGB image.")
    text: Optional[str] = Field(None, description="Optional accompanying text prompt.")


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


class RecommendItem(BaseModel):
    id: str
    embedding: Optional[List[float]] = None
    image_b64: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[dict] = None


class RecommendRequest(BaseModel):
    query_text: Optional[str] = None
    query_image_b64: Optional[str] = None
    query_text_for_image: Optional[str] = None
    items: List[RecommendItem]
    top_k: int = 5


class RecommendResponse(BaseModel):
    results: List[dict]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: EmbedTextRequest) -> EmbedResponse:
    encoder = get_encoder()
    vector = encoder.embed_text(req.text)
    return EmbedResponse(embedding=vector.tolist(), dim=vector.size)


@app.post("/embed/image", response_model=EmbedResponse)
def embed_image(req: EmbedImageRequest) -> EmbedResponse:
    encoder = get_encoder()
    try:
        image = encoder.decode_base64_image(req.image_b64)
    except (base64.binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    if req.text:
        vector = encoder.embed_image_text(image, req.text)
    else:
        vector = encoder.embed_image(image)
    return EmbedResponse(embedding=vector.tolist(), dim=vector.size)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if not req.items:
        raise HTTPException(status_code=400, detail="No candidate items provided.")

    encoder = get_encoder()

    # Build query embedding
    if req.query_image_b64:
        try:
            image = encoder.decode_base64_image(req.query_image_b64)
        except (base64.binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 query image: {exc}") from exc
        if req.query_text_for_image:
            query_vec = encoder.embed_image_text(image, req.query_text_for_image)
        else:
            query_vec = encoder.embed_image(image)
    elif req.query_text:
        query_vec = encoder.embed_text(req.query_text)
    else:
        raise HTTPException(status_code=400, detail="Provide query_text or query_image_b64.")

    # Prepare candidate matrix
    item_vectors = []
    for idx, item in enumerate(req.items):
        if item.embedding is not None:
            vec = np.asarray(item.embedding, dtype=np.float32)
        elif item.image_b64:
            try:
                image = encoder.decode_base64_image(item.image_b64)
            except (base64.binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid base64 in item {item.id}: {exc}") from exc
            if item.text:
                vec = encoder.embed_image_text(image, item.text)
            else:
                vec = encoder.embed_image(image)
        elif item.text:
            vec = encoder.embed_text(item.text)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Candidate item at index {idx} must provide embedding, image_b64, or text.",
            )
        item_vectors.append(vec)

    matrix = np.vstack(item_vectors)
    similarities = GeometryAwareCLIPEncoder.cosine_similarity(query_vec, matrix)
    top_k = min(req.top_k, len(req.items))
    indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(indices):
        item = req.items[idx]
        results.append(
            {
                "id": item.id,
                "score": float(similarities[idx]),
                "rank": rank + 1,
                "metadata": item.metadata or {},
            }
        )

    return RecommendResponse(results=results)


def main() -> None:
    uvicorn.run(
        "service:app",
        host=os.getenv("GEOMETRY_CLIP_HOST", "0.0.0.0"),
        port=int(os.getenv("GEOMETRY_CLIP_PORT", "8001")),
        reload=os.getenv("GEOMETRY_CLIP_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()


