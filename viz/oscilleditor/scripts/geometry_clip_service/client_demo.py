"""
Utility script to exercise the geometry-aware CLIP service endpoints.

Example commands:
    python client_demo.py embed-text "harmonious spiral domains"
    python client_demo.py embed-image assets/dog.jpg
    python client_demo.py recommend "gentle symmetry" presets/sample_presets.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
from PIL import Image


DEFAULT_ENDPOINT = os.getenv("GEOMETRY_CLIP_ENDPOINT", "http://localhost:8001")


def encode_image_to_base64(path: Path) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        data = buffer.getvalue()
    return base64.b64encode(data).decode("utf-8")


def embed_text(text: str) -> None:
    url = f"{DEFAULT_ENDPOINT}/embed/text"
    resp = requests.post(url, json={"text": text})
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def embed_image(path: Path) -> None:
    url = f"{DEFAULT_ENDPOINT}/embed/image"
    image_b64 = encode_image_to_base64(path)
    resp = requests.post(url, json={"image_b64": image_b64})
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def recommend(query: str, presets_path: Path, top_k: int) -> None:
    items = json.loads(presets_path.read_text())
    url = f"{DEFAULT_ENDPOINT}/recommend"
    payload: Dict[str, Any] = {
        "query_text": query,
        "items": items,
        "top_k": top_k,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIP service demo client")
    sub = parser.add_subparsers(dest="command", required=True)

    text_parser = sub.add_parser("embed-text", help="Embed a text prompt")
    text_parser.add_argument("text", help="Text prompt")

    img_parser = sub.add_parser("embed-image", help="Embed an image file")
    img_parser.add_argument("path", type=Path, help="Path to the image file")

    rec_parser = sub.add_parser("recommend", help="Get top-k recommendations")
    rec_parser.add_argument("query", help="Query text prompt")
    rec_parser.add_argument("presets", type=Path, help="JSON file containing candidate items")
    rec_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "embed-text":
        embed_text(args.text)
    elif args.command == "embed-image":
        embed_image(args.path)
    elif args.command == "recommend":
        recommend(args.query, args.presets, args.top_k)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()


