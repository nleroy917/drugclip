"""
Load pre-encoded molecule embeddings into Qdrant.

For use with the encoded_mol_embs folder containing fold pickle files.

Usage:
    uv run load-preencoded encoded_mol_embs/6_folds
    uv run load-preencoded encoded_mol_embs/6_folds --recreate

Environment variables:
    QDRANT_HOST: Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY: API key for Qdrant Cloud (optional for local)
"""

import argparse
import os
import pickle
import uuid
from glob import glob
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


COLLECTION_NAME = "drugclip"
VECTOR_DIM = 128


def get_qdrant_client() -> QdrantClient:
    """
    Create Qdrant client from environment variables.
    """
    host = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")

    if api_key:
        return QdrantClient(url=host, api_key=api_key)
    else:
        return QdrantClient(url=host)


def create_collection(client: QdrantClient, recreate: bool = False):
    """
    Create the drugclip collection if it doesn't exist.
    """
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        if recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists")
            return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="type",
        field_schema="keyword",
    )
    print(f"Created collection with {VECTOR_DIM}-dim vectors, cosine distance")


def load_fold(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a fold pickle file.

    Returns:
        embeddings: (N, D) array of embeddings
        entries: list of "ID,SMILES" strings
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    embeddings = data[0]
    entries = data[1]

    # convert to numpy if tensor
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()

    return embeddings, entries


def parse_entry(entry: str) -> Tuple[str, str]:
    """
    Parse an entry string "ID,SMILES" into components.
    """
    # split on first comma only (SMILES can contain commas in rare cases)
    parts = entry.split(",", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return entry, ""


def create_points_from_fold(
    embeddings: np.ndarray,
    entries: List[str],
    fold_name: str,
) -> List[PointStruct]:
    """
    Create Qdrant points from a fold's embeddings and entries.
    """
    points = []

    for emb, entry in zip(embeddings, entries):
        mol_id, smi = parse_entry(entry)

        payload = {
            "type": "molecule",
            "smi": smi,
            "source_id": mol_id,
            "fold": fold_name,
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload=payload,
        )
        points.append(point)

    return points


def upload_points(client: QdrantClient, points: List[PointStruct], batch_size: int = 1000):
    """
    Upload points to Qdrant in batches.
    """
    for i in tqdm(range(0, len(points), batch_size), desc="Uploading", leave=False):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


def main():
    parser = argparse.ArgumentParser(
        description="Load pre-encoded molecule embeddings into Qdrant"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to directory containing fold pickle files, or a single pickle file",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for uploads (default: 1000)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only load this many molecules per fold (for testing)",
    )
    args = parser.parse_args()

    # find pickle files
    if os.path.isdir(args.path):
        pkl_files = sorted(glob(os.path.join(args.path, "*.pkl")))
        if not pkl_files:
            print(f"No .pkl files found in {args.path}")
            return
    else:
        pkl_files = [args.path]

    print(f"Found {len(pkl_files)} pickle file(s)")

    # connect to Qdrant
    print("Connecting to Qdrant...")
    client = get_qdrant_client()
    create_collection(client, recreate=args.recreate)

    # process each fold
    total_loaded = 0
    for pkl_path in pkl_files:
        fold_name = os.path.basename(pkl_path).replace(".pkl", "")
        print(f"\nLoading {fold_name}...")

        embeddings, entries = load_fold(pkl_path)
        print(f"  {len(entries)} molecules, embedding dim {embeddings.shape[1]}")

        # apply subset limit
        if args.subset:
            embeddings = embeddings[: args.subset]
            entries = entries[: args.subset]
            print(f"  (subset: {len(entries)} molecules)")

        points = create_points_from_fold(embeddings, entries, fold_name)
        upload_points(client, points, batch_size=args.batch_size)

        total_loaded += len(points)
        print(f"  Uploaded {len(points)} points")

    # print summary
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nTotal loaded: {total_loaded} molecules")
    print(f"Collection '{COLLECTION_NAME}' now has {info.points_count} points")


if __name__ == "__main__":
    main()