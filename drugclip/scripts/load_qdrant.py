"""
Load DrugCLIP embeddings into Qdrant.

Usage:
    uv run load-qdrant --molecules data/mols.lmdb
    uv run load-qdrant --pockets data/pocket.lmdb
    uv run load-qdrant --molecules data/mols.lmdb --pockets data/pocket.lmdb

Environment variables:
    QDRANT_HOST: Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY: API key for Qdrant Cloud (optional for local)
"""

import argparse
import os
import uuid

from typing import Dict, List

import lmdb
import pickle
import torch
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .modeling_drugclip import (
    DrugCLIPModel,
    tokenize_molecule,
    tokenize_pocket,
    to_model_input,
)


COLLECTION_NAME = "drugclip"
VECTOR_DIM = 128  # projection_dim from config


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
    Create the drugclip collection if it doesn't exist. If recreate is True, delete existing collection first.
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


def load_lmdb(path: str, desc: str = "Loading", subset: int = None) -> List[Dict]:
    """
    Load all entries from an LMDB file with progress bar.
    """
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    data_list = []
    with env.begin() as txn:
        num_entries = txn.stat()["entries"]
        for _, value in tqdm(txn.cursor(), total=num_entries, desc=desc):
            data_list.append(pickle.loads(value))
    env.close()
    return data_list[:subset] if subset is not None else data_list

def load_encoded_mols(path: str) -> List[Dict]:
    """
    Load pre-encoded molecule embeddings from pickle file.
    """
    with open(path, "rb") as f:
        data_list = pickle.load(f)
    return data_list

def encode_molecules(
    model: DrugCLIPModel,
    data_list: List[Dict],
) -> List[PointStruct]:
    """
    Encode molecules and create Qdrant points.
    """
    points = []

    for data in tqdm(data_list, desc="Encoding molecules"):
        try:
            tokenized = tokenize_molecule(data, model.config.mol_dictionary)
            inputs = to_model_input(tokenized, device=model.device)

            with torch.no_grad():
                emb = model.encode_molecule(**inputs)

            vector = emb.squeeze(0).cpu().numpy().tolist()

            # extract metadata
            # ['coordinates', 'atoms', 'smi', 'IDs', 'subset']
            payload = {
                "type": "molecule",
                "smi": data.get("smi", ""),
                "num_atoms": len(data.get("atoms", [])),
                "subset": data.get("subset", ""),
                "ids": data.get("IDs", []),
            }

            # add any other useful fields from data
            if "name" in data:
                payload["name"] = data["name"]
            if "id" in data:
                payload["source_id"] = str(data["id"])

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        except Exception as e:
            smi = data.get("smi", "unknown")[:50]
            print(f"  Failed to encode molecule {smi}: {e}")

    return points


def encode_pockets(
    model: DrugCLIPModel,
    data_list: List[Dict],
) -> List[PointStruct]:
    """
    Encode pockets and create Qdrant points.
    """
    points = []

    for data in tqdm(data_list, desc="Encoding pockets"):
        try:
            tokenized = tokenize_pocket(data, model.config.pocket_dictionary)
            inputs = to_model_input(tokenized, device=model.device)

            with torch.no_grad():
                emb = model.encode_pocket(**inputs)

            vector = emb.squeeze(0).cpu().numpy().tolist()

            # extract metadata
            payload = {
                "type": "pocket",
                "pocket_id": data.get("pocket", ""),
                "num_atoms": len(data.get("pocket_atoms", [])),
            }

            # add any other useful fields
            if "pdb" in data:
                payload["pdb"] = data["pdb"]
            if "uniprot" in data:
                payload["uniprot"] = data["uniprot"]

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        except Exception as e:
            pocket_id = data.get("pocket", "unknown")
            print(f"  Failed to encode pocket {pocket_id}: {e}")

    return points


def upload_points(client: QdrantClient, points: List[PointStruct], batch_size: int = 100):
    """
    Upload points to Qdrant in batches.
    """
    print(f"Uploading {len(points)} points to Qdrant...")

    for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"Uploaded {len(points)} points")


def main():
    parser = argparse.ArgumentParser(description="Load DrugCLIP embeddings into Qdrant")
    parser.add_argument("--molecules", type=str, help="Path to molecules LMDB file")
    parser.add_argument("--pockets", type=str, help="Path to pockets LMDB file")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt", help="Model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps). Auto-detects if not specified")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if exists")
    parser.add_argument("--subset", type=int, default=None, help="Only process this many entries from each LMDB (for testing)")
    args = parser.parse_args()

    if not args.molecules and not args.pockets:
        parser.error("At least one of --molecules or --pockets is required")

    # initialize Qdrant
    print("Connecting to Qdrant...")
    client = get_qdrant_client()
    create_collection(client, recreate=args.recreate)

    # load model
    device_str = f" on {args.device}" if args.device else ""
    print(f"Loading DrugCLIP model from {args.checkpoint}{device_str}...")
    model = DrugCLIPModel.from_checkpoint(args.checkpoint, device=args.device)

    all_points = []

    # encode molecules
    if args.molecules:
        mol_data = load_lmdb(args.molecules, desc="Loading molecules", subset=args.subset)
        print(f"Loaded {len(mol_data)} molecules")

        mol_points = encode_molecules(model, mol_data)
        all_points.extend(mol_points)

    # encode pockets
    if args.pockets:
        pocket_data = load_lmdb(args.pockets, desc="Loading pockets")
        print(f"Loaded {len(pocket_data)} pockets")

        pocket_points = encode_pockets(model, pocket_data)
        all_points.extend(pocket_points)

    # upload to Qdrant
    if all_points:
        upload_points(client, all_points)

    # print summary
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection '{COLLECTION_NAME}' now has {info.points_count} points")


if __name__ == "__main__":
    main()