"""
Query DrugCLIP embeddings in Qdrant.

Given pocket LMDB files, find the best binding molecules.

Usage:
    uv run query-qdrant --pockets data/pocket.lmdb
    uv run query-qdrant --pockets data/pocket.lmdb --top-k 20

Environment variables:
    QDRANT_HOST: Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY: API key for Qdrant Cloud (optional for local)
"""

import argparse
import os

from typing import Dict, List

import lmdb
import pickle
import torch

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .modeling_drugclip import (
    DrugCLIPModel,
    tokenize_pocket,
    to_model_input,
)


COLLECTION_NAME = "drugclip"


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


def encode_pocket(model: DrugCLIPModel, data: Dict) -> torch.Tensor:
    """
    Encode a single pocket and return its embedding vector.
    """
    tokenized = tokenize_pocket(data, model.config.pocket_dictionary)
    inputs = to_model_input(tokenized, device=model.device)

    with torch.no_grad():
        emb = model.encode_pocket(**inputs)

    return emb.squeeze(0)


def query_molecules(
    client: QdrantClient,
    query_vector: List[float],
    top_k: int = 10,
) -> List[Dict]:
    """
    Query Qdrant for molecules similar to the query vector.
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="molecule"))]
        ),
        timeout=30,
    )
    results = results.points

    return [
        {
            "score": hit.score,
            "smi": hit.payload.get("smi", ""),
            "name": hit.payload.get("name"),
            "source_id": hit.payload.get("source_id"),
            "num_atoms": hit.payload.get("num_atoms"),
        }
        for hit in results
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Query DrugCLIP embeddings in Qdrant for best binding molecules"
    )
    parser.add_argument(
        "--pockets", type=str, required=True, help="Path to pockets LMDB file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pt",
        help="Model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps). Auto-detects if not specified",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of molecules to return per pocket"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only process this many pockets (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON). If not specified, prints to stdout",
    )
    args = parser.parse_args()

    # connect to Qdrant
    print("Connecting to Qdrant...")
    client = get_qdrant_client()

    # verify collection exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Error: Collection '{COLLECTION_NAME}' not found.")
        print("Run 'uv run load-qdrant --molecules <path>' first to load molecules.")
        return

    # load model
    device_str = f" on {args.device}" if args.device else ""
    print(f"Loading DrugCLIP model from {args.checkpoint}{device_str}...")
    model = DrugCLIPModel.from_checkpoint(args.checkpoint, device=args.device)

    # load pockets
    pocket_data = load_lmdb(args.pockets, desc="Loading pockets", subset=args.subset)
    print(f"Loaded {len(pocket_data)} pockets")

    # query for each pocket
    all_results = {}
    for data in tqdm(pocket_data, desc="Querying molecules"):
        pocket_id = data.get("pocket", data.get("pdb", "unknown"))

        try:
            emb = encode_pocket(model, data)
            query_vector = emb.cpu().numpy().tolist()

            matches = query_molecules(client, query_vector, top_k=args.top_k)
            all_results[pocket_id] = {
                "pocket_id": pocket_id,
                "pdb": data.get("pdb"),
                "uniprot": data.get("uniprot"),
                "num_atoms": len(data.get("pocket_atoms", [])),
                "matches": matches,
            }

        except Exception as e:
            print(f"  Failed to query for pocket {pocket_id}: {e}")

    # output results
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        # print to stdout
        print("\n" + "=" * 60)
        for pocket_id, result in all_results.items():
            print(f"\nPocket: {pocket_id}")
            if result.get("pdb"):
                print(f"  PDB: {result['pdb']}")
            if result.get("uniprot"):
                print(f"  UniProt: {result['uniprot']}")
            print(f"  Atoms: {result['num_atoms']}")
            print(f"  Top {len(result['matches'])} matches:")
            for i, match in enumerate(result["matches"], 1):
                name = f" ({match['name']})" if match.get("name") else ""
                print(f"    {i}. {match['smi'][:50]}{name}")
                print(f"       Score: {match['score']:.4f}")


if __name__ == "__main__":
    main()