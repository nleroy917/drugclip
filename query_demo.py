import os
import pickle

import torch
import lmdb

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from drugclip import (
    DrugCLIPModel,
    tokenize_pocket,
    to_model_input,
)

def load_lmdb(path: str) -> list:
    """
    Load all entries from an LMDB file.
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
        for _, value in txn.cursor():
            data_list.append(pickle.loads(value))
    env.close()
    return data_list


qdrant = QdrantClient(
    url=os.environ.get("QDRANT_HOST", "http://localhost:6333"),
    api_key=os.environ.get("QDRANT_API_KEY", None),
)
collection = qdrant.get_collection(collection_name="drugclip")
print(f"Collection 'drugclip' with {collection.points_count} points.")
print(f"  Status: {collection.status}")


model = DrugCLIPModel.from_pretrained("nleroy917/drugclip")

pocket_data_path = "data/pocket.lmdb"
data_list = load_lmdb(pocket_data_path)
pocket = next(iter(data_list))

tokenized = tokenize_pocket(pocket, model.config.pocket_dictionary)
inputs = to_model_input(tokenized, device=model.device)

with torch.no_grad():
    emb = model.encode_pocket(
        inputs["tokens"], inputs["distances"], inputs["edge_types"]
    )

# remove batch dimension
emb = emb.squeeze(0)
results = qdrant.query_points(
    collection_name="drugclip",
    query=emb.cpu().numpy().tolist(),
    limit=10,
    query_filter=Filter(
        must=[FieldCondition(key="type", match=MatchValue(value="molecule"))]
    )
)
results = results.points

for point in results:
    metadata = point.payload
    print("-----")
    print(
        f"Score: {point.score:.4f}\t"
        f"SMILE: {metadata.get('smi', 'N/A')}, "
    )
