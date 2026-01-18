> [!IMPORTANT]  
> I am still working on validating my implementation. The weights load correctly, and the forward pass is identical to the original implementation, but I have not yet been able to reproduce the results from the paper. If anyone is interested in collaborating on this, please reach out!

# drugclip

A PyTorch implementation of DrugCLIP, from the paper: https://www.science.org/doi/10.1126/science.ads9530

## Installation:
You can clone this repo and install this package with (preferably `uv`):
```bash
git clone git@github.com:nleroy917/drugclip.git
cd drugclip
uv pip install -e .
```

## Usage:

### Loading the model
```python
from drugclip import DrugCLIPModel

# load from original checkpoint the authors provided
model = DrugCLIPModel.from_checkpoint("checkpoint_best.pt")

# or load from HuggingFace format
model = DrugCLIPModel.from_pretrained("path/to/saved/model")

# save in HuggingFace format
model.save_pretrained("path/to/save")
```

### Tokenization API
Tokenization is separate from the model. The model only accepts pre-tokenized tensors.

| Function | Input | Output |
|----------|-------|--------|
| `tokenize_molecule(data, dict)` | LMDB mol format (`atoms`, `coordinates`) | `{tokens, distances, edge_types}` numpy |
| `tokenize_pocket(data, dict)` | LMDB pocket format (`pocket_atoms`, `pocket_coordinates`) | `{tokens, distances, edge_types}` numpy |
| `smiles_to_input(smiles, dict)` | SMILES string | `{tokens, distances, edge_types}` numpy |
| `to_model_input(tokenized, device)` | Tokenized dict | Batched tensors |

### Encoding molecules and pockets
```python
from drugclip import (
    DrugCLIPModel,
    tokenize_molecule,
    tokenize_pocket,
    smiles_to_input,
    to_model_input,
)

model = DrugCLIPModel.from_checkpoint("checkpoint_best.pt")

# from SMILES
tokenized = smiles_to_input("CCO", model.config.mol_dictionary)
inputs = to_model_input(tokenized, device=model.device)
mol_emb = model.encode_molecule(inputs["tokens"], inputs["distances"], inputs["edge_types"])

# from LMDB molecule data
mol_data = {"atoms": ["C", "C", "O"], "coordinates": coords_array}
tokenized = tokenize_molecule(mol_data, model.config.mol_dictionary)
inputs = to_model_input(tokenized, device=model.device)
mol_emb = model.encode_molecule(inputs["tokens"], inputs["distances"], inputs["edge_types"])

# from LMDB pocket data
pocket_data = {"pocket_atoms": ["N", "CA", "C", ...], "pocket_coordinates": [coord1, coord2, ...]}
tokenized = tokenize_pocket(pocket_data, model.config.pocket_dictionary)
inputs = to_model_input(tokenized, device=model.device)
pocket_emb = model.encode_pocket(inputs["tokens"], inputs["distances"], inputs["edge_types"])

# similarity score (higher = better binding)
score = mol_emb @ pocket_emb.T
```

### Reading from LMDB files
```python
import lmdb
import pickle

env = lmdb.open("data/mols.lmdb", subdir=False, readonly=True, lock=False)
with env.begin() as txn:
    for key, value in txn.cursor():
        data = pickle.loads(value)
        # data = {"atoms": [...], "coordinates": [...], "smi": "...", ...}
env.close()
```

See `example.py` for complete examples.
