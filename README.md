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
```python
from drugclip import DrugCLIPModel, DrugCLIPConfig

# load from original checkpoint the authors provided
model = DrugCLIPModel.from_checkpoint("path/to/checkpoint_best.pt")

# or load from HuggingFace format
model = DrugCLIPModel.from_pretrained("path/to/saved/model")

# embed molecules from SMILES
emb = model.embed_smiles("CCO")
emb = model.embed_smiles(["CCO", "CC(=O)O"])  # batch

# embed a pocket
emb = model.embed_pocket(atoms, coordinates)

# save in HuggingFace format
model.save_pretrained("path/to/save")

# similarity score (higher = better binding)
score = mol_emb @ pocket_emb.T
```