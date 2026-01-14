# drugclip 

A PyTorch implementation of DrugCLIP, from the paper: https://www.science.org/doi/10.1126/science.ads9530

## Installation:
```bash
pip install drugclip
```

## Usage:
```python
from drugclip import DrugCLIPModel, DrugCLIPConfig

# Load from original checkpoint
model = DrugCLIPModel.from_checkpoint("checkpoint_best.pt")

# Or load from HuggingFace format
model = DrugCLIPModel.from_pretrained("path/to/saved/model")

# Embed molecules from SMILES
emb = model.embed_smiles("CCO")
emb = model.embed_smiles(["CCO", "CC(=O)O"])  # batch

# Embed a pocket
emb = model.embed_pocket(atoms, coordinates)

# Save in HuggingFace format
model.save_pretrained("path/to/save")

# similarity score (higher = better binding)
score = mol_emb @ pocket_emb.T
```