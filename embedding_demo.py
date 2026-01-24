import torch

from drugclip import (
    DrugCLIPModel,
    smiles_to_input,
    to_model_input,
)

model = DrugCLIPModel.from_pretrained("nleroy917/drugclip")

smiles =  "CC(=O)O"   # acetic acid
tokenized = smiles_to_input(smiles, model.config.mol_dictionary)

inputs = to_model_input(tokenized, device=model.device)
with torch.no_grad():
    emb = model.encode_molecule(
        inputs["tokens"], inputs["distances"], inputs["edge_types"]
    )

print(emb)