import lmdb
import pickle
import torch

from drugclip import (
    DrugCLIPModel,
    tokenize_molecule,
    tokenize_pocket,
    smiles_to_input,
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


def example_smiles_encoding(model: DrugCLIPModel):
    """
    Encode molecules from SMILES strings.
    """
    print("\nEncoding from SMILES:")

    smiles_list = [
        "CCO",      # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1", # benzene
    ]

    embeddings = []
    for smi in smiles_list:
        tokenized = smiles_to_input(smi, model.config.mol_dictionary)
        if tokenized is None:
            print(f"  Failed to process: {smi}")
            continue

        inputs = to_model_input(tokenized, device=model.device)
        with torch.no_grad():
            emb = model.encode_molecule(
                inputs["tokens"], inputs["distances"], inputs["edge_types"]
            )
        embeddings.append(emb)
        print(f"  {smi}: embedding shape {emb.shape}")

    if len(embeddings) > 1:
        # compute pairwise similarities
        all_emb = torch.cat(embeddings, dim=0)
        sim = all_emb @ all_emb.T
        print(f"\n  Pairwise similarities:\n{sim}")


def example_lmdb_molecule(model: DrugCLIPModel, lmdb_path: str = "data/mols.lmdb"):
    """
    Encode molecules from LMDB data.
    """
    print("\nEncoding molecules from LMDB:")

    try:
        data_list = load_lmdb(lmdb_path)
    except Exception as e:
        print(f"  Could not load {lmdb_path}: {e}")
        return

    print(f"  Loaded {len(data_list)} molecules")

    # encode first few
    for i, data in enumerate(data_list[:3]):
        tokenized = tokenize_molecule(data, model.config.mol_dictionary)
        inputs = to_model_input(tokenized, device=model.device)

        with torch.no_grad():
            emb = model.encode_molecule(
                inputs["tokens"], inputs["distances"], inputs["edge_types"]
            )

        smi = data.get("smi", "unknown")
        print(f"  [{i}] {smi[:40]}: {len(data['atoms'])} atoms, embedding shape {emb.shape}")


def example_lmdb_pocket(model: DrugCLIPModel, lmdb_path: str = "data/pocket.lmdb"):
    """
    Encode pockets from LMDB data.
    """
    print("\nEncoding pockets from LMDB:")

    try:
        data_list = load_lmdb(lmdb_path)
    except Exception as e:
        print(f"  Could not load {lmdb_path}: {e}")
        return

    print(f"  Loaded {len(data_list)} pockets")

    # encode first few
    for i, data in enumerate(data_list[:3]):
        tokenized = tokenize_pocket(data, model.config.pocket_dictionary)
        inputs = to_model_input(tokenized, device=model.device)

        with torch.no_grad():
            emb = model.encode_pocket(
                inputs["tokens"], inputs["distances"], inputs["edge_types"]
            )

        pocket_id = data.get("pocket", "unknown")
        n_atoms = len(data["pocket_atoms"])
        print(f"  [{i}] {pocket_id}: {n_atoms} atoms, embedding shape {emb.shape}")


def example_similarity_search(model: DrugCLIPModel):
    """
    Compute molecule-pocket similarity scores.
    """
    print("\nSimilarity search:")

    # encode a molecule from SMILES
    smi = "CC1CCc2nc(NC(=O)C(C)(C)C)sc2C1"
    mol_tokenized = smiles_to_input(smi, model.config.mol_dictionary)
    if mol_tokenized is None:
        print(f"  Failed to process SMILES: {smi}")
        return

    mol_inputs = to_model_input(mol_tokenized, device=model.device)
    with torch.no_grad():
        mol_emb = model.encode_molecule(**mol_inputs)

    # load pockets from LMDB
    try:
        pocket_list = load_lmdb("data/pocket.lmdb")
    except Exception as e:
        print(f"  Could not load pockets: {e}")
        return

    # encode all pockets
    pocket_embs = []
    pocket_ids = []
    for data in pocket_list:
        tokenized = tokenize_pocket(data, model.config.pocket_dictionary)
        inputs = to_model_input(tokenized, device=model.device)

        with torch.no_grad():
            emb = model.encode_pocket(**inputs)
        pocket_embs.append(emb)
        pocket_ids.append(data.get("pocket", "unknown"))

    if not pocket_embs:
        print("  No pockets to compare")
        return

    # compute similarities
    all_pocket_emb = torch.cat(pocket_embs, dim=0)
    scores = (mol_emb @ all_pocket_emb.T).squeeze(0)

    # rank by similarity
    sorted_idx = scores.argsort(descending=True)
    print(f"\n  Molecule: {smi}")
    print("  Top matches:")
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"    {rank+1}. {pocket_ids[idx]}: {scores[idx]:.4f}")


if __name__ == "__main__":
    print("Loading DrugCLIP model...")
    model = DrugCLIPModel.from_checkpoint("checkpoint_best.pt")

    example_smiles_encoding(model)
    example_lmdb_molecule(model)
    example_lmdb_pocket(model)
    example_similarity_search(model)