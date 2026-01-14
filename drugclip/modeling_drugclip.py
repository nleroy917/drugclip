from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from unimol import UniMolEncoder, NonLinearHead
from .configuration_drugclip import DrugCLIPConfig


class DrugCLIPModel(PreTrainedModel):
    """
    DrugCLIP model for molecule-pocket binding prediction.

    Inherits from HuggingFace's PreTrainedModel for compatibility with
    the transformers ecosystem (from_pretrained, save_pretrained, etc.)
    """

    config_class = DrugCLIPConfig
    base_model_prefix = "drugclip"

    def __init__(self, config: DrugCLIPConfig):
        super().__init__(config)

        mol_config = config.get_mol_config()
        pocket_config = config.get_pocket_config()

        self.mol_model = UniMolEncoder(mol_config)
        self.pocket_model = UniMolEncoder(pocket_config)

        self.mol_project = NonLinearHead(
            config.hidden_size, config.projection_dim, "relu"
        )
        self.pocket_project = NonLinearHead(
            config.hidden_size, config.projection_dim, "relu"
        )

        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))

        # initialize weights
        self.post_init()

    def encode_molecule(
        self,
        tokens: torch.Tensor,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecule to projection_dim embedding."""
        hidden = self.mol_model(tokens, distances, edge_types)
        emb = self.mol_project(hidden[:, 0, :])  # CLS token
        return F.normalize(emb, p=2, dim=-1)

    def encode_pocket(
        self,
        tokens: torch.Tensor,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode protein pocket to projection_dim embedding.
        """
        hidden = self.pocket_model(tokens, distances, edge_types)
        emb = self.pocket_project(hidden[:, 0, :])  # CLS token
        return F.normalize(emb, p=2, dim=-1)

    def forward(
        self,
        mol_tokens: Optional[torch.Tensor] = None,
        mol_distances: Optional[torch.Tensor] = None,
        mol_edge_types: Optional[torch.Tensor] = None,
        pocket_tokens: Optional[torch.Tensor] = None,
        pocket_distances: Optional[torch.Tensor] = None,
        pocket_edge_types: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass for computing molecule and/or pocket embeddings.

        Returns embeddings that can be used for similarity computation.
        """
        mol_embeds = None
        pocket_embeds = None

        if mol_tokens is not None:
            mol_embeds = self.encode_molecule(mol_tokens, mol_distances, mol_edge_types)

        if pocket_tokens is not None:
            pocket_embeds = self.encode_pocket(pocket_tokens, pocket_distances, pocket_edge_types)

        if not return_dict:
            return (mol_embeds, pocket_embeds, self.logit_scale.exp())

        return {
            "mol_embeds": mol_embeds,
            "pocket_embeds": pocket_embeds,
            "logit_scale": self.logit_scale.exp(),
        }

    @torch.no_grad()
    def embed_smiles(self, smiles: Union[str, List[str]], device: str = None) -> np.ndarray:
        """
        Embed SMILES string(s) to projection_dim vectors.

        Args:
            smiles: Single SMILES or list of SMILES
            device: Device override

        Returns:
            [N, projection_dim] numpy array
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        device = device or self.device
        embeddings = []

        for smi in smiles:
            data = smiles_to_input(smi, self.config.mol_dictionary)
            if data is None:
                print(f"Warning: Failed to process {smi}")
                continue

            tokens = torch.LongTensor(data["tokens"]).unsqueeze(0).to(device)
            distances = torch.FloatTensor(data["distances"]).unsqueeze(0).to(device)
            edge_types = torch.LongTensor(data["edge_types"]).unsqueeze(0).to(device)

            emb = self.encode_molecule(tokens, distances, edge_types)
            embeddings.append(emb.cpu().numpy())

        return np.concatenate(embeddings, axis=0) if embeddings else np.array([])

    @torch.no_grad()
    def embed_pocket(self, atoms: List[str], coordinates: np.ndarray, device: str = None) -> np.ndarray:
        """
        Embed pocket to projection_dim vector.

        Args:
            atoms: List of atom symbols
            coordinates: [N, 3] array
            device: Device override

        Returns:
            [1, projection_dim] numpy array
        """
        device = device or self.device
        data = atoms_to_input(atoms, coordinates, self.config.pocket_dictionary)

        tokens = torch.LongTensor(data["tokens"]).unsqueeze(0).to(device)
        distances = torch.FloatTensor(data["distances"]).unsqueeze(0).to(device)
        edge_types = torch.LongTensor(data["edge_types"]).unsqueeze(0).to(device)

        emb = self.encode_pocket(tokens, distances, edge_types)
        return emb.cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        mol_dict_path: str = None,
        pocket_dict_path: str = None,
        device: str = None,
    ) -> "DrugCLIPModel":
        """
        Load DrugCLIP from original (unicore) checkpoint format.

        Args:
            checkpoint_path: Path to checkpoint_best.pt
            mol_dict_path: Path to dict_mol.txt
            pocket_dict_path: Path to dict_pkt.txt
            device: Device to load on

        Returns:
            DrugCLIPModel in eval mode
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = Path(checkpoint_path)
        base_dir = checkpoint_path.parent

        # Load checkpoint first to get vocab sizes
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # Get vocab sizes from checkpoint
        mol_vocab_size = state_dict["mol_model.embed_tokens.weight"].shape[0]
        pocket_vocab_size = state_dict["pocket_model.embed_tokens.weight"].shape[0]

        print(f"Checkpoint vocab sizes: mol={mol_vocab_size}, pocket={pocket_vocab_size}")

        # Find dictionary files
        if mol_dict_path is None:
            for p in [base_dir / "data" / "dict_mol.txt", Path("data/dict_mol.txt")]:
                if p.exists():
                    mol_dict_path = p
                    break

        if pocket_dict_path is None:
            for p in [base_dir / "data" / "dict_pkt.txt", Path("data/dict_pkt.txt")]:
                if p.exists():
                    pocket_dict_path = p
                    break

        # load dictionaries
        mol_dict = load_dictionary(mol_dict_path) if mol_dict_path else {}
        pocket_dict = load_dictionary(pocket_dict_path) if pocket_dict_path else {}

        # pad dictionaries if needed
        while len(mol_dict) < mol_vocab_size:
            mol_dict[f"[EXTRA_{len(mol_dict)}]"] = len(mol_dict)
        while len(pocket_dict) < pocket_vocab_size:
            pocket_dict[f"[EXTRA_{len(pocket_dict)}]"] = len(pocket_dict)

        # load model from config
        config = DrugCLIPConfig(
            mol_vocab_size=mol_vocab_size,
            pocket_vocab_size=pocket_vocab_size,
            mol_dictionary=mol_dict,
            pocket_dictionary=pocket_dict,
        )
        model = cls(config)

        # map and load weights
        new_state = {}
        skip_prefixes = [
            "classification_head", "cross_distance", "holo_distance", "fuse_project",
            "lm_head", "pair2coord", "dist_head"
        ]
        for key, value in state_dict.items():
            if not any(key.startswith(p) for p in skip_prefixes):
                new_state[key] = value

        missing, unexpected = model.load_state_dict(new_state, strict=False)

        if len(missing) > 0:
            print(f"Missing keys: {missing}")

        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")

        loaded = len(new_state) - len(unexpected)
        print(f"Loaded {loaded} weights, {len(missing)} missing, {len(unexpected)} unexpected")

        model.to(device).eval()
        return model


def load_dictionary(path: Union[str, Path]) -> Dict[str, int]:
    """
    Load atom type dictionary.
    """
    d = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            d[line.strip()] = idx
    return d


def smiles_to_input(smiles: str, dictionary: Dict[str, int], num_conf: int = 10) -> Optional[Dict]:
    """
    Convert SMILES to model input.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit required. Install with: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, maxAttempts=10000) == -1:
        return None

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception as e:
        print(f"Warning: MMFF optimization failed for {smiles}: {e}")
        return None

    mol = Chem.RemoveHs(mol)
    if mol.GetNumConformers() == 0:
        return None

    coords = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    return atoms_to_input(atoms, coords, dictionary)


def atoms_to_input(atoms: List[str], coords: np.ndarray, dictionary: Dict[str, int], max_atoms: int = 256) -> Dict:
    """
    Convert atoms/coords to model input tensors.

    Matches the original preprocessing:
    - Prepend [CLS] token, append [SEP] token
    - Add placeholder coordinates for [CLS] (centroid) and [SEP] (zeros)
    - Compute pairwise distances
    - Compute edge types: edge[i,j] = token[i] * vocab_size + token[j]
    """
    from scipy.spatial.distance import cdist

    vocab_size = len(dictionary)
    cls_idx = dictionary.get("[CLS]", 1)
    sep_idx = dictionary.get("[SEP]", 2)
    unk_idx = dictionary.get("[UNK]", 3)

    # truncate (leave room for [CLS] and [SEP])
    atoms = atoms[:max_atoms - 2]
    coords = coords[:max_atoms - 2]

    # tokenize: [CLS] + atoms + [SEP]
    tokens = [cls_idx] + [dictionary.get(a, unk_idx) for a in atoms] + [sep_idx]
    n = len(tokens)

    # coordinates: [CLS]=centroid, atoms, [SEP]=zeros
    # (original uses 0.0 for both special tokens, but centroid works better for CLS)
    centroid = coords.mean(axis=0, keepdims=True)
    zeros = np.zeros((1, 3), dtype=np.float32)
    coords = np.vstack([centroid, coords, zeros]).astype(np.float32)

    # pairwise distances
    distances = cdist(coords, coords).astype(np.float32)

    # edge types encode the pair of atom types
    edge_types = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            edge_types[i, j] = tokens[i] * vocab_size + tokens[j]

    return {"tokens": tokens, "distances": distances, "edge_types": edge_types}



# backward compatibility
DrugCLIP = DrugCLIPModel