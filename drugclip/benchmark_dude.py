"""
Benchmark DrugCLIP on DUD-E dataset.

Evaluates virtual screening performance: can the model rank active compounds
higher than decoys?

Usage:
    uv run benchmark-dude data/data/protein/DUD-E/raw/all
    uv run benchmark-dude data/data/protein/DUD-E/raw/all --targets abl1 ace
    uv run benchmark-dude data/data/protein/DUD-E/raw/all --output results.csv

Metrics:
    - AUC-ROC: Area under ROC curve (0.5 = random, 1.0 = perfect)
    - EF1%: Enrichment Factor at 1% (how many times better than random)
    - BEDROC: Early recognition metric (emphasizes top rankings)
"""

import argparse

from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import pickle
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats

from .modeling_drugclip import (
    DrugCLIPModel,
    tokenize_molecule,
    tokenize_pocket,
    to_model_input,
)


def load_lmdb(path: str) -> List[Dict]:
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


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC."""
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    # Count positives and negatives
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute AUC using Mann-Whitney U statistic
    # AUC = P(score(positive) > score(negative))
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    # Use scipy for numerical stability
    statistic, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auc = statistic / (n_pos * n_neg)

    return auc


def compute_enrichment_factor(
    labels: np.ndarray, scores: np.ndarray, fraction: float = 0.01
) -> float:
    """
    Compute Enrichment Factor at given fraction.

    EF = (actives in top X%) / (expected actives if random)
    """
    n_total = len(labels)
    n_actives = np.sum(labels)
    n_top = max(1, int(n_total * fraction))

    # sort by score descending
    sorted_indices = np.argsort(-scores)
    top_labels = labels[sorted_indices[:n_top]]

    actives_in_top = np.sum(top_labels)
    expected_random = n_actives * fraction

    if expected_random == 0:
        return 0.0

    return actives_in_top / expected_random


def compute_bedroc(labels: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """
    Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    BEDROC emphasizes early recognition - finding actives at the top of the ranked list.
    alpha controls how much to emphasize early recognition (higher = more emphasis).
    """
    n = len(labels)
    n_actives = np.sum(labels)

    if n_actives == 0 or n_actives == n:
        return 0.0

    # sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    # find ranks of actives (1-indexed, normalized to [0, 1])
    active_ranks = np.where(sorted_labels == 1)[0]
    ri = (active_ranks + 1) / n  # normalized ranks

    # compute BEDROC components
    s = np.sum(np.exp(-alpha * ri))

    # random expectation
    ra = n_actives / n
    rand_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)

    ri_opt = np.arange(1, n_actives + 1) / n
    opt_sum = np.sum(np.exp(-alpha * ri_opt))

    # BEDROC
    if opt_sum - rand_sum == 0:
        return 0.0

    bedroc = (s - rand_sum) / (opt_sum - rand_sum)
    return max(0.0, min(1.0, bedroc))


def evaluate_target(
    model: DrugCLIPModel,
    pocket_data: Dict,
    mol_data_list: List[Dict],
) -> Tuple[float, float, float, int, int]:
    """
    Evaluate a single target.

    Returns: (auc, ef1, bedroc, n_actives, n_total)
    """
    # encode pocket
    pocket_tokenized = tokenize_pocket(pocket_data, model.config.pocket_dictionary)
    pocket_inputs = to_model_input(pocket_tokenized, device=model.device)

    with torch.no_grad():
        pocket_emb = model.encode_pocket(**pocket_inputs)

    # encode molecules and collect labels
    mol_embs = []
    labels = []
    failed = 0

    for mol_data in tqdm(mol_data_list, desc="  Encoding molecules", leave=False):
        try:
            mol_tokenized = tokenize_molecule(mol_data, model.config.mol_dictionary)
            mol_inputs = to_model_input(mol_tokenized, device=model.device)

            with torch.no_grad():
                mol_emb = model.encode_molecule(**mol_inputs)

            mol_embs.append(mol_emb)
            labels.append(mol_data.get("label", 0))
        except Exception:
            failed += 1

    if failed > 0:
        print(f"    (failed to encode {failed} molecules)")

    if len(mol_embs) == 0:
        return 0.5, 1.0, 0.0, 0, 0

    # compute similarity scores
    all_mol_embs = torch.cat(mol_embs, dim=0)
    scores = (pocket_emb @ all_mol_embs.T).squeeze(0).cpu().numpy()
    labels = np.array(labels)

    # compute metrics
    auc = compute_auc(labels, scores)
    ef1 = compute_enrichment_factor(labels, scores, fraction=0.01)
    bedroc = compute_bedroc(labels, scores, alpha=20.0)

    n_actives = int(np.sum(labels))
    n_total = len(labels)

    return auc, ef1, bedroc, n_actives, n_total


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DrugCLIP on DUD-E dataset"
    )
    parser.add_argument(
        "dude_path",
        type=str,
        help="Path to DUD-E 'all' directory containing target folders",
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
        help="Device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=None,
        help="Specific targets to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results",
    )
    args = parser.parse_args()

    dude_path = Path(args.dude_path)
    if not dude_path.exists():
        print(f"Error: {dude_path} does not exist")
        return

    # find all targets
    if args.targets:
        targets = args.targets
    else:
        targets = sorted([
            d.name for d in dude_path.iterdir()
            if d.is_dir() and (d / "pocket.lmdb").exists()
        ])

    print(f"Found {len(targets)} targets")

    # load model
    device_str = f" on {args.device}" if args.device else ""
    print(f"Loading DrugCLIP model from {args.checkpoint}{device_str}...")
    model = DrugCLIPModel.from_checkpoint(args.checkpoint, device=args.device)

    # evaluate each target
    results = []
    for target in tqdm(targets, desc="Evaluating targets"):
        target_path = dude_path / target
        pocket_path = target_path / "pocket.lmdb"
        mols_path = target_path / "mols.lmdb"

        if not pocket_path.exists() or not mols_path.exists():
            print(f"  Skipping {target}: missing lmdb files")
            continue

        # load data
        pocket_data_list = load_lmdb(str(pocket_path))
        if len(pocket_data_list) == 0:
            print(f"  Skipping {target}: no pocket data")
            continue
        pocket_data = pocket_data_list[0]

        mol_data_list = load_lmdb(str(mols_path))

        # evaluate
        auc, ef1, bedroc, n_actives, n_total = evaluate_target(
            model, pocket_data, mol_data_list
        )

        results.append({
            "target": target,
            "auc": auc,
            "ef1": ef1,
            "bedroc": bedroc,
            "n_actives": n_actives,
            "n_total": n_total,
        })

        tqdm.write(
            f"  {target}: AUC={auc:.3f}, EF1%={ef1:.1f}, BEDROC={bedroc:.3f} "
            f"({n_actives}/{n_total})"
        )

    # summary
    if results:
        aucs = [r["auc"] for r in results]
        ef1s = [r["ef1"] for r in results]
        bedrocs = [r["bedroc"] for r in results]

        print("\n" + "=" * 60)
        print("Summary across all targets:")
        print(f"  Mean AUC:    {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"  Mean EF1%:   {np.mean(ef1s):.1f} ± {np.std(ef1s):.1f}")
        print(f"  Mean BEDROC: {np.mean(bedrocs):.3f} ± {np.std(bedrocs):.3f}")
        print(f"  Targets evaluated: {len(results)}")

        # save to CSV if requested
        if args.output:
            import csv

            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

                # add summary row
                writer.writerow({
                    "target": "MEAN",
                    "auc": np.mean(aucs),
                    "ef1": np.mean(ef1s),
                    "bedroc": np.mean(bedrocs),
                    "n_actives": sum(r["n_actives"] for r in results),
                    "n_total": sum(r["n_total"] for r in results),
                })

            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()