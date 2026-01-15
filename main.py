import lmdb
import pickle

from drugclip import DrugCLIPModel

# load from a checkpoint
model = DrugCLIPModel.from_checkpoint("checkpoint_best.pt")

# Open the LMDB environment (read-only)
env = lmdb.open(
    "data/mols.lmdb",
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)


cnt = env.stat()["entries"]
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        data = pickle.loads(value)
        data_list.append(data)


env.close()