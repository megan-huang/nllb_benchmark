# data_prep.py
from datasets import load_dataset

def load_en_zh_test():
    raw = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
    def extract(x):
        return {"source": x["translation"]["en"], "ref": x["translation"]["zh"]}
    ds = raw.map(extract, remove_columns=["translation"])
    return ds["source"], ds["ref"]

def load_wmt20_qe_en_zh():
    """
    Load WMT20 MLQE Task1 English→Chinese test split,
    returning source, reference, and human DA scores.
    """
    qe = load_dataset("wmt/wmt20_mlqe_task1", "en-zh", split="test")
    srcs       = [ex["translation"]["en"] for ex in qe]
    refs       = [ex["translation"]["zh"] for ex in qe]
    human_mean = [ex["mean"] for ex in qe]    # raw DA score (0–100)
    human_z    = [ex["z_mean"] for ex in qe]  # z-normalized DA
    return srcs, refs, human_mean, human_z
