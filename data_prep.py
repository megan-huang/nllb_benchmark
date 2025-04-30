# data_prep.py
from datasets import load_dataset

def load_en_zh_test():
    raw = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
    def extract(x):
        return {"source": x["translation"]["en"], "ref": x["translation"]["zh"]}
    ds = raw.map(extract, remove_columns=["translation"])
    return ds["source"], ds["ref"]
