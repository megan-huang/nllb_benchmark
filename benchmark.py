# benchmark.py
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from data_prep import load_en_zh_test
from metrics import compute_metrics

def run_benchmark(model_path, batch_size=16, max_length=128):
    src_texts, refs = load_en_zh_test()

    tokenizer = NllbTokenizerFast.from_pretrained(
        model_path, src_lang="eng_Latn", tgt_lang="zho_Hans"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda").eval()

    hyps = []
    for i in range(0, len(src_texts), batch_size):
        batch = src_texts[i : i + batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to("cuda")
        out = model.generate(**tok, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"])
        hyps += tokenizer.batch_decode(out, skip_special_tokens=True)

    results = compute_metrics(hyps, refs)
    for name, score in results.items():
        print(f"{name:10s}: {score}")

if __name__ == "__main__":
    run_benchmark("model/facebook-nllb-200-distilled-600M")
