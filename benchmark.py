# benchmark.py
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from data_prep import load_wmt20_qe_en_zh
from metrics import compute_metrics, compute_sentence_scores, compute_back_translation_loss
from scipy.stats import pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_translations(srcs, tokenizer, model, batch_size=16, max_length=128):
    # translations for list of source senetences
    hyps = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i : i + batch_size]
        tok = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        out = model.generate(
            **tok,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
            max_length=max_length
        )
        hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return hyps


# def run_correlation_benchmark(model_path, batch_size=16, max_length=128):
#     """
#     Load human-evaluated EN→ZH data, generate model translations,
#     compute sentence-level scores, and correlate with human z-scores.
#     """
#     # WMT20 QE human-evaluated dataset
#     srcs, refs, human_mean, human_z = load_wmt20_qe_en_zh()
#     tokenizer = NllbTokenizerFast.from_pretrained(
#         model, src_lang="eng_Latn", tgt_lang="zho_Hans"
#     )
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
#     hyps = generate_translations(srcs, tokenizer, model, batch_size, max_length)
#     sent_scores = compute_sentence_scores(hyps, refs)

#     print("Corpus-level automatic metrics:")
#     corpus_results = compute_metrics(hyps, refs)
#     for metric, score in corpus_results.items():
#         print(f"{metric:10s}: {score:.2f}")
    
#     print(f"\nCorrelation with human z-scores (n={len(human_z)}):")
#     for metric_name, scores in sent_scores.items():
#         r, p = pearsonr(scores, human_z)
#         print(f"{metric_name:10s} r = {r:.3f}, p = {p:.3e}")
    
#     # back‐translation reconstruction loss
#     bt_losses = compute_back_translation_loss(
#         model,
#         tokenizer,
#         srcs,
#         src_lang="eng_Latn",
#         tgt_lang="zho_Hans",
#         batch_size=batch_size,
#         max_length=max_length,
#     )

#     avg_bt = sum(bt_losses) / len(bt_losses)
#     print("\nBack-translation reconstruction loss (cross-entropy) per batch:")
#     print(bt_losses)
#     print(f"Average back-translation loss per batch: {avg_bt:.3f}")

#     # correlating loss with human DA z‐scores
#     r, p = pearsonr([-l for l in bt_losses], human_z[: len(bt_losses)])
#     print(f"BackTransLoss → r = {r:.3f}, p = {p:.3e}")


# if __name__ == "__main__":
#     run_correlation_benchmark("model")

def run_correlation_benchmark(model_path, batch_size=16, max_length=128):
    """
    Load human-evaluated EN→ZH data, generate model translations,
    compute corpus- and sentence-level metrics, back-translation loss,
    and correlate metrics with human z-scores.
    """
    srcs, refs, human_mean, human_z = load_wmt20_qe_en_zh()
    tokenizer = NllbTokenizerFast.from_pretrained(
        model_path, src_lang="eng_Latn", tgt_lang="zho_Hans"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()


    hyps = generate_translations(srcs, tokenizer, model, batch_size, max_length)

    # corpus-level metrics
    print("Corpus-level automatic metrics on QE dataset:")
    corpus_results = compute_metrics(hyps, refs)
    for metric, score in corpus_results.items():
        print(f"{metric:10s}: {score:.2f}")

    # sentence-level metrics and correlation
    sent_scores = compute_sentence_scores(hyps, refs)
    print(f"\nCorrelation with human z-scores (n={len(human_z)}):")
    for metric_name, scores in sent_scores.items():
        r, p = pearsonr(scores, human_z)
        print(f"{metric_name:10s} r = {r:.3f}, p = {p:.3e}")

    # back-translation per-sentence reconstruction loss
    bt_losses = compute_back_translation_loss(
        model,
        tokenizer,
        srcs,
        src_lang="eng_Latn",
        tgt_lang="zho_Hans",
        max_length=max_length,
    )
    print("\nBack-translation reconstruction loss (cross-entropy) per sentence:")
    print(bt_losses)
    avg_bt = sum(bt_losses) / len(bt_losses)
    print(f"Average back-translation loss per sentence: {avg_bt:.3f}")

    # correlate back-translation loss with human z-scores
    inv_bt = [-l for l in bt_losses]
    r_bt, p_bt = pearsonr(inv_bt, human_z)
    print(f"BackTransLoss     r = {r_bt:.3f}, p = {p_bt:.3e}")

if __name__ == "__main__":
    run_correlation_benchmark("model")

