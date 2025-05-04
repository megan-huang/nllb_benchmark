# metrics.py
import nltk
nltk.download('wordnet', quiet=True)

import sacrebleu
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from bert_score import score as bert_score
import torch
from tqdm.auto import tqdm

def compute_metrics(hyps, refs):
    sacre_refs = [refs]
    bleu = sacrebleu.corpus_bleu(hyps, sacre_refs)
    chrf = sacrebleu.corpus_chrf(hyps, sacre_refs)
    ter_char = TER(tokenize="char")
    ter = ter_char.corpus_score(hyps, sacre_refs) # fixing for chinese (no spaces in sentence)

    # split on whitespace
    meteor = sum(
        meteor_score([r.split()], h.split())
        for r, h in zip(refs, hyps)
    ) / len(hyps)
    P, R, F1 = bert_score(hyps, refs, lang="zh", rescale_with_baseline=True)
    bert = F1.mean().item()

    # all scores out of 100, 2 decimal places
    return {
        "BLEU": round(bleu.score, 2),
        "ChrF": round(chrf.score, 2),
        "TER": round(ter.score * 100, 2),
        "METEOR": round(meteor * 100, 2),
        "BERTScore": round(bert * 100, 2),
    }

def compute_sentence_scores(hyps, refs):
    # returns a dict mapping metric name to a list of scores (one per sentence)

    # sentence‐BLEU, ChrF, TER from sacrebleu
    sent_bleu =  [sacrebleu.sentence_bleu(h, [r]).score           for h,r in zip(hyps, refs)]
    sent_chrf =  [sacrebleu.sentence_chrf(h, [r]).score          for h,r in zip(hyps, refs)]
    ter_sent = TER(tokenize="char")
    sent_ter = [ter_sent.sentence_score(h, [r]).score * 100
                for h, r in zip(hyps, refs)]

    # sentence-METEOR (single_meteor_score expects token lists)
    sent_meteor = [
        single_meteor_score(r.split(), h.split()) * 100
        for h,r in zip(hyps, refs)
    ]

    # sentence-BERTScore
    _, _, F1 = bert_score(hyps, refs, lang="zh", rescale_with_baseline=True)
    sent_bertscore = (F1 * 100).tolist()

    return {
        "BLEU":      sent_bleu,
        "ChrF":      sent_chrf,
        "TER":       sent_ter,
        "METEOR":    sent_meteor,
        "BERTScore": sent_bertscore,
    }

def compute_back_translation_loss(
    model,
    tokenizer,
    src_texts,
    src_lang="eng_Latn",
    tgt_lang="zho_Hans",
    max_length=128,
):
    """
    Compute per-sentence back-translation reconstruction loss (cross-entropy).
    Returns a list of loss values (one per sentence).
    """
    device = next(model.parameters()).device
    model.eval()

    losses = []
    for sentence in tqdm(src_texts, desc="Back-translation"):  # A -> B
        # translate to target
        tokenizer.src_lang = src_lang
        inputs_A = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs_A,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length
            )
        back_trans = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

        # reconstruct to source
        tokenizer.src_lang = tgt_lang
        inputs_B = tokenizer(
            back_trans,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        # labels = original token ids
        tokenizer.src_lang = src_lang
        labels = inputs_A["input_ids"].to(device)

        # forward with labels to compute loss
        outputs = model(
            input_ids=inputs_B["input_ids"],
            attention_mask=inputs_B["attention_mask"],
            labels=labels,
        )
        losses.append(outputs.loss.item())

    return losses
