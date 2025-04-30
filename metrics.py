# metrics.py
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

def compute_metrics(hyps, refs):
    sacre_refs = [refs]
    bleu = sacrebleu.corpus_bleu(hyps, sacre_refs)
    chrf = sacrebleu.corpus_chrf(hyps, sacre_refs)
    ter  = sacrebleu.corpus_ter(hyps, sacre_refs)

    meteor = sum(meteor_score([r], h) for r, h in zip(refs, hyps)) / len(hyps)
    P, R, F1 = bert_score(hyps, refs, lang="zh", rescale_with_baseline=True)
    bert = F1.mean().item()

    return {
        "BLEU": round(bleu.score, 2),
        "ChrF": round(chrf.score, 2),
        "TER":   round(ter.score * 100, 2),
        "METEOR": round(meteor * 100, 2),
        "BERTScore": round(bert * 100, 2),
    }
