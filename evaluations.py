'''
Adapted from https://github.com/lupantech/ScienceQA
'''

import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util

########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)

    avg_bleu = sum(bleus) / len(bleus)

    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def caculate_rouge(results, data):
    rouges = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


########################
## Sentence Similarity
########################
def similariry_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def caculate_similariry(results, data, model):
    scores = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()

        score = similariry_score(target, prediction, model)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score
