import re

import evaluate
import nltk
import numpy as np
from transformers import T5Tokenizer


def postprocess_text(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return predictions, labels


def extract_prediciton_and_targets(eval_predictions, use_generate, tokenizer: T5Tokenizer):
    if use_generate:
        predictions, targets = eval_predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
    else:
        predictions = eval_predictions.predictions[0]
        targets = eval_predictions.label_ids
        predictions = predictions.argmax(axis=2)

    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    targets = tokenizer.batch_decode(
        targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return predictions, targets


def compute_metrics_rougel(eval_predictions, use_generate, tokenizer: T5Tokenizer):
    metric = evaluate.load("rouge")

    predictions, targets = extract_prediciton_and_targets(eval_predictions, use_generate, tokenizer)
    decoded_predictions, decoded_labels = postprocess_text(predictions, targets)

    result = metric.compute(predictions=decoded_predictions,
                            references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)

    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"
    return answer


def compute_metrics_acc(eval_predictions, use_generate, tokenizer: T5Tokenizer):
    """
    Accuracy for answer inference
    """

    predictions, targets = extract_prediciton_and_targets(eval_predictions, use_generate, tokenizer)
    correct = 0
    assert len(predictions) == len(targets)
    for idx, pred in enumerate(predictions):
        reference = targets[idx]
        reference = extract_ans(reference)
        extract_pred = extract_ans(pred)
        best_option = extract_pred
        if reference == best_option:
            correct += 1
    return {'Accuracy': 1.0 * correct / len(targets)}
