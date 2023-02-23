import evaluate
import numpy as np
from transformers import T5Tokenizer

from src.models.t5_multimodal_generation.t5_mg_utils import extract_predictions_and_targets, extract_ans, postprocess_text


def compute_metrics_rougel(eval_predictions):

    predictions, targets = extract_predictions_and_targets(eval_predictions)

    metric = evaluate.load("rouge")
    decoded_predictions, decoded_labels = postprocess_text(
        predictions, targets)

    result = metric.compute(predictions=decoded_predictions,
                            references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def compute_metrics_acc(eval_predictions):
    """
    Accuracy for answer inference
    """

    predictions, targets = extract_predictions_and_targets(eval_predictions)
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
