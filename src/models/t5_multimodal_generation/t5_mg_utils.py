import os
import re

import nltk
import torch


def extract_predictions_and_targets(eval_predictions, args, tokenizer):

    if args.use_generate:
        predictions, targets = eval_predictions

        # TODO check if necessary
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


def extract_ans(ans):
    pattern = re.compile(r'The answer is \(([A-Z])\)')
    res = pattern.findall(ans)

    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer


def postprocess_text(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return predictions, labels


def make_backup_dir(args):
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    return save_dir
