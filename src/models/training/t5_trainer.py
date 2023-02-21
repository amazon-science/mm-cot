import json
import os
import random

import numpy as np
import torch
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer

from src.models.evaluation.evaluation import get_scores
from src.models.training.t5_training_metrics import compute_metrics_rougel, extract_ans, compute_metrics_acc
from src.models.training.t5_training_params import get_t5_model, get_training_data, get_training_args
from src.models.training.t5_training_utils import extract_predictions_and_targets, make_backup_dir


def T5Trainer(
        dataframe, args
):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model)

    print(f"[Model]: Loading {args.model}...\n")
    print("[Data]: Reading data...\n")

    save_dir = make_backup_dir(args)

    model = get_t5_model(args, tokenizer, save_dir)
    train_set, eval_set, test_set = get_training_data(
        args, dataframe, tokenizer)

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("Model parameters: ", model.num_parameters())

    training_args = get_training_args(args, save_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_acc if args.prompt_format != "QCM-LE" else compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)

    metrics = trainer.evaluate(eval_dataset=test_set)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(
        test_dataset=test_set, max_length=args.output_len)

    if trainer.is_world_process_zero():
        preds, targets = extract_predictions_and_targets(
            predict_results, args.use_generate, tokenizer)
        results_ans = {}
        results_rationale = {}
        results_reference = {}
        num_fail = 0
        test_qids = dataframe['qids']['test']

        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            extract_pred = extract_ans(pred)
            if extract_pred != "FAILED":
                if extract_pred in args.options:
                    extract_pred = args.options.index(extract_pred)
                else:
                    extract_pred = random.choice(range(0, len(args.options)))
            else:
                num_fail += 1
                # random choose one option
                extract_pred = random.choice(range(len(args.options)))
            results_ans[str(qid)] = extract_pred
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(
            args.data_root, "scienceqa/problems.json"))
        preds = [pred.strip() for pred in preds]

        output_data = {
            "num_fail": num_fail,
            "scores": scores,
            "preds": preds,
            "labels": targets}

        output_prediction_file = os.path.join(
            save_dir, "predictions_ans_test.json")

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))

    # generate the rationale for the eval set
    if args.prompt_format == "QCM-LE":
        torch.cuda.empty_cache()
        del predict_results, preds, targets
        predict_results = trainer.predict(
            test_dataset=eval_set, max_length=args.output_len)
        if trainer.is_world_process_zero():
            if args.use_generate:
                preds, targets = predict_results.predictions, predict_results.label_ids
            else:
                preds = predict_results.predictions[0]
                targets = predict_results.label_ids
                preds = preds.argmax(axis=2)

            preds = tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            targets = tokenizer.batch_decode(
                targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            output_data = {"preds": preds,
                           "labels": targets}
            output_prediction_file = os.path.join(
                save_dir, "predictions_ans_eval.json")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))
