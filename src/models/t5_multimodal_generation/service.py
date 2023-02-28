import json
import os
import random

import evaluate
import numpy as np
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import T5Tokenizer

from src.data.science_qa_dataset_img import ScienceQADatasetIterator
from src.models.evaluation.evaluation import get_scores
from src.models.t5_multimodal_generation.training_params import get_t5_model, get_training_args
from src.models.t5_multimodal_generation.utils import extract_predictions_and_targets, extract_ans, \
    postprocess_text
from src.models.t5_multimodal_generation.utils import make_backup_dir
from transformers.trainer_utils import EvalLoopOutput


class T5ForMultimodalGenerationService:
    seq2seq_trainer = None

    def __init__(self, dataframe, args, tokenizer):
        self.args = args
        self.dataframe = dataframe
        self.save_dir = make_backup_dir(args)
        self.tokenizer = tokenizer or T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.model)

    def fit(self, train_set, eval_set):
        self.build_seq2seq_base_trainer(train_set, eval_set)
        self.seq2seq_trainer.train()
        self.seq2seq_trainer.save_model(self.save_dir)

    def build_seq2seq_base_trainer(self, train_set, eval_set):
        """
            Build a base seq2seq trainer.
            It is mandatory to run this method if t5 model isn't being trained
        """

        print(f"[Model]: Loading {self.args.model}...\n")
        print("[Data]: Reading data...\n")

        model = get_t5_model(self.args, self.tokenizer, self.save_dir)

        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        print("Model parameters: ", model.num_parameters())

        training_args = get_training_args(self.args, self.save_dir)

        self.seq2seq_trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=eval_set,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics_acc if self.args.prompt_format != "QCM-LE" else self.compute_metrics_rougel
        )

    def evaluate(self, test_set):
        """ Generate the answer for the eval set """

        self._seq2seq_existing_check()

        trainer = self.seq2seq_trainer

        metrics = trainer.evaluate(eval_dataset=test_set)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        predict_results = trainer.predict(
            test_dataset=test_set, max_length=self.args.output_len)

        if trainer.is_world_process_zero():
            preds, targets = extract_predictions_and_targets(
                predict_results, self.args.use_generate, self.tokenizer)
            results_ans = {}
            results_rationale = {}
            results_reference = {}
            num_fail = 0
            test_qids = self.dataframe['qids']['test']

            for idx, qid in enumerate(test_qids):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                extract_pred = extract_ans(pred)
                if extract_pred != "FAILED":
                    if extract_pred in self.args.options:
                        extract_pred = self.args.options.index(extract_pred)
                    else:
                        extract_pred = random.choice(
                            range(0, len(self.args.options)))
                else:
                    num_fail += 1
                    # random choose one option
                    extract_pred = random.choice(range(len(self.args.options)))
                results_ans[str(qid)] = extract_pred
                results_rationale[str(qid)] = pred
                results_reference[str(qid)] = ref

            scores = get_scores(results_ans, results_rationale, results_reference, os.path.join(
                self.args.data_root, "scienceqa/problems.json"))
            preds = [pred.strip() for pred in preds]

            output_data = {
                "num_fail": num_fail,
                "scores": scores,
                "preds": preds,
                "labels": targets
            }

            output_prediction_file = os.path.join(
                self.save_dir, "predictions_ans_test.json")

            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))

    def inference(self, eval_set):
        """ Generate the rationale for the eval set """

        self._seq2seq_existing_check()

        output_data = {
            "preds": [],
            "labels": []
        }

        for batch in ScienceQADatasetIterator(dataset=eval_set, batch_size=self.args.batch_size_in_memory):
            predict_results = self.seq2seq_trainer.predict(
                test_dataset=batch, max_length=self.args.output_len)

            if self.seq2seq_trainer.is_world_process_zero():
                predictions, targets = extract_predictions_and_targets(
                    predict_results, self.args, self.tokenizer)
                predictions = [pred.strip() for pred in predictions]

                output_data["preds"].extend(predictions)
                output_data["labels"].extend(targets)

        if self.seq2seq_trainer.is_world_process_zero():
            output_prediction_file = os.path.join(
                self.save_dir, "predictions_ans_eval.json")

            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(output_data, indent=4))

    def _seq2seq_existing_check(self):
        if not self.seq2seq_trainer:
            raise NotImplementedError(
                "ERROR T5000001 | Fit model or if model exists build a seq2seq trainer")
        return True

    def compute_metrics_rougel(self, eval_predictions):

        predictions, targets = extract_predictions_and_targets(
            eval_predictions, self.args, self.tokenizer)

        metric = evaluate.load("rouge")
        decoded_predictions, decoded_labels = postprocess_text(
            predictions, targets)

        result = metric.compute(predictions=decoded_predictions,
                                references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(
            pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def compute_metrics_acc(self, eval_predictions):
        """
        Accuracy for answer inference
        """

        predictions, targets = extract_predictions_and_targets(
            eval_predictions, self.args, self.tokenizer)
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
