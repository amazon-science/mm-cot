import numpy as np
import torch
from transformers import T5Tokenizer

from src.models.t5_multimodal_generation.t5_mg_service import T5ForMultimodalGenerationService
from src.models.t5_multimodal_generation.t5_mg_training_params import get_training_data


class T5ForMultimodalGenerationOrchestrationService:

    def __init__(self, args, dataframe):
        if args.evaluate_dir is not None:
            args.model = args.evaluate_dir
        if args.prompt_format == "QCM-LE":
            torch.cuda.empty_cache()

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model)
        self.t5_model = T5ForMultimodalGenerationService(dataframe, args, self.tokenizer)
        self.args = args
        self.dataframe = dataframe

    def _orchestration_create_trainer(self):

        train_set, eval_set, test_set = get_training_data(
            self.args, self.dataframe, self.tokenizer)

        existing_model_dir = self.args.evaluate_dir
        if existing_model_dir is None:
            self.t5_model.fit(train_set, eval_set)
        else:
            self.t5_model.build_seq2seq_base_trainer(train_set, eval_set)

        self.t5_model.evaluate(test_set)

    def _orchestration_inference(self):
        if self.args.prompt_format == "QCM-LE":
            _, eval_set, _ = get_training_data(
                self.args, self.dataframe, self.tokenizer)
            self.t5_model.inference(eval_set)

    def run_pipeline(self, run_inference: bool = True):

        print("Orchestration | create trainer \n")
        self._orchestration_create_trainer()
        if run_inference:
            print("Orchestration | Generate rationale \n")
            self._orchestration_inference()
