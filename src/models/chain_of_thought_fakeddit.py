import numpy as np
import torch
from transformers import T5Tokenizer
import random
from src.models.t5_multimodal_generation.service import T5ForMultimodalGenerationService
from src.constants import PromptFormat
from src.data.fakeddit.dataset import FakedditDataset


class ChainOfThought:

    def __init__(
        self,
        args
    ):
        self.args = args
        self._set_random_seed()
        self.dataframe = None
        self.train_set = None
        self.eval_set = None
        self.test_set = None
        self.t5_model = None
        self.tokenizer = None

    def _set_random_seed(self):
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def set_tokenizer(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        return self

    def set_train_set(self, train_set: FakedditDataset):
        self.train_set = train_set
        return self

    def set_validation_set(self, validation_set: FakedditDataset):
        self.validation_set = validation_set
        return self

    def set_test_set(self, test_set: FakedditDataset):
        self.test_set = test_set
        return self

    def load_model(self):
        if not self.tokenizer:
            raise AttributeError(
                "A tokenizer is required to load the model. Use set_tokenizer")
        self.t5_model = T5ForMultimodalGenerationService(self.dataframe,
                                                         self.args, self.tokenizer)

        # Here we shouldn't need the models
        run_training = self.args.evaluate_dir is None
        if run_training:
            self.t5_model.fit(self.train_set, self.eval_set)
        else:
            self.t5_model.build_seq2seq_base_trainer(
                self.train_set, self.eval_set)

    def evaluate(self):

        generate_answer = self.args.prompt_format in [
            PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION_ANSWER.value,
        ]

        generate_rationale = self.args.prompt_format in [
            PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION_ANSWER.value,
            PromptFormat.QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION.value,
        ]

        torch.cuda.empty_cache()

        if generate_answer:
            self.t5_model.evaluate(self.test_set)

        if generate_rationale:
            self.t5_model.inference(self.eval_set)
