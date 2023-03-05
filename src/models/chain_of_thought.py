import numpy as np
import torch
from transformers import T5Tokenizer
import random
from src.models.t5_multimodal_generation.service import T5ForMultimodalGenerationService
from src.models.t5_multimodal_generation.training_params import get_training_data
from src.data.data import load_data_std, load_data_img
from src.constants import PromptFormat

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

    def load_data(self):
        if self.args.img_type is not None:
            problems, qids, name_maps, image_features = load_data_img(
                self.args)  # probelms, test question ids, shot example ids
            dataframe = {'problems': problems, 'qids': qids,
                         'name_maps': name_maps, 'image_features': image_features}
        else:
            # probelms, test question ids, shot example ids
            problems, qids = load_data_std(self.args)
            dataframe = {'problems': problems, 'qids': qids}

        self.dataframe = dataframe

    def load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.model)
        self.t5_model = T5ForMultimodalGenerationService(self.dataframe,
                                                         self.args, self.tokenizer)

        self.train_set, self.eval_set, self.test_set = get_training_data(
            self.args, self.dataframe, self.tokenizer)

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
