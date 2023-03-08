from enum import Enum

import os
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
SRC_PATH = os.path.join(ROOT_PATH, "src")
DATA_PATH = os.path.join(ROOT_PATH, "data")
FAKEDDIT_DATASET_PATH = os.path.join(DATA_PATH, "fakeddit", "partial", "dataset.csv")
FAKEDDIT_IMG_DATASET_PATH = os.path.join(DATA_PATH, "fakeddit", "images")
FAKEDDIT_VISION_FEATURES = os.path.join(DATA_PATH, "fakeddit", "partial", "vision_features", "image_features.npy")
FAKEDDIT_VISION_FEATURES_CHECKPOINT = os.path.join(DATA_PATH, "checkpoint.txt")

class PromptFormat(Enum):
    """
    Possible values for the prompt format
    The template is:
    <INPUT_FORMAT>-<OUTPUT_FORMAT>
    """

    QUESTION_CONTEXT_OPTIONS_ANSWER = "QCM-A"
    QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION = "QCM-LE"
    QUESTION_CONTEXT_OPTIONS_SOLUTION_ANSWER = "QCMG-A"  # Does G stand for solution?
    QUESTION_CONTEXT_OPTIONS_LECTURE_SOLUTION_ANSWER = "QCM-LEA"
    QUESTION_CONTEXT_OPTIONS_ANSWER_LECTURE_SOLUTION = "QCM-ALE"

    @classmethod
    def get_values(cls):
        return [e.value for e in cls]
