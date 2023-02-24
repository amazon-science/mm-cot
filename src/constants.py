from enum import Enum


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
