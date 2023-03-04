from enum import Enum

# These have been taken by a comment https://github.com/entitize/Fakeddit/issues/14
# Check if it is truthful


class LabelsTypes(Enum):
    TWO_WAY = "TWO_WAY"
    THREE_WAY = "THREE_WAY"
    SIX_WAY = "SIX_WAY"


class TwoWayLabels(Enum):
    TRUE: 1
    FALSE: 0


class ThreeWayLabels(Enum):
    TRUE: 0
    FAKE_WITH_TRUE_TEXT: 1
    FAKE_WITH_FALSE_TEXT: 0


class SixWayLabels(Enum):
    TRUE: 0
    SATIRE: 1
    FALSE_CONNECTION: 2
    IMPOSTER_CONTENT: 3
    MANIPULATED_CONTENT: 4
    MISLEADING_CONTENT: 5


def get_options_text(labels_type: LabelsTypes) -> str:
    if labels_type == LabelsTypes.TWO_WAY:
        return "(A) True (B) False"
    if labels_type == LabelsTypes.THREE_WAY:
        return "(A) True (B) Fake with true text (C) Fake with false text"
    if labels_type == LabelsTypes.SIX_WAY:
        return "(A) True (B) Satire (C) False connection (D) Imposter content (E) Manipulated content (F) Misleading content"
