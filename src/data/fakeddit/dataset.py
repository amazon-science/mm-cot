
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms
from transformers import T5Tokenizer
from src import constants

from src.data.fakeddit.labels import LabelsTypes, get_options_text

# test_le: Probably it is the previously generated rationale, needed to inference the answer (so it will be null when)
# inferencing the rationale

# prompt: The question
#   e.i: "Question: What does the verbal irony in this text suggest?\nAccording to Mr. Herrera's kids, his snoring is as quiet as a jackhammer.\nContext: N/A\nOptions: (A) The snoring is loud. (B) The snoring occurs in bursts.\nSolution:"
# target: The rationale
#   e.i: "Solution: Figures of speech are words or phrases that use language in a nonliteral or unusual way. They can make writing more expressive.\\nVerbal irony involves saying one thing but implying something very different. People often use verbal irony when they are being sarcastic.\\nOlivia seems thrilled that her car keeps breaking down.\\nEach breakdown is as enjoyable as a punch to the face. The text uses verbal irony, which involves saying one thing but implying something very different.\\nAs quiet as a jackhammer suggests that the snoring is loud. A jackhammer is not quiet, and neither is Mr. Herrera's snoring.."

# source = self.process_data(prompt, self.source_len)
# source_id = source["input_ids"].squeeze().to(device)
# source_mask = source["attention_mask"].squeeze().to(device)
# target = target["input_ids"].squeeze().to(device)


# "input_ids": self.source_ids[index].to(torch.long),
# "attention_mask": self.source_masks[index].to(torch.long),
# "image_ids": self.image_ids[index].to(torch.float),
# "labels": self.target_ids[index].to(torch.long).tolist(),

DATASET_PATH = 'data/fakeddit/partial/dataset.csv'
DEFAULT_PROMPT = """Question: What adjective better describes the following fact?
\n<TEXT>
\nOptions: <OPTIONS>
\nSolution:"""


def load_data(args):
    return pd.read_csv(DATASET_PATH)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FakedditDataset(IterableDataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: T5Tokenizer,
        max_length: int = 512,
        labels_type: LabelsTypes = LabelsTypes.TWO_WAY,
        images_path: str = None
    ) -> None:

        self.labels_type = labels_type
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.input_ids = torch.tensor([], device=device)
        self.image_ids = torch.tensor([], device=device)
        self.attention_masks = torch.tensor([], device=device)
        self.labels = torch.tensor([], device=device)

        self._build_dataset()

    def _build_dataset(self) -> None:

        for row in self.dataframe.to_dict(orient="records")[:10]:
            _input_ids, _attention_mask = self.get_input_ids(row)
            _image_ids = self.get_image_ids(row)
            _labels = self.get_labels(row)

            self.input_ids = torch.cat(
                (self.input_ids, _input_ids.unsqueeze(0)), 0)
            self.image_ids = torch.cat(
                (self.image_ids, _image_ids.unsqueeze(0)), 0)
            self.attention_masks = torch.cat(
                (self.attention_masks, _attention_mask.unsqueeze(0)), 0)
            self.labels = torch.cat(
                (self.labels, _labels.unsqueeze(0)), 0)

    def get_input_ids(self, row: dict) -> Tuple[Tensor, Tensor]:

        clean_title = row["clean_title"]
        full_text = self._get_question_text(clean_title)
        processed = self.process_data(full_text)

        input_ids = processed["input_ids"].squeeze().to(device)
        attention_mask = processed["attention_mask"].squeeze().to(device)

        return input_ids, attention_mask

    def _get_question_text(self, clean_title: str) -> str:
        options_text = get_options_text(self.labels_type)

        question_text = DEFAULT_PROMPT.replace("<TEXT>", clean_title)
        question_text = question_text.replace("<OPTIONS>", options_text)

        return question_text

    def get_image_ids(self, row: dict) -> Tensor:
        image_id = row["id"]
        image_obj = Image.open(f"{constants.FAKEDDIT_IMG_DATASET_PATH}/{image_id}.jpg")
        img_to_tensor_transformer = transforms.Compose([transforms.PILToTensor()])
        image_tensor = img_to_tensor_transformer(image_obj)

        return image_tensor.to(device=device)

    def get_labels(self, row: dict) -> Tensor:
        return torch.zeros(256, device=device)

    def process_data(
            self,
            text
    ):
        text = " ".join(str(text).split())
        return self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.input_ids)

    def __getitem__(self, index) -> dict:
        return {
            "input_ids": self.input_ids[index].to(torch.long),
            "attention_mask": self.attention_masks[index].to(torch.long),
            "image_ids": self.image_ids[index].to(torch.float),
            "labels": self.labels[index].to(torch.long).tolist(),
        }
