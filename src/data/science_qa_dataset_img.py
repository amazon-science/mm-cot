import numpy as np
import torch

from src.data.science_qa_dataset_std import ScienceQADatasetStd

# TODO img_shape should not be here!
img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ScienceQADatasetImg(ScienceQADatasetStd):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self,
            problems,
            qids,
            tokenizer,
            source_len,
            target_len,
            args,
            test_le=None,
            image_features=None,
            name_maps=None,
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """

        super(ScienceQADatasetImg, self).__init__(
            problems,
            qids,
            tokenizer,
            source_len,
            target_len,
            args,
            test_le
        )

        self.image_ids = torch.tensor([]).to(device)
        for qid in self.data:

            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
            else:
                shape = img_shape[args.img_type]
                i_vectors = np.zeros(shape)
            i_vectors = torch.tensor(i_vectors).squeeze().to(device)

            self.image_ids = torch.cat(
                (self.image_ids, i_vectors.unsqueeze(0)), 0)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        return {
            "input_ids": self.source_ids[index].to(torch.long),
            "attention_mask": self.source_masks[index].to(torch.long),
            "image_ids": self.image_ids[index].to(torch.float),
            "labels": self.target_ids[index].to(torch.long).tolist(),
        }


