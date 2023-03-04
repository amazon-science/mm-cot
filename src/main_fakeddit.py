import json
import os

import pandas as pd
from rich import box
from rich.table import Column, Table
from transformers import T5Tokenizer

from src.args_parser import parse_args
from src import constants
from src.data.fakeddit.dataset import FakedditDataset
from src.models.chain_of_thought_fakeddit import ChainOfThought

if __name__ == '__main__':

    # import nltk
    # nltk.download('punkt')

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()

    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    dataframe = pd.read_csv(constants.FAKEDDIT_DATASET_PATH)
    tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model)
    test_set = FakedditDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_length=args.output_len
    )

    chain_of_thought = ChainOfThought(args) \
        .set_tokenizer(tokenizer) \
        .set_eval_set(test_set) \
        .load_model() \
        .evaluate()
    # chain_of_thought.evaluate()
