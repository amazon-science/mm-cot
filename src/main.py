import json
import os

from rich import box
from rich.table import Column, Table

from src.args_parser import parse_args
from src.models.chain_of_thought import ChainOfThought

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

    chain_of_thought = ChainOfThought(args)
    chain_of_thought.load_data()
    chain_of_thought.load_model()
    chain_of_thought.evaluate()
