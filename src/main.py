import json
import os

from rich import box
from rich.table import Column, Table

from src.args_parser import parse_args
from src.models.t5_mg_orchestration_service import T5ForMultimodalGenerationOrchestrationService

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

    t5_orchestration_service = T5ForMultimodalGenerationOrchestrationService(
        args)
    t5_orchestration_service.load_data()
    t5_orchestration_service.load_model()
    t5_orchestration_service.evaluate()
