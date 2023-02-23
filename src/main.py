import json
import os
import random

from rich import box
from rich.table import Column, Table

from src.args_parser import parse_args
from src.data.data import load_data_std, load_data_img
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

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(
            args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids,
                     'name_maps': name_maps, 'image_features': image_features}
    else:
        # probelms, test question ids, shot example ids
        problems, qids = load_data_std(args)
        dataframe = {'problems': problems, 'qids': qids}

    t5_orchestration_service = T5ForMultimodalGenerationOrchestrationService(args, dataframe)
    t5_orchestration_service.run_pipeline(run_inference=True)
