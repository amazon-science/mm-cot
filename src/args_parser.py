from src.constants import PromptFormat
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str,
                        default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list,
                        default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None,
                        help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train',
                        choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val',
                        choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str,
                        default='test', choices=['test', 'minitest'])

    parser.add_argument('--use_generate', action='store_true',
                        help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true',
                        help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline",
                        help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None,
                        choices=['detr', 'clip', 'resnet'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None,
                        help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None,
                        help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None,
                        help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str,
                        default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true',
                        help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default=PromptFormat.QUESTION_CONTEXT_OPTIONS_ANSWER.value, help='prompt format template',
                        choices=PromptFormat.get_values())
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    return args
