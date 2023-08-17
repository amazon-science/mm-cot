# base
CUDA_VISIBLE_DEVICES=0 python main_central.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCM-E \
    --output_dir experiments0620

CUDA_VISIBLE_DEVICES=0 python main_central.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 8 --epoch 20 --lr 8e-5 --output_len 64 \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments0620 \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr8e-05_bs8_op512_ep20/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-base_vit_QCM-E_lr8e-05_bs8_op512_ep20/predictions_ans_test.json

# large
# rationale generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4 --epoch 50 --lr 5e-5 --output_len 512 \
    --use_caption --use_generate --prompt_format QCM-E \
    --output_dir experiments

# answer inference
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_central.py \
    --data_root data/ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-large \
    --user_msg answer --img_type vit \
    --bs 4 --eval_bs 8 --epoch 50 --lr 5e-5 --output_len 64 \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir experiments \
    --eval_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_eval.json \
    --test_le experiments/rationale_declare-lab-flan-alpaca-large_vit_QCM-E_lr5e-05_bs8_op512_ep50/predictions_ans_test.json