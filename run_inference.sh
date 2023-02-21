# rationale generation
CUDA_VISIBLE_DEVICES=0,1 python src/main.py \
  --model allenai/unifiedqa-t5-base \
  --user_msg rationale \
  --img_type detr \
  --bs 8 \
  --eval_bs 4 \
  --eval_acc 10 \
  --output_len 512 \
  --final_eval \
  --prompt_format QCM-LE \
  --evaluate_dir models/rationale \
  --data_root data/dataset \
  --caption_file data/dataset/captions.json \
  --evaluate_dir models/MM-CoT-UnifiedQA-base-Rationa

# answer inference
# ...