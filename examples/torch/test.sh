python causal_language_modeling.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /workspace/flora/flora-opt/examples/torch/output3 \
    --gradient_accumulation_steps 4 \
    # --optimizer flora

