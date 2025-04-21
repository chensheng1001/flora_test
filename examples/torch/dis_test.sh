python -m torch.distributed.launch --nproc_per_node 2 dp_causal_language_modeling.py \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--model_name_or_path openai-community/gpt2 \
--output_dir /workspace/flora/flora-opt/examples/torch/output \
--num_train_epochs 100 \
--per_device_train_batch_size 32 \
--eval_steps 100
