import torch
import sys
import os

from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    Adafactor,
    )
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from flora_opt import Flora  # 你自定义的优化器
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

# === 实时监控显存、优化器和loss ===
class MonitorCallback(TrainerCallback):
    def __init__(self, writer):
        self.writer = writer

    def on_step_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)

        optimizer_name = "Unknown"
        if trainer is not None:
            # 有些优化器只有在 trainer.train() 执行中才会创建
            try:
                # create_optimizer 不是公共API，但 Trainer 内部依赖该方法创建优化器
                if trainer.optimizer is None:
                    trainer.create_optimizer()
                optimizer_name = type(trainer.optimizer).__name__
            except Exception as e:
                optimizer_name = f"Error({str(e)})"

        if torch.cuda.is_available():
            # 实时
            # allocated = torch.cuda.memory_allocated() / 1024**2
            # reserved = torch.cuda.memory_reserved() / 1024**2
            # 最大
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2    
            self.writer.add_scalar(f"memory_max_allocated_MB", max_allocated, state.global_step)
            self.writer.add_scalar(f"memory_max_reserved_MB", max_reserved, state.global_step)                    
            print(
                f"[监控] Step {state.global_step} | 显存: Allocated={max_allocated:.2f}MB, Reserved={max_reserved:.2f}MB | "
                f"优化器: {optimizer_name}"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.writer.add_scalar("train/loss", logs["loss"], state.global_step)
        if "eval_loss" in logs:
            self.writer.add_scalar("eval/loss", logs["eval_loss"], state.global_step)
        if "learning_rate" in logs:
            self.writer.add_scalar("train/learning_rate", logs["learning_rate"], state.global_step)

# === 可选优化器的Trainer ===
class CustomTrainer(Trainer):
    def __init__(self, *args, custom_optim=None, flora_rank=None, **kwargs):
        self.custom_optim = custom_optim
        self.flora_rank = flora_rank
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None:
            if self.custom_optim == "flora":
                self.optimizer = Flora(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    rank=self.flora_rank,  # 你自定义的参数
                )
            elif self.custom_optim == "adafactor":
                self.optimizer = Adafactor(
                    self.model.parameters(),
                    scale_parameter=True,
                    relative_step=False,
                    warmup_init=False,
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            elif self.custom_optim == "adamw":
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                )
            else:
                raise ValueError(f"未知优化器类型: {self.custom_optim}")
        return self.optimizer

# === 预处理 ===
def preprocess(dataset, tokenizer):
    lines = [line for line in dataset['text'] if line.strip()]
    encodings = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=512)
    return [{"input_ids": torch.tensor(ids, dtype=torch.long)} for ids in encodings["input_ids"]]

# === 主函数 ===
def train(model_name, base_output_dir, overwrite_output_dir, per_device_train_batch_size,
          num_train_epochs, save_steps, optim_type="flora", flora_rank=128, is_lora=False, lora_rank = 128 ):   
    if optim_type == "flora":
        output_dir = os.path.join(base_output_dir, f"{optim_type.capitalize()}{flora_rank}_NoGradAccumulation")
        tf_dir = f"./runs/{optim_type.capitalize()}{flora_rank}_NoGradAccumulation"
    else:
        output_dir = os.path.join(base_output_dir, f"{optim_type.capitalize()}_NoGradAccumulation")
        tf_dir = f"./runs/{optim_type.capitalize()}_NoGradAccumulation"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    val_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    train_dataset = preprocess(train_raw, tokenizer)
    eval_dateset = preprocess(val_raw, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 可选开启 LoRA 支持（如使用 PEFT）
    if is_lora:
        output_dir = output_dir + f"_Lora{lora_rank}"
        tf_dir = tf_dir + f"_Lora{lora_rank}"
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(r=lora_rank, lora_alpha=16, lora_dropout=0.1, task_type='CAUSAL_LM')
        model = get_peft_model(model, lora_config)

    writer = SummaryWriter(log_dir=tf_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dateset,
        callbacks=[MonitorCallback(writer)],
        custom_optim=optim_type,
        flora_rank=flora_rank
    )

    print(f"[启动训练] 当前使用优化器: {optim_type}")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    torch.cuda.empty_cache()

# === 启动 ===
if __name__ == "__main__":
    train(
        model_name="/mnt/self-define/Xinbz/models/llms/gpt2",
        base_output_dir="./output_cs",
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        save_steps=1000,
        optim_type="adafactor",  # 可选："adamw", "adafactor", "flora"
        flora_rank=8,  # 仅flora时需要指定flora_rank, 8, 32, 128, 256
        is_lora=True,
        lora_rank=32  # 仅is_lora=True时需要指定lora_rank 
    )
